"""PRO GP sampling on a PeMS road-network split, seeded from an exact graph GP."""

import json
import logging
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px
from omegaconf import DictConfig, OmegaConf
from util import load_gp_state, pro_out_dir

from non_parametric_pro import replica_gibbs, ula
from non_parametric_pro.data.pems.pems import load_pems_graph_data, pems_regression_split
from non_parametric_pro.density import ProParameters, pro_logdensity_fn
from non_parametric_pro.inducing import PointInducingBasis
from non_parametric_pro.parameter_adaptation import parameter_adaptation
from non_parametric_pro.replica_gibbs import parametric_replica_gibbs, validate_r
from non_parametric_pro.sgld import parametric_sgld, sgld
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import (
    cholesky_basis,
    nlpd_pro,
    pit_values,
    posterior_function_draws,
    prediction_basis,
    run_inference_algorithm_with_burn_in,
    train_val_split,
    variogram_score,
)

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


def _check_replica_gibbs_config(cfg: DictConfig) -> None:
    validate_r(cfg.num_particles, cfg.alpha)


def _adaptation_algorithm(cfg: DictConfig):
    if cfg.algorithm == "ula":
        return ula
    if cfg.algorithm == "sgld":
        return sgld(batch_size=cfg.sgld_batch_size)
    if cfg.algorithm == "replica_gibbs":
        _check_replica_gibbs_config(cfg)
        return replica_gibbs
    msg = f"Unknown algorithm={cfg.algorithm!r}; expected 'ula', 'sgld', or 'replica_gibbs'."
    raise ValueError(msg)


def _sampling_algorithm(cfg: DictConfig, pro_params: ProParameters):
    if cfg.algorithm == "ula":
        return parametric_ula(pro_logdensity_fn, pro_params)
    if cfg.algorithm == "sgld":
        return parametric_sgld(
            pro_logdensity_fn, pro_params, batch_size=cfg.sgld_batch_size
        )
    if cfg.algorithm == "replica_gibbs":
        _check_replica_gibbs_config(cfg)
        return parametric_replica_gibbs(pro_logdensity_fn, pro_params)
    msg = f"Unknown algorithm={cfg.algorithm!r}; expected 'ula', 'sgld', or 'replica_gibbs'."
    raise ValueError(msg)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_pro")
def main(cfg: DictConfig) -> None:
    log.info("PRO (graph GP): split=%d num_train=%d", cfg.split, cfg.num_train)

    if cfg.algorithm == "replica_gibbs":
        _check_replica_gibbs_config(cfg)

    key = jr.PRNGKey(cfg.seed)
    out_dir = pro_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load GP state -- fixed kernel, never adapted here -------------------
    kernel, gp_sigma_val, scaler_y = load_gp_state(cfg)

    # --- Data ------------------------------------------------------------------
    graph_data = load_pems_graph_data()
    if cfg.default_kernel_init:
        kernel = gpx.kernels.GraphKernel(
            laplacian=graph_data.laplacian,
            lengthscale=jnp.array(5.0),
            variance=jnp.array(1.0),
            smoothness=jnp.array(3.0),
        )
        log.info("default_kernel_init=true -- ignoring fitted GP kernel.")
    sigma_init_val = gp_sigma_val if cfg.sigma_init is None else cfg.sigma_init
    log.info(
        "Loaded exact GP state: kernel fixed, gp_sigma=%.4f, sigma starts at %.4f",
        gp_sigma_val,
        sigma_init_val,
    )

    split = pems_regression_split(
        graph_data, cfg.split, num_train=cfg.num_train, seed=cfg.split + cfg.split_seed_offset
    )
    x_train = jnp.asarray(split.x_train)
    y_train = jnp.asarray(scaler_y.transform(split.y_train))
    x_test = jnp.asarray(split.x_test)
    y_test = jnp.asarray(scaler_y.transform(split.y_test))

    log.info("N_train=%d  N_test=%d", x_train.shape[0], x_test.shape[0])

    # --- Full basis, from *all* training points -------------------------------
    basis_full = cholesky_basis(kernel, x_train)
    basis_dim = basis_full.shape[1]

    log.info(
        "Splitting train/val (val_fraction=%.2f) and adapting sigma...",
        cfg.val_fraction,
    )
    key, split_key, pos_key, adapt_key = jr.split(key, 4)
    tv_split = train_val_split(
        split_key, x_train, y_train, val_fraction=cfg.val_fraction
    )

    fold_params = ProParameters(
        y=tv_split.y_train,
        basis=None,
        step_size=cfg.step_size,
        sigma=gpx.parameters.SigmoidBounded(
            sigma_init_val, low=cfg.sigma_min, high=cfg.sigma_max
        ),
        alpha=cfg.alpha,
        residual_std=None,
    )
    initial_position = jr.normal(pos_key, (basis_dim, cfg.num_particles))

    adaptation = parameter_adaptation(
        _adaptation_algorithm(cfg),
        pro_logdensity_fn,
        fold_params,
        x_train=tv_split.x_train,
        initial_kernel=kernel,
        warmup_steps=cfg.warmup_steps,
        sigma_adapt_steps=cfg.sigma_adapt_steps,
        kernel_adapt_steps=cfg.kernel_adapt_steps,
        objective_fn=pro_logdensity_fn,
        inducing_basis=PointInducingBasis(z=x_train),
        x_val=tv_split.x_val,
        y_val=tv_split.y_val,
        adapt_target=cfg.adapt_target,
        sigma_optimizer=ox.adam(cfg.sigma_lr),
        kernel_optimizer=ox.adam(cfg.kernel_lr),
        progress_bar=True,
    )
    adaptation_results, adaptation_info = adaptation.run(
        adapt_key, initial_position, num_steps=cfg.num_adapt_steps
    )
    adapted_sigma_val = float(np.array(px.unwrap(adaptation_results.parameters.sigma)))
    log.info("Adapted sigma=%.4f", adapted_sigma_val)

    if cfg.kernel_adapt_steps > 0:
        adapted_kernel = px.unwrap(
            jax.tree.map(lambda x: x[-1], adaptation_info.kernel)
        )
        log.info(
            "Kernel was adapted (kernel_adapt_steps=%d) -- recomputing full basis...",
            cfg.kernel_adapt_steps,
        )
        basis_full = cholesky_basis(adapted_kernel, x_train)
    else:
        adapted_kernel = kernel

    # --- Final run: same full basis, particles warm-started from adaptation's final
    # position -- but now against the *full* training set's likelihood (train + val
    # combined), so a real burn-in is warranted since the likelihood scope just widened.
    log.info(
        "Running final sampling on full training set (steps=%d, algorithm=%s)...",
        cfg.num_sample_steps,
        cfg.algorithm,
    )
    pro_params = ProParameters(
        y=y_train,
        basis=basis_full,
        step_size=cfg.step_size,
        sigma=adaptation_results.parameters.sigma,
        alpha=cfg.alpha,
        residual_std=None,
    )
    algorithm = _sampling_algorithm(cfg, pro_params)
    key, sample_key = jr.split(key)
    _, (states, _) = run_inference_algorithm_with_burn_in(
        rng_key=sample_key,
        inference_algorithm=algorithm,
        num_steps=cfg.num_sample_steps,
        burn_ratio=cfg.burn_fraction,
        initial_position=adaptation_results.state.position,
        progress_bar=True,
    )
    particles = states.position[:: cfg.thin]

    # --- Evaluation ------------------------------------------------------------
    test_basis, test_cov = prediction_basis(adapted_kernel, x_train, x_test, pro_params)
    sigma_val = px.unwrap(pro_params.sigma)
    y_test_flat = jnp.asarray(y_test).squeeze()

    key, draw_key, noise_key = jr.split(key, 3)
    latent_draws = posterior_function_draws(
        draw_key, test_basis, test_cov, particles, num_draws=cfg.num_predictive_draws
    )
    y_draws = latent_draws + sigma_val * jr.normal(noise_key, latent_draws.shape)
    pro_variogram = float(variogram_score(y_draws, y_test_flat))
    pro_pit = pit_values(y_draws, y_test_flat)

    metrics = {
        "split": cfg.split,
        "num_train": cfg.num_train,
        "pro_nlpd": float(
            nlpd_pro(y_test, test_basis, test_cov, particles, parameters=pro_params)
        ),
        "pro_variogram_score": pro_variogram,
        "pro_pit": [float(v) for v in pro_pit],
    }
    log.info(
        "PRO  NLPD=%.4f  variogram=%.4f",
        metrics["pro_nlpd"],
        pro_variogram,
    )

    # --- Save ----------------------------------------------------------------
    with open(out_dir / "pro_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "pro_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
