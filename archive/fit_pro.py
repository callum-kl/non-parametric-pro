"""
PRO GP sampling, fitting a fresh kernel from scratch via cross-validation.

Same overall structure as ``fit_pro_cv.py`` (cross-validated adaptation via
``cross_validated_parameter_adaptation``), but ``initial_kernel`` is a brand-new RBF with
a default-initialised lengthscale rather than the kernel loaded from ``fit_vgp.py``/
``fit_exact_gp.py`` state -- only that state's ``sigma``/inducing points/scalers are
reused, not its fitted kernel.
"""

import json
import logging
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import hydra
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px
from omegaconf import DictConfig, OmegaConf

from non_parametric_pro import ula
from non_parametric_pro.parameter_adaptation import (
    cross_validated_parameter_adaptation,
)
from non_parametric_pro.data.uci.uci import load_uci_regression_dataset
from non_parametric_pro.density import ProParameters, pro_logdensity_fn, regularised_score
from non_parametric_pro.inducing import compute_inducing_basis
from non_parametric_pro.sgld import parametric_sgld, sgld
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import crps_pro, nlpd_pro, prediction_basis, run_inference_algorithm_with_burn_in, cholesky_basis

from util import load_gp_state

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)

def pro_out_dir(cfg: DictConfig) -> Path:
    subdir = "inducing_pro_gp_scratch_cv" if cfg.inducing else "pro_gp_scratch_cv"
    if cfg.name:
        subdir = f"{subdir}_{cfg.name}"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / subdir


def _adaptation_algorithm(cfg: DictConfig):
    """`algorithm` argument for `cross_validated_parameter_adaptation` -- `ula` (exact,
    full-batch) or `sgld(batch_size=...)` (minibatched; see `non_parametric_pro.sgld`).
    Both satisfy the same `build_kernel`/`init`/`refresh` duck-type, so this is the only
    place that needs to branch."""
    if cfg.algorithm == "ula":
        return ula
    if cfg.algorithm == "sgld":
        return sgld(batch_size=cfg.sgld_batch_size)
    msg = f"Unknown algorithm={cfg.algorithm!r}; expected 'ula' or 'sgld'."
    raise ValueError(msg)


def _sampling_algorithm(cfg: DictConfig, pro_params: ProParameters):
    """Standalone sampler for the final post-adaptation draw, mirroring
    `_adaptation_algorithm`'s choice of `ula` vs `sgld`."""
    if cfg.algorithm == "ula":
        return parametric_ula(pro_logdensity_fn, pro_params)
    if cfg.algorithm == "sgld":
        return parametric_sgld(pro_logdensity_fn, pro_params, batch_size=cfg.sgld_batch_size)
    msg = f"Unknown algorithm={cfg.algorithm!r}; expected 'ula' or 'sgld'."
    raise ValueError(msg)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_pro")
def main(cfg: DictConfig) -> None:
    mode = "inducing" if cfg.inducing else "exact GP"
    log.info(
        "PRO-scratch-CV (%s): dataset=%s split=%d folds=%d",
        mode, cfg.dataset, cfg.split, cfg.num_folds,
    )

    key = jr.PRNGKey(cfg.seed)
    out_dir = pro_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load GP state -------------------------------------------------------
    _, gp_sigma_val, inducing_basis, scaler_x, scaler_y = load_gp_state(cfg)

    # --- Data ------------------------------------------------------------------
    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test)

    log.info("N_train=%d  N_test=%d", x_train.shape[0], x_test.shape[0])

    # --- PRO setup ---------------------------------------------------------------
    sigma = gpx.parameters.SigmoidBounded(0.4, low=cfg.sigma_min, high=1.0)
    pro_params = ProParameters(
        y=y_train,
        basis=None,
        step_size=cfg.step_size,
        sigma=sigma,
        alpha=cfg.alpha,
        residual_std=None,
    )

    D = x_train.shape[1]
    init_lengthscale = jnp.sqrt(D) * jnp.ones((D,))
    lengthscale = gpx.parameters.SigmoidBounded(
        init_lengthscale, low=cfg.lengthscale_min, high=cfg.lengthscale_max
    )
    kernel = gpx.kernels.RBF(lengthscale=lengthscale, variance=px.NonTrainable(jnp.array(1.0)))

    # --- Cross-validated adaptation ---------------------------------------------
    log.info(
        "Running cross-validated adaptation (folds=%d, steps/fold=%d)...",
        cfg.num_folds, cfg.num_adapt_steps,
    )
    key, cv_key = jr.split(key)
    cv_result = cross_validated_parameter_adaptation(
        _adaptation_algorithm(cfg),
        pro_logdensity_fn,
        pro_params,
        x_full=x_train,
        y_full=y_train,
        initial_kernel=kernel,
        num_folds=cfg.num_folds,
        val_fraction=cfg.val_fraction,
        num_particles=cfg.num_particles,
        num_steps=cfg.num_adapt_steps,
        warmup_steps=cfg.warmup_steps,
        sigma_adapt_steps=cfg.sigma_adapt_steps,
        kernel_adapt_steps=cfg.kernel_adapt_steps,
        objective_fn=regularised_score,
        rng_key=cv_key,
        inducing_basis=inducing_basis,
        adapt_target=cfg.adapt_target,
        sigma_optimizer=ox.adam(cfg.sigma_lr),
        kernel_optimizer=ox.adam(cfg.kernel_lr),
        progress_bar=True,
    )
    adapted_kernel = cv_result.kernel
    adapted_sigma_val = float(np.array(cv_result.sigma).reshape(()))
    log.info(
        "CV sigma: min=%.4f  per-fold=%s", adapted_sigma_val, np.array(cv_result.fold_sigma)
    )

    # --- Recompute basis on the full training set with the CV-averaged kernel --
    log.info("Recomputing basis on full training set with cross-validated kernel...")
    if cfg.inducing:
        basis_full, residual_std_full = compute_inducing_basis(
            inducing_basis, adapted_kernel, x_train
        )
        basis_dim = inducing_basis.output_dim()
    else:
        basis_full = cholesky_basis(adapted_kernel, x_train)
        residual_std_full = None
        basis_dim = x_train.shape[0]

    key, pos_key = jr.split(key)
    pro_position = jr.normal(pos_key, (basis_dim, cfg.num_particles))

    sigma = gpx.parameters.SigmoidBounded(adapted_sigma_val, low=cfg.sigma_min, high=1.0)
    pro_params = pro_params._replace(
        basis=basis_full,
        residual_std=residual_std_full,
        sigma=sigma,
    )

    # --- Sampling ------------------------------------------------------------
    log.info("Running sampling (steps=%d, algorithm=%s)...", cfg.num_sample_steps, cfg.algorithm)
    algorithm = _sampling_algorithm(cfg, pro_params)

    key, sample_key = jr.split(key)
    _, (states, _) = run_inference_algorithm_with_burn_in(
        rng_key=sample_key,
        inference_algorithm=algorithm,
        num_steps=cfg.num_sample_steps,
        burn_ratio=cfg.burn_fraction,
        initial_position=pro_position,
        progress_bar=True,
    )
    particles = states.position[::cfg.thin]

    # --- Evaluation ------------------------------------------------------------
    test_basis, test_cov = prediction_basis(
        adapted_kernel, x_train, x_test, pro_params, inducing_basis=inducing_basis
    )
    pro_sigma = float(np.array(px.unwrap(pro_params.sigma)).reshape(()))
    metrics = {
        "dataset": cfg.dataset,
        "split": cfg.split,
        "inducing": cfg.inducing,
        "num_folds": cfg.num_folds,
        "gp_sigma": float(gp_sigma_val),
        "pro_sigma": pro_sigma,
        "pro_sigma_per_fold": np.array(cv_result.fold_sigma).tolist(),
        "pro_nlpd": float(nlpd_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
        "pro_crps": float(crps_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
    }
    log.info("PRO-scratch-CV  NLPD=%.4f  CRPS=%.4f", metrics["pro_nlpd"], metrics["pro_crps"])

    # --- Save ----------------------------------------------------------------
    with open(out_dir / "pro_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "pro_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
