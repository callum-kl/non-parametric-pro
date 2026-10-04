"""PRO-GP with replica Gibbs, seeded from a saved exact GP or sparse GP depending on cfg.inducing."""

import json
import logging
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import hydra
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px
from omegaconf import DictConfig, OmegaConf
from util import load_gp_state, pro_out_dir

from non_parametric_pro import replica_gibbs
from non_parametric_pro.data.uci.uci import load_uci_regression_dataset
from non_parametric_pro.density import ProParameters, pro_logdensity_fn
from non_parametric_pro.inducing import PointInducingBasis, compute_inducing_basis
from non_parametric_pro.parameter_adaptation import parameter_adaptation
from non_parametric_pro.replica_gibbs import parametric_replica_gibbs, validate_r
from non_parametric_pro.util import (
    cholesky_basis,
    nlpd_pro,
    prediction_basis,
    run_inference_algorithm_with_burn_in,
    train_val_split,
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_pro")
def main(cfg: DictConfig) -> None:
    mode = "inducing" if cfg.inducing else "exact GP"
    log.info("PRO (%s): dataset=%s split=%d", mode, cfg.dataset, cfg.split)
    validate_r(cfg.num_particles, cfg.alpha)

    key = jr.PRNGKey(cfg.seed)
    out_dir = pro_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel, gp_sigma_val, inducing_basis, scaler_x, scaler_y = load_gp_state(cfg)

    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test)
    log.info("N_train=%d  N_test=%d", x_train.shape[0], x_test.shape[0])

    if cfg.inducing:
        basis_full, residual_std_full = compute_inducing_basis(
            inducing_basis, kernel, x_train
        )
        row_selectable_basis = inducing_basis
    else:
        basis_full = cholesky_basis(kernel, x_train)
        residual_std_full = None
        row_selectable_basis = PointInducingBasis(z=x_train)

    key, split_key, pos_key, adapt_key = jr.split(key, 4)
    split = train_val_split(split_key, x_train, y_train, val_fraction=cfg.val_fraction)

    fold_params = ProParameters(
        y=split.y_train,
        basis=None,
        step_size=None,
        sigma=gpx.parameters.SigmoidBounded(
            cfg.sigma_init, low=cfg.sigma_min, high=cfg.sigma_max
        ),
        alpha=cfg.alpha,
        residual_std=None,
    )
    initial_position = jr.normal(pos_key, (basis_full.shape[1], cfg.num_particles))

    adaptation = parameter_adaptation(
        replica_gibbs,
        pro_logdensity_fn,
        fold_params,
        x_train=split.x_train,
        initial_kernel=kernel,
        warmup_steps=cfg.warmup_steps,
        sigma_adapt_steps=cfg.sigma_adapt_steps,
        kernel_adapt_steps=0,
        objective_fn=pro_logdensity_fn,
        inducing_basis=row_selectable_basis,
        x_val=split.x_val,
        y_val=split.y_val,
        sigma_optimizer=ox.adam(cfg.sigma_lr),
        progress_bar=False,
    )
    adaptation_results, _ = adaptation.run(
        adapt_key, initial_position, num_steps=cfg.num_adapt_steps
    )
    adapted_sigma_val = float(np.array(px.unwrap(adaptation_results.parameters.sigma)))
    log.info("Adapted sigma=%.4f", adapted_sigma_val)

    # Warm-started from adaptation, but now against the full training set (train + val),
    # so a real burn-in is warranted.
    pro_params = ProParameters(
        y=y_train,
        basis=basis_full,
        step_size=None,
        sigma=adaptation_results.parameters.sigma,
        alpha=cfg.alpha,
        residual_std=residual_std_full,
    )
    key, sample_key = jr.split(key)
    _, (states, _) = run_inference_algorithm_with_burn_in(
        rng_key=sample_key,
        inference_algorithm=parametric_replica_gibbs(pro_logdensity_fn, pro_params),
        num_steps=cfg.num_sample_steps,
        burn_ratio=cfg.burn_fraction,
        initial_position=adaptation_results.state.position,
        # The progress-bar host callback hung joblib workers on protein.
        progress_bar=False,
    )
    particles = states.position[:: cfg.thin]

    test_basis, test_cov = prediction_basis(
        kernel, x_train, x_test, pro_params, inducing_basis=inducing_basis
    )
    metrics = {
        "dataset": cfg.dataset,
        "split": cfg.split,
        "inducing": cfg.inducing,
        "gp_sigma": float(gp_sigma_val),
        "pro_sigma": adapted_sigma_val,
        "pro_nlpd": float(
            nlpd_pro(y_test, test_basis, test_cov, particles, parameters=pro_params)
        ),
    }
    log.info("PRO  NLPD=%.4f", metrics["pro_nlpd"])

    with open(out_dir / "pro_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "pro_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
