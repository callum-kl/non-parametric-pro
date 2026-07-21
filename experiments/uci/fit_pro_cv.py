"""PRO GP sampling, with sigma/kernel hyperparameters chosen via cross-validation.

Same overall structure as ``fit_pro.py`` (seeded from an exact GP or VGP depending on
``cfg.inducing``), but the adaptation step is replaced by
``cross_validated_parameter_adaptation``: sigma/kernel are adapted independently on
``cfg.num_folds`` folds of the training data and averaged, rather than adapted once via a
single train/validation split.
"""

import json
import logging
from pathlib import Path

import gpjax as gpx
import hydra
import jax
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px
from omegaconf import DictConfig, OmegaConf

from fit_pro import _cholesky_basis, load_gp_state  # noqa: F401 (gp_state_dir used by load_gp_state)
from non_parametric_pro import ula
from non_parametric_pro.adaptation.parameter_adaptation import (
    cross_validated_parameter_adaptation,
)
from non_parametric_pro.data.uci import load_uci_regression_dataset
from non_parametric_pro.density import ProParameters, pro_logdensity_fn, regularised_score
from non_parametric_pro.inducing import compute_inducing_basis
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import crps_pro, nlpd_pro, prediction_basis, run_inference_algorithm_with_burn_in

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)

# Anchors results_root/hydra.run.dir/hydra.sweep.dir to this script's own directory
# (experiments/uci/), regardless of the caller's current working directory.
OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parent), replace=True
)


def pro_out_dir(cfg: DictConfig) -> Path:
    subdir = "inducing_pro_gp_cv" if cfg.inducing else "pro_gp_cv"
    if cfg.name:
        subdir = f"{subdir}_{cfg.name}"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / subdir


@hydra.main(version_base=None, config_path="conf", config_name="fit_pro_cv")
def main(cfg: DictConfig) -> None:
    mode = "inducing" if cfg.inducing else "exact GP"
    log.info(
        "PRO-CV (%s): dataset=%s split=%d folds=%d", mode, cfg.dataset, cfg.split, cfg.num_folds
    )

    key = jr.PRNGKey(cfg.seed)
    out_dir = pro_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load GP state -------------------------------------------------------
    kernel, sigma_val, inducing_basis, scaler_x, scaler_y = load_gp_state(cfg)
    if cfg.inducing:
        log.info("Loaded VGP state: M=%d  sigma=%.4f", inducing_basis.z.shape[0], sigma_val)
    else:
        log.info("Loaded exact GP state: sigma=%.4f", sigma_val)

    # --- Data ------------------------------------------------------------------
    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test)

    log.info("N_train=%d  N_test=%d", x_train.shape[0], x_test.shape[0])

    # --- PRO setup ---------------------------------------------------------------
    # basis/basis_dim here are only a placeholder for ProParameters -- each fold inside
    # cross_validated_parameter_adaptation recomputes its own basis from its own
    # training split, overwriting this.
    if cfg.inducing:
        basis, residual_std = compute_inducing_basis(inducing_basis, kernel, x_train)
        basis_dim = inducing_basis.z.shape[0]
    else:
        basis, residual_std = _cholesky_basis(kernel, x_train), None
        basis_dim = x_train.shape[0]

    sigma = gpx.parameters.SigmoidBounded(sigma_val, low=cfg.sigma_min, high=100.0)
    pro_params = ProParameters(
        y=y_train,
        basis=basis,
        step_size=cfg.step_size,
        sigma=sigma,
        alpha=cfg.alpha,
        residual_std=residual_std,
    )

    # --- Cross-validated adaptation ---------------------------------------------
    log.info(
        "Running cross-validated adaptation (folds=%d, steps/fold=%d)...",
        cfg.num_folds, cfg.num_adapt_steps,
    )
    key, cv_key = jr.split(key)
    cv_result = cross_validated_parameter_adaptation(
        ula,
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
        sigma_adapt_every=cfg.sigma_adapt_every,
        kernel_adapt_every=cfg.kernel_adapt_every,
        objective_fn=regularised_score,
        rng_key=cv_key,
        inducing_basis=inducing_basis,
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
    else:
        basis_full = _cholesky_basis(adapted_kernel, x_train)
        residual_std_full = None
        basis_dim = x_train.shape[0]

    key, pos_key = jr.split(key)
    pro_position = jr.normal(pos_key, (basis_dim, cfg.num_particles))

    sigma = gpx.parameters.SigmoidBounded(adapted_sigma_val, low=cfg.sigma_min, high=100.0)
    pro_params = pro_params._replace(
        basis=basis_full,
        residual_std=residual_std_full,
        sigma=sigma,
    )

    # --- Sampling ------------------------------------------------------------
    log.info("Running sampling (steps=%d)...", cfg.num_sample_steps)
    algorithm = parametric_ula(pro_logdensity_fn, pro_params)

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
        "gp_sigma": float(sigma_val),
        "pro_sigma": pro_sigma,
        "pro_sigma_per_fold": np.array(cv_result.fold_sigma).tolist(),
        "pro_nlpd": float(nlpd_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
        "pro_crps": float(crps_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
    }
    log.info("PRO-CV  NLPD=%.4f  CRPS=%.4f", metrics["pro_nlpd"], metrics["pro_crps"])

    # --- Save ----------------------------------------------------------------
    with open(out_dir / "pro_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "pro_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
