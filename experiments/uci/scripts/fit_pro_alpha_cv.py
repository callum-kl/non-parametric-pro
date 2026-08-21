import json
import logging
import math
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import hydra
import jax
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px
from omegaconf import DictConfig, OmegaConf

from non_parametric_pro import ula
from non_parametric_pro.data.uci import load_uci_regression_dataset
from non_parametric_pro.density import ProParameters, pro_logdensity_fn, regularised_score
from non_parametric_pro.inducing import InducingBasis, PointInducingBasis, compute_inducing_basis
from non_parametric_pro.parameter_adaptation import parameter_adaptation
from non_parametric_pro.sgld import parametric_sgld, sgld
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import (
    cholesky_basis,
    crps_pro,
    nlpd_pro,
    prediction_basis,
    run_inference_algorithm_with_burn_in,
    train_val_split,
)

from util import load_gp_state

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


def pro_out_dir(cfg: DictConfig) -> Path:
    subdir = "inducing_pro_gp_alpha_cv" if cfg.inducing else "pro_gp_alpha_cv"
    if cfg.name:
        subdir = f"{subdir}_{cfg.name}"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / subdir


def _row_selectable_basis(cfg: DictConfig, inducing_basis, x_train) -> InducingBasis:
    """Identical to ``fit_pro_full_basis.py``'s helper of the same name -- see there for
    the full derivation of why this gives exact row-slices of the shared full basis for
    both the inducing and exact-GP cases."""
    return inducing_basis if cfg.inducing else PointInducingBasis(z=x_train)


def _adaptation_algorithm(cfg: DictConfig):
    if cfg.algorithm == "ula":
        return ula
    if cfg.algorithm == "sgld":
        return sgld(batch_size=cfg.sgld_batch_size)
    msg = f"Unknown algorithm={cfg.algorithm!r}; expected 'ula' or 'sgld'."
    raise ValueError(msg)


def _sampling_algorithm(cfg: DictConfig, pro_params: ProParameters):
    if cfg.algorithm == "ula":
        return parametric_ula(pro_logdensity_fn, pro_params)
    if cfg.algorithm == "sgld":
        return parametric_sgld(pro_logdensity_fn, pro_params, batch_size=cfg.sgld_batch_size)
    msg = f"Unknown algorithm={cfg.algorithm!r}; expected 'ula' or 'sgld'."
    raise ValueError(msg)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_pro_alpha_cv")
def main(cfg: DictConfig) -> None:
    mode = "inducing" if cfg.inducing else "exact GP"
    log.info(
        "PRO-alpha-CV-full-basis (%s): dataset=%s split=%d c_grid=%s",
        mode, cfg.dataset, cfg.split, list(cfg.c_grid),
    )

    key = jr.PRNGKey(cfg.seed)
    out_dir = pro_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load GP state -- fixed kernel unless kernel_adapt_steps > 0 ---------
    kernel, gp_sigma_val, inducing_basis, scaler_x, scaler_y = load_gp_state(cfg)
    sigma_init_val = gp_sigma_val if cfg.sigma_init is None else cfg.sigma_init
    log.info(
        "Loaded %s state: gp_sigma=%.4f, sigma starts at %.4f",
        "VGP" if cfg.inducing else "exact GP", gp_sigma_val, sigma_init_val,
    )

    # --- Data ------------------------------------------------------------------
    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test)
    n = x_train.shape[0]

    log.info("N_train=%d  N_test=%d", n, x_test.shape[0])

    # --- Full basis, from *all* training points -------------------------------
    if cfg.inducing:
        basis_full, residual_std_full = compute_inducing_basis(inducing_basis, kernel, x_train)
    else:
        basis_full = cholesky_basis(kernel, x_train)
        residual_std_full = None
    basis_dim = basis_full.shape[1]
    row_selectable_basis = _row_selectable_basis(cfg, inducing_basis, x_train)

    # --- Cross-validate c (alpha = c / sqrt(n)) --------------------------------
    c_val_nlpd = []
    c_results = []
    for c in cfg.c_grid:
        alpha = float(c) / math.sqrt(n)
        key, split_key, pos_key, adapt_key = jr.split(key, 4)
        split = train_val_split(split_key, x_train, y_train, val_fraction=cfg.val_fraction)

        fold_params = ProParameters(
            y=split.y_train, basis=None, step_size=cfg.step_size,
            sigma=gpx.parameters.SigmoidBounded(sigma_init_val, low=cfg.sigma_min, high=cfg.sigma_max),
            alpha=alpha, residual_std=None,
        )
        initial_position = jr.normal(pos_key, (basis_dim, cfg.num_particles))

        adaptation = parameter_adaptation(
            _adaptation_algorithm(cfg),
            pro_logdensity_fn,
            fold_params,
            x_train=split.x_train,
            initial_kernel=kernel,
            warmup_steps=cfg.warmup_steps,
            sigma_adapt_steps=cfg.sigma_adapt_steps,
            kernel_adapt_steps=cfg.kernel_adapt_steps,
            objective_fn=regularised_score,
            inducing_basis=row_selectable_basis,
            x_val=split.x_val,
            y_val=split.y_val,
            adapt_target=cfg.adapt_target,
            sigma_optimizer=ox.adam(cfg.sigma_lr),
            kernel_optimizer=ox.adam(cfg.kernel_lr),
            progress_bar=True,
        )
        adaptation_results, adaptation_info = adaptation.run(
            adapt_key, initial_position, num_steps=cfg.num_adapt_steps
        )
        adapted_kernel_c = (
            px.unwrap(jax.tree.map(lambda x: x[-1], adaptation_info.kernel))
            if cfg.kernel_adapt_steps > 0 else kernel
        )
        val_basis, val_cov = prediction_basis(
            adapted_kernel_c, split.x_train, split.x_val,
            adaptation_results.parameters, inducing_basis=row_selectable_basis,
        )
        val_nlpd = float(nlpd_pro(
            split.y_val, val_basis, val_cov, adaptation_results.state.position,
            parameters=adaptation_results.parameters,
        ))
        log.info(
            "  c=%.4g  alpha=%.4g  adapted sigma=%.4f  val NLPD=%.4f",
            c, alpha, float(np.array(px.unwrap(adaptation_results.parameters.sigma))), val_nlpd,
        )
        c_val_nlpd.append(val_nlpd)
        c_results.append((adaptation_results, adapted_kernel_c))

    best_idx = int(np.argmin(c_val_nlpd))
    best_c = float(cfg.c_grid[best_idx])
    best_alpha = best_c / math.sqrt(n)
    adaptation_results, adapted_kernel = c_results[best_idx]
    adapted_sigma_val = float(np.array(px.unwrap(adaptation_results.parameters.sigma)))
    log.info(
        "Best c=%.4g (alpha=%.4g, val NLPD=%.4f)", best_c, best_alpha, c_val_nlpd[best_idx]
    )

    if cfg.kernel_adapt_steps > 0:
        log.info("Kernel was adapted (kernel_adapt_steps=%d) -- recomputing full basis...", cfg.kernel_adapt_steps)
        if cfg.inducing:
            basis_full, residual_std_full = compute_inducing_basis(inducing_basis, adapted_kernel, x_train)
        else:
            basis_full = cholesky_basis(adapted_kernel, x_train)
            residual_std_full = None

    log.info(
        "Running final sampling on full training set (steps=%d, algorithm=%s)...",
        cfg.num_sample_steps, cfg.algorithm,
    )
    pro_params = ProParameters(
        y=y_train, basis=basis_full, step_size=cfg.step_size,
        sigma=adaptation_results.parameters.sigma, alpha=best_alpha, residual_std=residual_std_full,
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
    test_basis, test_cov = prediction_basis(
        adapted_kernel, x_train, x_test, pro_params, inducing_basis=inducing_basis if cfg.inducing else None
    )
    metrics = {
        "dataset": cfg.dataset,
        "split": cfg.split,
        "inducing": cfg.inducing,
        "gp_sigma": float(gp_sigma_val),
        "pro_sigma": adapted_sigma_val,
        "n_train": n,
        "c_grid": [float(c) for c in cfg.c_grid],
        "alpha_grid": [float(c) / math.sqrt(n) for c in cfg.c_grid],
        "c_val_nlpd": c_val_nlpd,
        "best_c": best_c,
        "best_alpha": best_alpha,
        "pro_nlpd": float(nlpd_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
        "pro_crps": float(crps_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
    }
    log.info(
        "PRO-alpha-CV-full-basis  best_c=%.4g  NLPD=%.4f  CRPS=%.4f",
        best_c, metrics["pro_nlpd"], metrics["pro_crps"],
    )

    # --- Save ----------------------------------------------------------------
    with open(out_dir / "pro_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "pro_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
