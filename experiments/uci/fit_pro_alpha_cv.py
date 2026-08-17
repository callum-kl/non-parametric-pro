"""
PRO GP sampling with a fixed kernel (as loaded from ``fit_vgp.py``/``fit_exact_gp.py``
state, never adapted here), cross-validating the ``alpha`` (likelihood tempering)
parameter over a fixed grid.

For each candidate in ``alpha_grid``:

1. Split the training set into train/val (``non_parametric_pro.util.train_val_split``).
2. Run ``non_parametric_pro.parameter_adaptation.parameter_adaptation`` on the train
   fold to optimise ``sigma`` only (``kernel_adapt_steps=0`` -- the kernel stays fixed).
   Its own warmup + adaptation steps already move the particles, so no separate burn-in
   is needed afterwards.
3. Continue sampling from where adaptation left off via a plain
   ``blackjax.util.run_inference_algorithm`` call (no burn-in), using the adapted sigma
   and the basis adaptation already computed.
4. Thin the resulting particles and score them by NLPD on the val fold.

The alpha with the lowest val NLPD is then run through the same adapt-then-sample
procedure once more on the *full* training set (no val split, so sigma adapts against
the training set itself), and that final run is what gets evaluated on the test set.
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
from blackjax.util import run_inference_algorithm
from omegaconf import DictConfig, OmegaConf

from non_parametric_pro import ula
from non_parametric_pro.data.uci import load_uci_regression_dataset
from non_parametric_pro.density import ProParameters, pro_logdensity_fn, regularised_score
from non_parametric_pro.parameter_adaptation import parameter_adaptation
from non_parametric_pro.sgld import parametric_sgld, sgld
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import crps_pro, nlpd_pro, prediction_basis, train_val_split

from util import load_gp_state

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parent), replace=True
)


def pro_out_dir(cfg: DictConfig) -> Path:
    subdir = "inducing_pro_gp_alpha_cv" if cfg.inducing else "pro_gp_alpha_cv"
    if cfg.name:
        subdir = f"{subdir}_{cfg.name}"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / subdir


def _adaptation_algorithm(cfg: DictConfig):
    """`algorithm` argument for `parameter_adaptation` -- `ula` (exact, full-batch) or
    `sgld(batch_size=...)` (minibatched; see `non_parametric_pro.sgld`)."""
    if cfg.algorithm == "ula":
        return ula
    if cfg.algorithm == "sgld":
        return sgld(batch_size=cfg.sgld_batch_size)
    msg = f"Unknown algorithm={cfg.algorithm!r}; expected 'ula' or 'sgld'."
    raise ValueError(msg)


def _sampling_algorithm(cfg: DictConfig, pro_params: ProParameters):
    """Standalone sampler for the post-adaptation draw, mirroring `_adaptation_algorithm`'s
    choice of `ula` vs `sgld`."""
    if cfg.algorithm == "ula":
        return parametric_ula(pro_logdensity_fn, pro_params)
    if cfg.algorithm == "sgld":
        return parametric_sgld(pro_logdensity_fn, pro_params, batch_size=cfg.sgld_batch_size)
    msg = f"Unknown algorithm={cfg.algorithm!r}; expected 'ula' or 'sgld'."
    raise ValueError(msg)


def _adapt_and_sample(
    cfg: DictConfig,
    kernel: gpx.kernels.AbstractKernel,
    inducing_basis,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    alpha: float,
    rng_key,
) -> tuple[ProParameters, jnp.ndarray]:
    """Adapt sigma (kernel fixed) on `(x_train, y_train)`, then continue sampling (no
    burn-in) from where that adaptation left off. Returns `(adapted_params, particles)`,
    `particles` already thinned by `cfg.thin`.

    `x_val`/`y_val` given -> sigma adapts against that held-out fold (matching
    `fit_pro_cv.py`'s default `adapt_target=None` behaviour); `None` -> sigma adapts
    against `(x_train, y_train)` itself (used for the final full-training-set run).
    """
    pos_key, adapt_key, sample_key = jr.split(rng_key, 3)

    sigma = gpx.parameters.SigmoidBounded(cfg.sigma, low=cfg.sigma_min, high=100.0)
    fold_params = ProParameters(
        y=y_train, basis=None, step_size=cfg.step_size, sigma=sigma, alpha=alpha,
        residual_std=None,
    )

    basis_dim = inducing_basis.output_dim() if cfg.inducing else x_train.shape[0]
    initial_position = jr.normal(pos_key, (basis_dim, cfg.num_particles))

    adaptation = parameter_adaptation(
        _adaptation_algorithm(cfg),
        pro_logdensity_fn,
        fold_params,
        x_train=x_train,
        initial_kernel=kernel,
        warmup_steps=cfg.warmup_steps,
        sigma_adapt_steps=cfg.sigma_adapt_steps,
        kernel_adapt_steps=0,
        objective_fn=regularised_score,
        inducing_basis=inducing_basis,
        x_val=x_val,
        y_val=y_val,
        adapt_target=cfg.adapt_target,
        sigma_optimizer=ox.adam(cfg.sigma_lr),
        progress_bar=True,
    )
    adaptation_results, _ = adaptation.run(adapt_key, initial_position, num_steps=cfg.num_adapt_steps)
    adapted_params = adaptation_results.parameters

    algorithm = _sampling_algorithm(cfg, adapted_params)
    _, (states, _) = run_inference_algorithm(
        rng_key=sample_key,
        inference_algorithm=algorithm,
        num_steps=cfg.num_sample_steps,
        initial_state=adaptation_results.state,
        progress_bar=True,
    )
    particles = states.position[:: cfg.thin]
    return adapted_params, particles


@hydra.main(version_base=None, config_path="conf", config_name="fit_pro_alpha_cv")
def main(cfg: DictConfig) -> None:
    mode = "inducing" if cfg.inducing else "exact GP"
    log.info(
        "PRO-alpha-CV (%s): dataset=%s split=%d alpha_grid=%s",
        mode, cfg.dataset, cfg.split, list(cfg.alpha_grid),
    )

    key = jr.PRNGKey(cfg.seed)
    out_dir = pro_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load GP state -- fixed kernel, never adapted here -------------------
    kernel, gp_sigma_val, inducing_basis, scaler_x, scaler_y = load_gp_state(cfg)
    log.info(
        "Loaded %s state: kernel fixed, gp_sigma=%.4f (unused -- sigma starts fresh at "
        "cfg.sigma=%.4f each alpha and is re-adapted)",
        "VGP" if cfg.inducing else "exact GP", gp_sigma_val, cfg.sigma,
    )

    # --- Data ------------------------------------------------------------------
    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test)

    log.info("N_train=%d  N_test=%d", x_train.shape[0], x_test.shape[0])

    # --- Cross-validate alpha -----------------------------------------------------
    # Each alpha's adapted params/particles (from its own train/val split) are kept --
    # the winning alpha's are reused directly for test evaluation below, rather than
    # rerunning adapt-and-sample on the full training set.
    alpha_val_nlpd = []
    alpha_results = []
    for alpha in cfg.alpha_grid:
        key, split_key, run_key = jr.split(key, 3)
        split = train_val_split(split_key, x_train, y_train, val_fraction=cfg.val_fraction)

        adapted_params, particles = _adapt_and_sample(
            cfg, kernel, inducing_basis, split.x_train, split.y_train,
            split.x_val, split.y_val, float(alpha), run_key,
        )
        val_basis, val_cov = prediction_basis(
            kernel, split.x_train, split.x_val, adapted_params, inducing_basis=inducing_basis
        )
        val_nlpd = float(
            nlpd_pro(split.y_val, val_basis, val_cov, particles, parameters=adapted_params)
        )
        log.info("  alpha=%.4g  adapted sigma=%.4f  val NLPD=%.4f",
                  alpha, float(np.array(px.unwrap(adapted_params.sigma))), val_nlpd)
        alpha_val_nlpd.append(val_nlpd)
        alpha_results.append((adapted_params, particles, split.x_train))

    best_idx = int(np.argmin(alpha_val_nlpd))
    best_alpha = float(cfg.alpha_grid[best_idx])
    pro_params, particles, basis_x_train = alpha_results[best_idx]
    log.info("Best alpha=%.4g (val NLPD=%.4f)", best_alpha, alpha_val_nlpd[best_idx])

    # --- Evaluation --------------------------------------------------------------
    # `basis_x_train` is the winning alpha's own train-fold split (not the full training
    # set) -- it has to match whichever rows `pro_params.basis` was actually built from
    # (only matters for the exact-GP case; the inducing basis doesn't use this argument).
    test_basis, test_cov = prediction_basis(
        kernel, basis_x_train, x_test, pro_params, inducing_basis=inducing_basis
    )
    metrics = {
        "dataset": cfg.dataset,
        "split": cfg.split,
        "inducing": cfg.inducing,
        "gp_sigma": float(gp_sigma_val),
        "pro_sigma": float(np.array(px.unwrap(pro_params.sigma))),
        "alpha_grid": [float(a) for a in cfg.alpha_grid],
        "alpha_val_nlpd": alpha_val_nlpd,
        "best_alpha": best_alpha,
        "pro_nlpd": float(nlpd_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
        "pro_crps": float(crps_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
    }
    log.info(
        "PRO-alpha-CV  best_alpha=%.4g  NLPD=%.4f  CRPS=%.4f",
        best_alpha, metrics["pro_nlpd"], metrics["pro_crps"],
    )

    # --- Save ----------------------------------------------------------------
    with open(out_dir / "pro_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "pro_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
