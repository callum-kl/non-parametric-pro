"""
PRO GP sampling with the basis built from *all* training points up front (dimension
``N`` for the exact-GP case, ``M`` for the inducing case) -- not just a train fold's
worth, unlike ``fit_pro_cv.py``/``fit_pro_alpha_cv.py``. That basis, and the particles
sampled against it, are shared across the whole run: only *which rows'* likelihood
terms get used changes between phases, never the basis or particle dimensionality.

1. Split the training set into train/val (``non_parametric_pro.util.train_val_split``).
2. Run ``parameter_adaptation`` to optimise sigma only (kernel fixed, loaded from GP
   state) -- the sampler's likelihood uses only the train fold's rows, and sigma is
   scored against the val fold's rows, both as row-selections of the *same* full basis
   (see ``_row_selectable_basis`` below for how the exact-GP case gets this property).
3. One final run reusing that adaptation's basis and final particle *position* (not its
   cached state, which is stale under a different likelihood scope) on the full training
   set's likelihood, via ``run_inference_algorithm_with_burn_in`` -- a real burn-in this
   time, since the likelihood scope just widened from the train fold to everything.

Only a single train/val split is used (no ``num_folds > 1`` support).
"""

import json
import logging
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
    subdir = "inducing_pro_gp_full" if cfg.inducing else "pro_gp_full"
    if cfg.name:
        subdir = f"{subdir}_{cfg.name}"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / subdir


def _row_selectable_basis(cfg: DictConfig, inducing_basis, x_train) -> InducingBasis:
    """An ``InducingBasis`` that ``parameter_adaptation`` can evaluate at any row subset
    of ``x_train`` and get back exactly that subset's rows of the *shared* full basis.

    For the real inducing case this is just ``inducing_basis`` unchanged -- each row of
    an (N, M) inducing basis only depends on its own point and the M fixed inducing
    locations, so evaluating it at a subset already equals row-slicing the full basis.

    For the exact-GP case there's no ``inducing_basis`` to reuse, but the same property
    can be manufactured: treat the *entire* training set as a (degenerate, exact -- no
    Nystrom approximation, since M=N) set of inducing points. For ``L`` the training
    Cholesky and ``A`` a row subset, ``compute_inducing_basis(PointInducingBasis(x_train),
    kernel, x_train[A])`` works out to ``solve_triangular(L, K_full[:, A]).T``, which
    equals ``L[A, :]`` -- exactly the row-slice ``basis_full[A]`` -- since ``K_full[:, A]
    = L @ L[A, :].T`` by definition of ``L``. That's the same formula
    ``non_parametric_pro.util.prediction_basis`` already uses for genuinely new (test)
    points, so this is just that formula applied to points already inside ``x_train``.
    """
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


@hydra.main(version_base=None, config_path="../conf", config_name="fit_pro_full_basis")
def main(cfg: DictConfig) -> None:
    mode = "inducing" if cfg.inducing else "exact GP"
    log.info("PRO-full-basis (%s): dataset=%s split=%d", mode, cfg.dataset, cfg.split)

    key = jr.PRNGKey(cfg.seed)
    out_dir = pro_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load GP state -- fixed kernel, never adapted here -------------------
    kernel, gp_sigma_val, inducing_basis, scaler_x, scaler_y = load_gp_state(cfg)
    sigma_init_val = gp_sigma_val if cfg.sigma_init is None else cfg.sigma_init
    log.info(
        "Loaded %s state: kernel fixed, gp_sigma=%.4f, sigma starts at %.4f",
        "VGP" if cfg.inducing else "exact GP", gp_sigma_val, sigma_init_val,
    )

    # --- Data ------------------------------------------------------------------
    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test)

    log.info("N_train=%d  N_test=%d", x_train.shape[0], x_test.shape[0])

    # --- Full basis, from *all* training points -------------------------------
    if cfg.inducing:
        basis_full, residual_std_full = compute_inducing_basis(inducing_basis, kernel, x_train)
    else:
        basis_full = cholesky_basis(kernel, x_train)
        residual_std_full = None
    basis_dim = basis_full.shape[1]
    row_selectable_basis = _row_selectable_basis(cfg, inducing_basis, x_train)

    # --- Sigma adaptation: train fold's likelihood fits the particles, val fold scores
    # sigma -- both as row-selections of the one shared full basis (see
    # `_row_selectable_basis`), so the particle dimensionality never changes.
    log.info("Splitting train/val (val_fraction=%.2f) and adapting sigma...", cfg.val_fraction)
    key, split_key, pos_key, adapt_key = jr.split(key, 4)
    split = train_val_split(split_key, x_train, y_train, val_fraction=cfg.val_fraction)

    fold_params = ProParameters(
        y=split.y_train, basis=None, step_size=cfg.step_size,
        sigma=gpx.parameters.SigmoidBounded(sigma_init_val, low=cfg.sigma_min, high=cfg.sigma_max),
        alpha=cfg.alpha, residual_std=None,
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
    adaptation_results, adaptation_info = adaptation.run(adapt_key, initial_position, num_steps=cfg.num_adapt_steps)
    adapted_sigma_val = float(np.array(px.unwrap(adaptation_results.parameters.sigma)))
    log.info("Adapted sigma=%.4f", adapted_sigma_val)

    # `adaptation_results.parameters` (a ProParameters) has no `kernel` field to read the
    # adapted kernel back from -- only `adaptation_info.kernel`, the full per-step trace,
    # does (see cross_validated_parameter_adaptation's own use of this same pattern).
    if cfg.kernel_adapt_steps > 0:
        adapted_kernel = px.unwrap(jax.tree.map(lambda x: x[-1], adaptation_info.kernel))
        log.info("Kernel was adapted (kernel_adapt_steps=%d) -- recomputing full basis...", cfg.kernel_adapt_steps)
        if cfg.inducing:
            basis_full, residual_std_full = compute_inducing_basis(inducing_basis, adapted_kernel, x_train)
        else:
            basis_full = cholesky_basis(adapted_kernel, x_train)
            residual_std_full = None
    else:
        adapted_kernel = kernel

    # --- Final run: same full basis, particles warm-started from adaptation's final
    # position -- but now against the *full* training set's likelihood (train + val
    # combined), so a real burn-in is warranted since the likelihood scope just widened.
    log.info(
        "Running final sampling on full training set (steps=%d, algorithm=%s)...",
        cfg.num_sample_steps, cfg.algorithm,
    )
    pro_params = ProParameters(
        y=y_train, basis=basis_full, step_size=cfg.step_size,
        sigma=adaptation_results.parameters.sigma, alpha=cfg.alpha, residual_std=residual_std_full,
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
        "pro_nlpd": float(nlpd_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
        "pro_crps": float(crps_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
    }
    log.info("PRO-full-basis  NLPD=%.4f  CRPS=%.4f", metrics["pro_nlpd"], metrics["pro_crps"])

    # --- Save ----------------------------------------------------------------
    with open(out_dir / "pro_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "pro_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
