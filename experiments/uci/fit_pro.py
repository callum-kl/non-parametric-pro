"""PRO GP sampling, seeded from an exact GP or VGP depending on cfg.inducing."""

import json
import logging
from pathlib import Path

import gpjax as gpx
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from non_parametric_pro import ula
from non_parametric_pro.adaptation.parameter_adaptation import parameter_adaptation
from non_parametric_pro.data.uci import load_uci_regression_dataset
from non_parametric_pro.density import ProParameters, pro_logdensity_fn, regularised_score
from non_parametric_pro.inducing import PointInducingBasis, compute_inducing_basis
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import crps_pro, nlpd_pro, prediction_basis, run_inference_algorithm_with_burn_in

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)

# Anchors results_root/hydra.run.dir/hydra.sweep.dir to this script's own directory
# (experiments/uci/), regardless of the caller's current working directory.
OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parent), replace=True
)


def gp_state_dir(cfg: DictConfig) -> Path:
    subdir = "vgp" if cfg.inducing else "exact_gp"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / subdir


def pro_out_dir(cfg: DictConfig) -> Path:
    subdir = "inducing_pro_gp" if cfg.inducing else "pro_gp"
    if cfg.name:
        subdir = f"{subdir}_{cfg.name}"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / subdir


def _cholesky_basis(kernel, x, jitter=1e-6):
    k = kernel.gram(x).as_matrix()
    return jnp.linalg.cholesky(k + jitter * jnp.eye(k.shape[0]))


def load_gp_state(cfg: DictConfig):
    """Load state from fit_vgp.py (inducing=true) or fit_exact_gp.py (inducing=false)."""
    path = gp_state_dir(cfg) / "gp_state.npz"
    script = "fit_vgp.py" if cfg.inducing else "fit_exact_gp.py"
    if not path.exists():
        raise FileNotFoundError(f"No state at {path} — run {script} first.")

    state = np.load(path)
    if "kernel_type" in state:
        kernel_type = str(state["kernel_type"])
    else:
        # Pre-existing states saved before kernel_type was recorded -- these were all
        # fit with Matern32, so this matches their actual behaviour; re-run
        # fit_exact_gp.py/fit_vgp.py to pick up kernel_type for new fits.
        kernel_type = "Matern32"
        log.warning(
            "%s has no 'kernel_type' (pre-fix save); assuming Matern32. "
            "Re-run %s to save kernel_type explicitly.",
            path, script,
        )
    kernel_cls = getattr(gpx.kernels, kernel_type)
    kernel = kernel_cls(
        lengthscale=jnp.array(state["lengthscale"]),
        variance=jnp.array(state["variance"]),
    )
    sigma_val = float(state["sigma"])
    inducing_basis = PointInducingBasis(jnp.array(state["z"])) if cfg.inducing else None

    scaler_x = StandardScaler()
    scaler_x.mean_ = state["scaler_x_mean"]
    scaler_x.scale_ = state["scaler_x_scale"]
    scaler_y = StandardScaler()
    scaler_y.mean_ = state["scaler_y_mean"]
    scaler_y.scale_ = state["scaler_y_scale"]

    return kernel, sigma_val, inducing_basis, scaler_x, scaler_y


@hydra.main(version_base=None, config_path="conf", config_name="fit_pro")
def main(cfg: DictConfig) -> None:
    mode = "inducing" if cfg.inducing else "exact GP"
    log.info("PRO (%s): dataset=%s split=%d", mode, cfg.dataset, cfg.split)

    key = jr.PRNGKey(cfg.seed)
    out_dir = pro_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load GP state -------------------------------------------------------
    kernel, sigma_val, inducing_basis, scaler_x, scaler_y = load_gp_state(cfg)
    if cfg.inducing:
        log.info("Loaded VGP state: M=%d  sigma=%.4f",
                 inducing_basis.z.shape[0], sigma_val)
    else:
        log.info("Loaded exact GP state: sigma=%.4f", sigma_val)

    if cfg.initial_sigma is not None:
        log.info("Overriding loaded sigma=%.4f with cfg.initial_sigma=%.4f", sigma_val, cfg.initial_sigma)
        sigma_val = cfg.initial_sigma

    # --- Data ----------------------------------------------------------------
    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test)

    log.info("N_train=%d  N_test=%d", x_train.shape[0], x_test.shape[0])

    # --- PRO setup -----------------------------------------------------------
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

    key, pos_key = jr.split(key)
    pro_position = jr.normal(pos_key, (basis_dim, cfg.num_particles))

    # --- Parameter adaptation ------------------------------------------------
    log.info("Running adaptation (steps=%d)...", cfg.num_adapt_steps)
    adaptation = parameter_adaptation(
        ula,
        pro_logdensity_fn,
        pro_params,
        x_train=x_train,
        initial_kernel=kernel,
        warmup_steps=cfg.warmup_steps,
        sigma_adapt_every=cfg.sigma_adapt_every,
        kernel_adapt_every=cfg.kernel_adapt_every,
        objective_fn=regularised_score,
        inducing_basis=inducing_basis,
        sigma_optimizer=ox.adam(cfg.sigma_lr),
        kernel_optimizer=ox.adam(cfg.kernel_lr),
        progress_bar=True,
    )

    key, adapt_key = jr.split(key)
    adaptation_results, adaptation_info = adaptation.run(
        adapt_key, pro_position, num_steps=cfg.num_adapt_steps
    )
    adapted_kernel = jax.tree.map(lambda x: x[-1], adaptation_info.kernel)
    pro_params = adaptation_results.parameters
    pro_position = adaptation_results.state.position

    # Recompute basis on the full training set with the adapted kernel.
    # For inducing, basis_dim = M is unchanged so particles carry over directly.
    # For the full GP, basis_dim changes from N_train to N_full, so we reinitialise
    # particles — the adapted kernel and sigma are still used as the starting point.
    if cfg.kernel_adapt_every > 0:
        log.info("Recomputing basis with adapted kernel...")
        if cfg.inducing:
            basis_full, residual_std_full = compute_inducing_basis(
                inducing_basis, adapted_kernel, x_train
            )
        else:
            basis_full = _cholesky_basis(adapted_kernel, x_train)
            residual_std_full = None
            key, pos_key = jr.split(key)
            pro_position = jr.normal(pos_key, (x_train.shape[0], cfg.num_particles))

        pro_params = pro_params._replace(
            basis=basis_full,
            residual_std=residual_std_full,
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

    # --- Evaluation ----------------------------------------------------------
    test_basis, test_cov = prediction_basis(
        adapted_kernel, x_train, x_test, pro_params, inducing_basis=inducing_basis
    )
    pro_sigma = float(np.array(px.unwrap(pro_params.sigma)).reshape(()))
    metrics = {
        "dataset": cfg.dataset,
        "split": cfg.split,
        "inducing": cfg.inducing,
        "gp_sigma": float(sigma_val),
        "pro_sigma": pro_sigma,
        "pro_nlpd": float(nlpd_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
        "pro_crps": float(crps_pro(
            y_test, test_basis, test_cov, particles, parameters=pro_params
        )),
    }
    log.info("PRO  NLPD=%.4f  CRPS=%.4f", metrics["pro_nlpd"], metrics["pro_crps"])

    # --- Save ----------------------------------------------------------------
    with open(out_dir / "pro_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "pro_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
