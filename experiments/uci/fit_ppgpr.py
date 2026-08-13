"""Fit a PPGPR sparse variational GP on a UCI split and save its hyperparameters.

PPGPR (Jankowiak et al. 2020, "Parametric Gaussian Process Regressors") trains the
same sparse-variational predictive family as `fit_vgp.py`'s non-collapsed path
(`gpx.variational_families.VariationalGaussian`), but on the predictive log
likelihood objective (`non_parametric_pro.gp.predictive_log_likelihood`) instead of
the ELBO -- see that function's docstring for why this gives better-calibrated
predictive variances than plain SVGP (the ELBO's Jensen bound makes the latent
function variance enter as a separate KL-like penalty rather than as part of the same
Gaussian's variance as the observation noise, so SVGP tends to push almost all
predictive uncertainty into observation noise; PPGPR's data-fit term scores against
`sigma_obs^2 + sigma_f(x)^2` directly, so it can't do that).
"""

import json
import logging
import os
from pathlib import Path

# Must be set before `import jax` (and before any transitive jax import, e.g. via
# gpjax) -- jax.config.update("jax_enable_x64", True) here isn't enough, since under
# `-m hydra/launcher=joblib` the fit runs in a joblib worker process that doesn't
# reliably replay this module's own top-level statements before jax's backend
# initializes, silently leaving that worker on float32.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import hydra
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from non_parametric_pro.data.uci import load_uci_regression_dataset
from non_parametric_pro.gp import predictive_log_likelihood
from non_parametric_pro.inducing import kmeans_inducing_points
from non_parametric_pro.util import crps_gp, nlpd_gp

log = logging.getLogger(__name__)

# Anchors results_root/hydra.run.dir/hydra.sweep.dir to this script's own directory
# (experiments/uci/), regardless of the caller's current working directory.
OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parent), replace=True
)


def state_dir(cfg: DictConfig) -> Path:
    variant = "ppgpr" if cfg.name is None else f"ppgpr_{cfg.name}"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / variant


@hydra.main(version_base=None, config_path="conf", config_name="fit_ppgpr")
def main(cfg: DictConfig) -> None:
    log.info("Fitting PPGPR: dataset=%s split=%d", cfg.dataset, cfg.split)

    key = jr.PRNGKey(cfg.seed)
    out_dir = state_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ----------------------------------------------------------------
    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train = jnp.array(scaler_x.fit_transform(example.x_train))
    y_train = jnp.array(scaler_y.fit_transform(example.y_train))
    x_test = jnp.array(scaler_x.transform(example.x_test))
    y_test = jnp.array(scaler_y.transform(example.y_test))

    D = x_train.shape[1]
    log.info("N_train=%d  N_test=%d  D=%d", x_train.shape[0], x_test.shape[0], D)

    # --- PPGPR fit -------------------------------------------------------------
    # Lengthscale bounded, variance fixed to 1 -- same setup/rationale as
    # fit_exact_gp.py/fit_vgp.py (data is already standardised, so sigma/lengthscale
    # alone are sufficient, and a trainable variance risks the same degenerate
    # variance-vs-lengthscale trade-off documented in fit_vgp.py).
    data = gpx.Dataset(X=x_train, y=y_train)
    lengthscale = gpx.parameters.SigmoidBounded(
        jnp.sqrt(D) * jnp.ones((D,)), low=cfg.lengthscale_min, high=cfg.lengthscale_max
    )
    kernel = gpx.kernels.RBF(lengthscale=lengthscale, variance=px.NonTrainable(jnp.array(1.0)))
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=jnp.sqrt(0.01))
    posterior = prior * likelihood

    key, km_key = jr.split(key)
    z_init = kmeans_inducing_points(km_key, x_train, cfg.num_inducing).z

    variational_family = gpx.variational_families.VariationalGaussian(
        posterior=posterior,
        inducing_inputs=z_init,
        jitter=cfg.jitter,
    )
    objective = lambda p, d: -predictive_log_likelihood(p, d, beta=cfg.beta_reg)  # noqa: E731
    optim = ox.chain(ox.clip_by_global_norm(cfg.grad_clip_norm), ox.adam(cfg.kernel_lr))
    opt_vf, _ = gpx.fit(
        model=variational_family,
        objective=objective,
        train_data=data,
        optim=optim,
        num_iters=cfg.gp_num_iters,
        verbose=True,
        batch_size=cfg.ppgpr_batch_size,
    )

    opt_kernel = opt_vf.posterior.prior.kernel
    opt_sigma = opt_vf.posterior.likelihood.obs_stddev
    z_opt = np.array(px.unwrap(opt_vf.inducing_inputs))

    # --- Evaluate ------------------------------------------------------------
    # Non-collapsed family: predictive only depends on q(u), never on train_data.
    posterior_ = opt_vf.posterior
    latent = opt_vf.predict(x_test)
    predictive = posterior_.likelihood(latent)
    mean = predictive.mean
    std = jnp.sqrt(predictive.variance)

    metrics = {
        "ppgpr_nlpd": float(nlpd_gp(y_test, mean, std)),
        "ppgpr_crps": float(crps_gp(y_test, mean, std)),
        "gp_sigma": float(np.array(px.unwrap(opt_sigma)).reshape(())),
        "beta_reg": float(cfg.beta_reg),
    }
    log.info("PPGPR  NLPD=%.4f  CRPS=%.4f", metrics["ppgpr_nlpd"], metrics["ppgpr_crps"])

    # --- Save ----------------------------------------------------------------
    np.savez(
        out_dir / "gp_state.npz",
        kernel_type=type(kernel).__name__,
        lengthscale=np.array(px.unwrap(opt_kernel.lengthscale)),
        variance=np.array(px.unwrap(opt_kernel.variance)),
        sigma=np.array(px.unwrap(opt_sigma)).reshape(()),
        z=z_opt,
        scaler_x_mean=scaler_x.mean_,
        scaler_x_scale=scaler_x.scale_,
        scaler_y_mean=scaler_y.mean_,
        scaler_y_scale=scaler_y.scale_,
    )

    with open(out_dir / "gp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "gp_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved state to %s", out_dir)


if __name__ == "__main__":
    main()
