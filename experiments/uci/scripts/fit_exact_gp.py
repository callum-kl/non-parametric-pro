"""Fit an exact GP on a UCI split and save its hyperparameters."""

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
from sklearn.preprocessing import StandardScaler

from non_parametric_pro.data.uci import load_uci_regression_dataset
from non_parametric_pro.util import crps_gp, nlpd_gp

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)

def state_dir(cfg: DictConfig) -> Path:
    subdir = f"exact_gp_{cfg.name}" if cfg.name else "exact_gp"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / subdir


def _objective_fn(name: str):
    """`cfg.objective` -> gpjax objective callable, negated for minimisation."""
    if name == "mll":
        return lambda p, d: -gpx.objectives.conjugate_mll(p, d)
    if name == "loocv":
        return lambda p, d: -gpx.objectives.conjugate_loocv(p, d)
    msg = f"Unknown objective={name!r}; expected 'mll' or 'loocv'."
    raise ValueError(msg)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_exact_gp")
def main(cfg: DictConfig) -> None:
    log.info("Fitting exact GP: dataset=%s split=%d", cfg.dataset, cfg.split)

    out_dir = state_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ----------------------------------------------------------------
    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train = scaler_x.fit_transform(example.x_train)
    y_train = scaler_y.fit_transform(example.y_train)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test)

    D = x_train.shape[1]
    log.info("N_train=%d  N_test=%d  D=%d", x_train.shape[0], x_test.shape[0], D)

    # --- Exact GP fit --------------------------------------------------------
    data = gpx.Dataset(X=x_train, y=y_train)
    key = jr.PRNGKey(cfg.seed)
    restart_keys = jr.split(key, cfg.num_restarts)
    objective_fn = _objective_fn(cfg.objective)

    best_posterior = None
    best_loss = jnp.inf
    for i, restart_key in enumerate(restart_keys):
        if i == 0:
            init_lengthscale = jnp.sqrt(D) * jnp.ones((D,))
        else:
            jitter = cfg.restart_jitter_std * jr.normal(restart_key, (D,))
            init_lengthscale = jnp.clip(
                jnp.sqrt(D) * jnp.exp(jitter),
                cfg.lengthscale_min * 1.01,
                cfg.lengthscale_max * 0.99,
            )

        lengthscale = gpx.parameters.SigmoidBounded(
            init_lengthscale, low=cfg.lengthscale_min, high=cfg.lengthscale_max
        )
        kernel = gpx.kernels.RBF(lengthscale=lengthscale, variance=px.NonTrainable(jnp.array(1.0)))
        prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=jnp.sqrt(0.01))
        posterior = prior * likelihood

        candidate, history = gpx.fit_scipy(
            model=posterior,
            objective=objective_fn,
            train_data=data,
            verbose=(cfg.num_restarts == 1),
        )
        final_loss = float(history[-1])
        log.info(
            "Restart %d/%d: final negative %s=%.4f",
            i + 1, cfg.num_restarts, cfg.objective, final_loss,
        )
        if final_loss < best_loss:
            best_loss = final_loss
            best_posterior = candidate

    opt_posterior = best_posterior
    log.info("Best restart: negative %s=%.4f", cfg.objective, best_loss)

    opt_kernel = opt_posterior.prior.kernel
    opt_sigma = opt_posterior.likelihood.obs_stddev

    print(px.unwrap(opt_kernel.lengthscale))

    # --- Evaluate ------------------------------------------------------------
    latent = opt_posterior.predict(x_test, train_data=data)
    predictive = opt_posterior.likelihood(latent)
    mean = predictive.mean
    std = jnp.sqrt(predictive.variance)

    metrics = {
        "gp_nlpd": float(nlpd_gp(y_test, mean, std)),
        "gp_crps": float(crps_gp(y_test, mean, std)),
        "gp_sigma": float(np.array(px.unwrap(opt_sigma)).reshape(())),
        "objective": cfg.objective,
    }
    log.info("GP  NLPD=%.4f  CRPS=%.4f", metrics["gp_nlpd"], metrics["gp_crps"])

    # --- Save ----------------------------------------------------------------
    # No 'z' key — fit_pro.py uses this to detect the non-inducing case.
    # kernel_type records the actual class used to fit (e.g. "RBF") so load_gp_state
    # reconstructs the same kernel shape rather than assuming one.
    np.savez(
        out_dir / "gp_state.npz",
        kernel_type=type(kernel).__name__,
        lengthscale=np.array(px.unwrap(opt_kernel.lengthscale)),
        variance=np.array(px.unwrap(opt_kernel.variance)),
        sigma=np.array(px.unwrap(opt_sigma)).reshape(()),
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
