"""Fit an exact GP on a UCI split and save its hyperparameters."""

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

from non_parametric_pro.data.uci import load_uci_regression_dataset
from non_parametric_pro.util import crps_gp, nlpd_gp

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)


def state_dir(cfg: DictConfig) -> Path:
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / "exact_gp"


@hydra.main(version_base=None, config_path="conf", config_name="fit_exact_gp")
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
    kernel = gpx.kernels.RBF(lengthscale=jnp.sqrt(D) * jnp.ones((D,)))
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=jnp.sqrt(0.01))
    posterior = prior * likelihood

    opt_posterior, _ = gpx.fit(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=data,
        optim=ox.adam(cfg.kernel_lr),
        num_iters=cfg.gp_num_iters,
        verbose=False,
    )

    opt_kernel = opt_posterior.prior.kernel
    opt_sigma = opt_posterior.likelihood.obs_stddev

    # --- Evaluate ------------------------------------------------------------
    latent = opt_posterior.predict(x_test, train_data=data)
    predictive = opt_posterior.likelihood(latent)
    mean = predictive.mean
    std = jnp.sqrt(predictive.variance)

    metrics = {
        "gp_nlpd": float(nlpd_gp(y_test, mean, std)),
        "gp_crps": float(crps_gp(y_test, mean, std)),
    }
    log.info("GP  NLPD=%.4f  CRPS=%.4f", metrics["gp_nlpd"], metrics["gp_crps"])

    # --- Save ----------------------------------------------------------------
    # No 'z' key — fit_pro.py uses this to detect the non-inducing case.
    np.savez(
        out_dir / "gp_state.npz",
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
