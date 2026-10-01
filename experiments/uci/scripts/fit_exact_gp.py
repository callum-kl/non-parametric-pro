"""Fit an exact GP on a UCI split and save its hyperparameters."""

import json
import logging
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import hydra
import jax.numpy as jnp
import numpy as np
import paramax as px
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from util import split_dir

from non_parametric_pro.data.uci.uci import load_uci_regression_dataset
from non_parametric_pro.util import nlpd_gp

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_exact_gp")
def main(cfg: DictConfig) -> None:
    log.info("Fitting exact GP: dataset=%s split=%d", cfg.dataset, cfg.split)

    out_dir = split_dir(cfg) / "exact_gp"
    out_dir.mkdir(parents=True, exist_ok=True)

    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train = scaler_x.fit_transform(example.x_train)
    y_train = scaler_y.fit_transform(example.y_train)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test)

    D = x_train.shape[1]
    log.info("N_train=%d  N_test=%d  D=%d", x_train.shape[0], x_test.shape[0], D)

    data = gpx.Dataset(X=x_train, y=y_train)
    lengthscale = gpx.parameters.SigmoidBounded(
        jnp.sqrt(D) * jnp.ones((D,)), low=cfg.lengthscale_min, high=cfg.lengthscale_max
    )
    variance = (
        gpx.parameters.PositiveReal(jnp.array(1.0))
        if cfg.train_kernel_variance
        else px.NonTrainable(jnp.array(1.0))
    )
    kernel = gpx.kernels.RBF(lengthscale=lengthscale, variance=variance)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=data.n, obs_stddev=jnp.sqrt(0.01)
    )

    opt_posterior, _ = gpx.fit_scipy(
        model=prior * likelihood,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=data,
    )
    opt_kernel = opt_posterior.prior.kernel
    opt_sigma = opt_posterior.likelihood.obs_stddev

    predictive = opt_posterior.likelihood(opt_posterior.predict(x_test, train_data=data))
    metrics = {
        "gp_nlpd": float(
            nlpd_gp(y_test, predictive.mean, jnp.sqrt(predictive.variance))
        ),
        "gp_sigma": float(np.array(px.unwrap(opt_sigma)).reshape(())),
        "gp_variance": float(np.array(px.unwrap(opt_kernel.variance)).reshape(())),
    }
    log.info("GP  NLPD=%.4f", metrics["gp_nlpd"])

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
