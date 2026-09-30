"""Fit an exact GP with a graph Matern kernel on a PeMS road-network split."""

import json
import logging
import os
import warnings

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import hydra
import jax.numpy as jnp
import numpy as np
import optax as ox
import paramax as px
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from util import gp_state_dir, load_split

from non_parametric_pro.data.pems.pems import load_pems_graph_data
from non_parametric_pro.util import nlpd_gp

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_exact_gp")
def main(cfg: DictConfig) -> None:
    log.info("Fitting exact graph GP: split=%d num_train=%d", cfg.split, cfg.num_train)

    out_dir = gp_state_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_data = load_pems_graph_data()
    split = load_split(cfg, graph_data)

    scaler_y = StandardScaler()
    x_train = jnp.asarray(split.x_train)
    x_test = jnp.asarray(split.x_test)
    y_train = scaler_y.fit_transform(split.y_train)
    y_test = scaler_y.transform(split.y_test)

    log.info(
        "N_train=%d  N_test=%d  num_nodes=%d",
        x_train.shape[0],
        x_test.shape[0],
        graph_data.num_nodes,
    )

    # X holds integer node indices for the GraphKernel, so GPJax's float64 check
    # doesn't apply.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X is not of type float64")
        data = gpx.Dataset(X=x_train, y=y_train)
    kernel = gpx.kernels.GraphKernel(
        laplacian=graph_data.laplacian,
        lengthscale=gpx.parameters.SigmoidBounded(
            jnp.array(cfg.lengthscale_init),
            low=cfg.lengthscale_min,
            high=cfg.lengthscale_max,
        ),
        variance=gpx.parameters.PositiveReal(jnp.array(1.0)),
        smoothness=gpx.parameters.SigmoidBounded(
            jnp.array(cfg.smoothness_init),
            low=cfg.smoothness_min,
            high=cfg.smoothness_max,
        ),
    )
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=data.n, obs_stddev=jnp.sqrt(0.01)
    )

    opt_posterior, _ = gpx.fit(
        model=prior * likelihood,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=data,
        optim=ox.adam(cfg.kernel_lr),
        num_iters=cfg.num_iters,
    )
    opt_kernel = opt_posterior.prior.kernel
    opt_sigma = opt_posterior.likelihood.obs_stddev

    log.info("lengthscale=%s", px.unwrap(opt_kernel.lengthscale))
    log.info("smoothness=%s", px.unwrap(opt_kernel.smoothness))
    log.info("variance=%s", px.unwrap(opt_kernel.variance))

    predictive = opt_posterior.likelihood(opt_posterior.predict(x_test, train_data=data))
    metrics = {
        "split": cfg.split,
        "num_train": cfg.num_train,
        "gp_nlpd": float(
            nlpd_gp(y_test, predictive.mean, jnp.sqrt(predictive.variance))
        ),
    }
    log.info("GP  NLPD=%.4f", metrics["gp_nlpd"])

    np.savez(
        out_dir / "gp_state.npz",
        kernel_type="GraphKernel",
        laplacian=np.array(graph_data.laplacian),
        lengthscale=np.array(px.unwrap(opt_kernel.lengthscale)),
        variance=np.array(px.unwrap(opt_kernel.variance)),
        smoothness=np.array(px.unwrap(opt_kernel.smoothness)),
        sigma=np.array(px.unwrap(opt_sigma)).reshape(()),
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
