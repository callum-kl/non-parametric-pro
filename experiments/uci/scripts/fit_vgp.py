"""Fit a variational sparse GP on a UCI split and save its hyperparameters."""

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

from non_parametric_pro.data.uci.uci import load_uci_regression_dataset
from non_parametric_pro.gp import natural_gradient_svgp_fit
from non_parametric_pro.inducing import kmeans_inducing_points
from non_parametric_pro.util import nlpd_gp

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


def state_dir(cfg: DictConfig) -> Path:
    variant = "vgp" if cfg.collapsed else "vgp_noncollapsed"
    if cfg.name:
        variant = f"{variant}_{cfg.name}"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / variant


@hydra.main(version_base=None, config_path="../conf", config_name="fit_vgp")
def main(cfg: DictConfig) -> None:
    log.info("Fitting sparse GP: dataset=%s split=%d", cfg.dataset, cfg.split)

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

    # --- Sparse GP fit -------------------------------------------------------
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

    if cfg.collapsed:
        variational_family = gpx.variational_families.CollapsedVariationalGaussian(
            posterior=posterior,
            inducing_inputs=z_init,
        )
        objective = lambda p, d: -gpx.objectives.collapsed_elbo(p, d)  # noqa: E731
        opt_vf, _ = gpx.fit_scipy(
            model=variational_family,
            objective=objective,
            train_data=data,
            verbose=True,
        )
    elif cfg.natural_gradients:
        key, fit_key = jr.split(key)
        opt_vf, _ = natural_gradient_svgp_fit(
            posterior,
            z_init,
            data,
            natural_lr=cfg.natural_lr,
            hyper_optimizer=ox.adam(cfg.kernel_lr),
            num_iters=cfg.gp_num_iters,
            batch_size=cfg.vgp_batch_size,
            key=fit_key,
            progress_bar=True,
        )
    else:
        variational_family = gpx.variational_families.VariationalGaussian(
            posterior=posterior,
            inducing_inputs=z_init,
        )
        objective = lambda p, d: -gpx.objectives.elbo(p, d)  # noqa: E731
        opt_vf, _ = gpx.fit(
            model=variational_family,
            objective=objective,
            train_data=data,
            optim=ox.adam(cfg.kernel_lr),
            num_iters=cfg.gp_num_iters,
            verbose=True,
            batch_size=cfg.vgp_batch_size,
        )

    opt_kernel = opt_vf.posterior.prior.kernel
    opt_sigma = opt_vf.posterior.likelihood.obs_stddev
    z_opt = np.array(px.unwrap(opt_vf.inducing_inputs))

    # --- Evaluate ------------------------------------------------------------
    posterior_ = opt_vf.posterior
    latent = opt_vf.predict(x_test, train_data=data) if cfg.collapsed else opt_vf.predict(x_test)
    predictive = posterior_.likelihood(latent)
    mean = predictive.mean
    std = jnp.sqrt(predictive.variance)

    metrics = {
        "vgp_nlpd": float(nlpd_gp(y_test, mean, std)),
        "gp_sigma": float(np.array(px.unwrap(opt_sigma)).reshape(())),
        "collapsed": bool(cfg.collapsed),
        "natural_gradients": bool(cfg.natural_gradients),
    }
    log.info("VGP  NLPD=%.4f", metrics["vgp_nlpd"])

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
