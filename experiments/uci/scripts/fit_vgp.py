"""Fit a non-collapsed sparse variational GP on a UCI split and save its hyperparameters."""

import json
import logging
import os

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
from util import build_vgp_posterior, save_gp_state, split_dir

from non_parametric_pro.data.uci.uci import load_uci_regression_dataset
from non_parametric_pro.inducing import kmeans_inducing_points
from non_parametric_pro.util import nlpd_gp

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_vgp")
def main(cfg: DictConfig) -> None:
    log.info("Fitting sparse GP: dataset=%s split=%d", cfg.dataset, cfg.split)

    key = jr.PRNGKey(cfg.seed)
    out_dir = split_dir(cfg) / "vgp_noncollapsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train = jnp.array(scaler_x.fit_transform(example.x_train))
    y_train = jnp.array(scaler_y.fit_transform(example.y_train))
    x_test = jnp.array(scaler_x.transform(example.x_test))
    y_test = jnp.array(scaler_y.transform(example.y_test))

    log.info(
        "N_train=%d  N_test=%d  D=%d", x_train.shape[0], x_test.shape[0], x_train.shape[1]
    )

    data = gpx.Dataset(X=x_train, y=y_train)
    posterior = build_vgp_posterior(x_train, cfg)

    key, km_key = jr.split(key)
    z_init = kmeans_inducing_points(km_key, x_train, cfg.num_inducing).z

    variational_family = gpx.variational_families.VariationalGaussian(
        posterior=posterior,
        inducing_inputs=z_init,
    )
    opt_vf, _ = gpx.fit(
        model=variational_family,
        objective=lambda p, d: -gpx.objectives.elbo(p, d),
        train_data=data,
        optim=ox.adam(cfg.kernel_lr),
        num_iters=cfg.gp_num_iters,
        verbose=True,
        batch_size=cfg.vgp_batch_size,
    )

    opt_kernel = opt_vf.posterior.prior.kernel
    opt_sigma = opt_vf.posterior.likelihood.obs_stddev

    predictive = opt_vf.posterior.likelihood(opt_vf.predict(x_test))
    metrics = {
        "vgp_nlpd": float(
            nlpd_gp(y_test, predictive.mean, jnp.sqrt(predictive.variance))
        ),
        "gp_sigma": float(np.array(px.unwrap(opt_sigma)).reshape(())),
        "gp_variance": float(np.array(px.unwrap(opt_kernel.variance)).reshape(())),
    }
    log.info("VGP  NLPD=%.4f", metrics["vgp_nlpd"])

    save_gp_state(out_dir, opt_vf, scaler_x, scaler_y)
    with open(out_dir / "gp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "gp_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved state to %s", out_dir)


if __name__ == "__main__":
    main()
