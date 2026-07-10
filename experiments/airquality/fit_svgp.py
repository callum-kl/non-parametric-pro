"""Fit a standard SVGP on one Kampala air-quality site, save its RMSE/NLPD/CRPS.

Ports ``sparse_approximations/sparse_gp.py`` from
https://github.com/claramst/gps-kampala-airquality (see
``non_parametric_pro.data.kampala_airquality`` for what is/isn't reproduced exactly).

Fits one site per invocation (mirroring how ``experiments/uci/fit_exact_gp.py`` fits one
dataset+split per invocation); sweep all sites with hydra multirun, e.g.:

    python fit_svgp.py -m mode=forecasting site_index="range(0,66)"
"""

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

from non_parametric_pro.data.kampala_airquality import (
    KampalaAirQualityRecords,
    KampalaSplit,
    kampala_forecasting_split,
    kampala_nowcasting_split,
    kampala_outlier_mask,
    kampala_site_ids,
    load_kampala_airquality_records,
)
from non_parametric_pro.inducing import kmeans_inducing_points
from non_parametric_pro.util import crps_gp, nlpd_gp

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)

_SPLIT_FNS = {"forecasting": kampala_forecasting_split, "nowcasting": kampala_nowcasting_split}


def state_dir(cfg: DictConfig, site_id: str) -> Path:
    method = "svgp" if cfg.remove_outliers else "svgp_no_outliers"
    return Path(cfg.results_root) / cfg.mode / site_id / method


def _load_split(cfg: DictConfig, records: KampalaAirQualityRecords, site_id: str) -> KampalaSplit:
    split_fn = _SPLIT_FNS[cfg.mode]
    outlier_mask = kampala_outlier_mask(records, iqr_multiplier=cfg.outlier_iqr_multiplier) if cfg.remove_outliers else None
    return split_fn(records, site_id, fold=cfg.fold, max_train=cfg.max_train, outlier_mask=outlier_mask)


@hydra.main(version_base=None, config_path="conf", config_name="fit_svgp")
def main(cfg: DictConfig) -> None:
    if cfg.mode not in _SPLIT_FNS:
        msg = f"Unknown mode: {cfg.mode!r}. Expected 'forecasting' or 'nowcasting'."
        raise ValueError(msg)

    records = load_kampala_airquality_records()
    site_ids = kampala_site_ids(records)
    if not 0 <= cfg.site_index < len(site_ids):
        msg = f"site_index must be in 0..{len(site_ids) - 1}, got {cfg.site_index}"
        raise ValueError(msg)
    site_id = site_ids[cfg.site_index]

    log.info(
        "Fitting SVGP: mode=%s site_id=%s (index %d/%d) remove_outliers=%s",
        cfg.mode, site_id, cfg.site_index, len(site_ids), cfg.remove_outliers,
    )

    split = _load_split(cfg, records, site_id)
    x_train, y_train = jnp.array(split.x_train), jnp.array(split.y_train)
    x_test, y_test = jnp.array(split.x_test), jnp.array(split.y_test)

    D = x_train.shape[1]
    N = x_train.shape[0]
    num_inducing = min(cfg.num_inducing, N)
    log.info("N_train=%d  N_test=%d  D=%d  num_inducing=%d", N, x_test.shape[0], D, num_inducing)

    # --- SVGP fit --------------------------------------------------------------
    data = gpx.Dataset(X=x_train, y=y_train)
    kernel = gpx.kernels.RBF(lengthscale=jnp.ones((D,)), variance=px.NonTrainable(jnp.array(1.0)))
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n)
    posterior = prior * likelihood

    key = jr.PRNGKey(cfg.seed)
    z_init = kmeans_inducing_points(key, x_train, num_inducing).z
    variational_family = gpx.variational_families.VariationalGaussian(
        posterior=posterior, inducing_inputs=z_init,
    )

    opt_variational_family, _ = gpx.fit(
        model=variational_family,
        objective=lambda p, d: -gpx.objectives.elbo(p, d),
        train_data=data,
        optim=ox.adam(cfg.kernel_lr),
        num_iters=cfg.num_iters,
        verbose=False,
        batch_size=cfg.batch_size,
    )

    # --- Evaluate ----------------------------------------------------------------
    # Predict from the variational family itself (cost depends only on num_inducing, not
    # N) -- NOT opt_variational_family.posterior.predict(x_test, train_data=...), which
    # would run exact (O(N^3)) GP inference on the full training set instead.
    latent = opt_variational_family.predict(x_test)
    predictive = opt_variational_family.posterior.likelihood(latent)
    mean, std = predictive.mean, jnp.sqrt(predictive.variance)

    y_test_raw = split.y_std * y_test.squeeze() + split.y_mean
    mean_raw = split.y_std * mean + split.y_mean
    rmse_raw = float(jnp.sqrt(jnp.mean((y_test_raw - mean_raw) ** 2)))

    metrics = {
        "site_id": site_id,
        "mode": cfg.mode,
        "fold": cfg.fold,
        "remove_outliers": bool(cfg.remove_outliers),
        "num_inducing": num_inducing,
        "n_train": N,
        "n_test": int(x_test.shape[0]),
        "nlpd": float(nlpd_gp(y_test, mean, std)),
        "crps": float(crps_gp(y_test, mean, std)),
        "rmse": rmse_raw,
    }
    log.info(
        "SVGP site=%s NLPD=%.4f CRPS=%.4f RMSE=%.4f",
        site_id, metrics["nlpd"], metrics["crps"], metrics["rmse"],
    )

    # --- Save ----------------------------------------------------------------
    out_dir = state_dir(cfg, site_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    z_opt = np.array(px.unwrap(opt_variational_family.inducing_inputs))
    np.savez(
        out_dir / "svgp_state.npz",
        lengthscale=np.array(px.unwrap(opt_variational_family.posterior.prior.kernel.lengthscale)),
        sigma=np.array(px.unwrap(opt_variational_family.posterior.likelihood.obs_stddev)).reshape(()),
        z=z_opt,
        y_mean=split.y_mean,
        y_std=split.y_std,
    )

    with open(out_dir / "svgp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "svgp_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
