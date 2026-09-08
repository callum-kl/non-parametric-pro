"""Fit an exact GP with a graph Matern kernel on a PeMS road-network split."""

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

from non_parametric_pro.data.pems.pems import load_pems_graph_data, pems_regression_split
from non_parametric_pro.util import nlpd_gp, pit_values, variogram_score

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


def state_dir(cfg: DictConfig) -> Path:
    subdir = f"exact_gp_{cfg.name}" if cfg.name else "exact_gp"
    return Path(cfg.results_root) / f"num_train_{cfg.num_train}" / f"split_{cfg.split}" / subdir


def _objective_fn(name: str):
    if name == "mll":
        return lambda p, d: -gpx.objectives.conjugate_mll(p, d)
    if name == "loocv":
        return lambda p, d: -gpx.objectives.conjugate_loocv(p, d)
    msg = f"Unknown objective={name!r}; expected 'mll' or 'loocv'."
    raise ValueError(msg)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_exact_gp")
def main(cfg: DictConfig) -> None:
    log.info("Fitting exact graph GP: split=%d num_train=%d", cfg.split, cfg.num_train)

    out_dir = state_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ----------------------------------------------------------------
    graph_data = load_pems_graph_data()
    split = pems_regression_split(
        graph_data, cfg.split, num_train=cfg.num_train, seed=cfg.split + cfg.split_seed_offset
    )

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

    # --- Exact GP fit --------------------------------------------------------
    data = gpx.Dataset(X=x_train, y=y_train)
    key = jr.PRNGKey(cfg.seed)
    restart_keys = jr.split(key, cfg.num_restarts)
    objective_fn = _objective_fn(cfg.objective)

    best_posterior = None
    best_loss = jnp.inf
    for i, restart_key in enumerate(restart_keys):
        if i == 0:
            init_lengthscale = jnp.array(cfg.lengthscale_init)
            init_smoothness = jnp.array(cfg.smoothness_init)
            init_variance = jnp.array(1.0)
        else:
            ls_key, sm_key, var_key = jr.split(restart_key, 3)
            ls_jitter = cfg.restart_jitter_std * jr.normal(ls_key, ())
            sm_jitter = cfg.restart_jitter_std * jr.normal(sm_key, ())
            var_jitter = cfg.restart_jitter_std * jr.normal(var_key, ())
            init_lengthscale = jnp.clip(
                cfg.lengthscale_init * jnp.exp(ls_jitter),
                cfg.lengthscale_min * 1.01,
                cfg.lengthscale_max * 0.99,
            )
            init_smoothness = jnp.clip(
                cfg.smoothness_init * jnp.exp(sm_jitter),
                cfg.smoothness_min * 1.01,
                cfg.smoothness_max * 0.99,
            )
            init_variance = jnp.exp(var_jitter)  # lognormal jitter around 1.0

        lengthscale = gpx.parameters.SigmoidBounded(
            init_lengthscale, low=cfg.lengthscale_min, high=cfg.lengthscale_max
        )
        smoothness = gpx.parameters.SigmoidBounded(
            init_smoothness, low=cfg.smoothness_min, high=cfg.smoothness_max
        )
        variance = gpx.parameters.PositiveReal(init_variance)
        kernel = gpx.kernels.GraphKernel(
            laplacian=graph_data.laplacian,
            lengthscale=lengthscale,
            variance=variance,
            smoothness=smoothness,
        )
        prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=data.n, obs_stddev=jnp.sqrt(0.01)
        )
        posterior = prior * likelihood

        candidate, history = gpx.fit(
            model=posterior,
            objective=objective_fn,
            train_data=data,
            optim=ox.adam(cfg.kernel_lr),
            num_iters=cfg.num_iters,
            verbose=(cfg.num_restarts == 1),
        )
        final_loss = float(history[-1])
        log.info(
            "Restart %d/%d: final negative %s=%.4f",
            i + 1,
            cfg.num_restarts,
            cfg.objective,
            final_loss,
        )
        if final_loss < best_loss:
            best_loss = final_loss
            best_posterior = candidate

    opt_posterior = best_posterior
    log.info("Best restart: negative %s=%.4f", cfg.objective, best_loss)

    opt_kernel = opt_posterior.prior.kernel
    opt_sigma = opt_posterior.likelihood.obs_stddev

    log.info("lengthscale=%s", px.unwrap(opt_kernel.lengthscale))
    log.info("smoothness=%s", px.unwrap(opt_kernel.smoothness))
    log.info("variance=%s", px.unwrap(opt_kernel.variance))

    # --- Evaluate ------------------------------------------------------------
    latent = opt_posterior.predict(x_test, train_data=data)
    predictive = opt_posterior.likelihood(latent)
    mean = predictive.mean
    std = jnp.sqrt(predictive.variance)
    y_test_flat = jnp.asarray(y_test).squeeze()

    draw_key = jr.fold_in(key, 0)
    gp_draws = jnp.transpose(predictive.sample(draw_key, (cfg.num_predictive_draws,)))
    gp_variogram = float(variogram_score(gp_draws, y_test_flat))
    gp_pit = pit_values(gp_draws, y_test_flat)

    metrics = {
        "split": cfg.split,
        "num_train": cfg.num_train,
        "gp_nlpd": float(nlpd_gp(y_test, mean, std)),
        "gp_variogram_score": gp_variogram,
        "gp_pit": [float(v) for v in gp_pit],
    }

    log.info(
        "GP  NLPD=%.4f  variogram=%.4f",
        metrics["gp_nlpd"],
        gp_variogram,
    )

    # --- Save ----------------------------------------------------------------
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
