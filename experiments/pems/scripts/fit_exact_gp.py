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

from non_parametric_pro.data.pems.pems import (
    load_pems_graph_data,
    pems_regression_split,
)
from non_parametric_pro.util import energy_score, nlpd_gp, rmse

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


def state_dir(cfg: DictConfig) -> Path:
    subdir = f"exact_gp_{cfg.name}" if cfg.name else "exact_gp"
    return Path(cfg.results_root) / f"split_{cfg.split}" / subdir


def _objective_fn(name: str):
    if name == "mll":
        return lambda p, d: -gpx.objectives.conjugate_mll(p, d)
    if name == "loocv":
        return lambda p, d: -gpx.objectives.conjugate_loocv(p, d)
    msg = f"Unknown objective={name!r}; expected 'mll' or 'loocv'."
    raise ValueError(msg)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_exact_gp")
def main(cfg: DictConfig) -> None:
    log.info("Fitting exact graph GP: split=%d", cfg.split)

    out_dir = state_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ----------------------------------------------------------------
    graph_data = load_pems_graph_data()
    split = pems_regression_split(graph_data, cfg.split, num_train=cfg.num_train)

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
        # Trainable (not NonTrainable(1.0)) -- see discussion: standardizing y removes
        # the y-scale/kernel-variance redundancy, but variance and noise sigma remain
        # two independent degrees of freedom in the MLL (variance controls how much of
        # the standardized-y variance the smooth graph signal explains; sigma explains
        # the rest), so fixing variance=1 does cost real flexibility relative to this.
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

    mean_raw = scaler_y.inverse_transform(np.asarray(mean).reshape(-1, 1)).squeeze()

    # diag/full NLL follow the pems-regression notebook's convention (metrics.py's
    # diag_cov_nll/full_cov_nll): the *summed* (not averaged) joint NLL of the whole
    # test vector, either treating test points as independent (diag, matches gp_nlpd
    # * N_test exactly) or under the full correlated predictive covariance (full).
    # These exist for direct comparison against that notebook; gp_nlpd (mean
    # per-point) stays the primary metric, matching this repo's other experiments.
    y_test_flat = jnp.asarray(y_test).squeeze()
    gp_diag_nll = float(jnp.sum(nlpd_gp(y_test, mean, std, return_per_point=True)))
    gp_full_nll = float(-predictive.log_prob(y_test_flat))

    # Energy score, for direct comparison against PrO's energy score: a single
    # Gaussian has no particle disagreement to expose, but scoring it via samples
    # (rather than its closed-form log-density) puts it on equal footing with PrO's
    # sample-based score, both estimating the same proper scoring rule.
    draw_key = jr.fold_in(key, 0)
    gp_draws = jnp.transpose(predictive.sample(draw_key, (cfg.energy_score_draws,)))
    gp_energy_score = float(energy_score(gp_draws, y_test_flat))

    metrics = {
        "split": cfg.split,
        "gp_nlpd": float(nlpd_gp(y_test, mean, std)),
        "gp_diag_nll": gp_diag_nll,
        "gp_full_nll": gp_full_nll,
        "gp_energy_score": gp_energy_score,
        "gp_rmse": float(rmse(jnp.asarray(split.y_test), jnp.asarray(mean_raw))),
        "gp_sigma": float(np.array(px.unwrap(opt_sigma)).reshape(())),
        "objective": cfg.objective,
    }
    log.info(
        "GP  NLPD=%.4f  RMSE=%.4f  diag-NLL=%.4f  full-NLL=%.4f  energy=%.4f",
        metrics["gp_nlpd"],
        metrics["gp_rmse"],
        gp_diag_nll,
        gp_full_nll,
        gp_energy_score,
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
