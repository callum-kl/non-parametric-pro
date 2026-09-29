"""Graph PRO-GP with replica Gibbs, using the kernel of a saved exact graph GP."""

import json
import logging
import os
from typing import NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import hydra
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px
from omegaconf import DictConfig, OmegaConf
from util import load_gp_state, load_split, pro_out_dir

from non_parametric_pro import replica_gibbs
from non_parametric_pro.data.pems.pems import load_pems_graph_data
from non_parametric_pro.density import ProParameters, pro_logdensity_fn
from non_parametric_pro.inducing import PointInducingBasis
from non_parametric_pro.parameter_adaptation import parameter_adaptation
from non_parametric_pro.replica_gibbs import parametric_replica_gibbs, validate_r
from non_parametric_pro.util import (
    cholesky_basis,
    nlpd_pro,
    prediction_basis,
    run_inference_algorithm_with_burn_in,
    train_val_split,
)

log = logging.getLogger(__name__)


class ProInference(NamedTuple):
    particles: jnp.ndarray
    pro_params: ProParameters


def run_pro_inference(cfg, key, kernel, x_train, y_train, sigma_init) -> ProInference:
    """Adapt sigma on a train/val split, then sample particles against the full training set."""
    validate_r(cfg.num_particles, cfg.alpha)
    basis_full = cholesky_basis(kernel, x_train)

    key, split_key, pos_key, adapt_key = jr.split(key, 4)
    tv_split = train_val_split(
        split_key, x_train, y_train, val_fraction=cfg.val_fraction
    )

    fold_params = ProParameters(
        y=tv_split.y_train,
        basis=None,
        step_size=None,
        sigma=gpx.parameters.SigmoidBounded(
            sigma_init, low=cfg.sigma_min, high=cfg.sigma_max
        ),
        alpha=cfg.alpha,
        residual_std=None,
    )
    initial_position = jr.normal(pos_key, (basis_full.shape[1], cfg.num_particles))

    adaptation = parameter_adaptation(
        replica_gibbs,
        pro_logdensity_fn,
        fold_params,
        x_train=tv_split.x_train,
        initial_kernel=kernel,
        warmup_steps=cfg.warmup_steps,
        sigma_adapt_steps=cfg.sigma_adapt_steps,
        kernel_adapt_steps=0,
        objective_fn=pro_logdensity_fn,
        inducing_basis=PointInducingBasis(z=x_train),
        x_val=tv_split.x_val,
        y_val=tv_split.y_val,
        sigma_optimizer=ox.adam(cfg.sigma_lr),
        progress_bar=True,
    )
    adaptation_results, _ = adaptation.run(
        adapt_key, initial_position, num_steps=cfg.num_adapt_steps
    )
    log.info(
        "Adapted sigma=%.4f", float(np.array(px.unwrap(adaptation_results.parameters.sigma)))
    )

    # Warm-started from adaptation, but now against the full training set (train + val),
    # so a real burn-in is warranted.
    pro_params = ProParameters(
        y=y_train,
        basis=basis_full,
        step_size=None,
        sigma=adaptation_results.parameters.sigma,
        alpha=cfg.alpha,
        residual_std=None,
    )
    _, sample_key = jr.split(key)
    _, (states, _) = run_inference_algorithm_with_burn_in(
        rng_key=sample_key,
        inference_algorithm=parametric_replica_gibbs(pro_logdensity_fn, pro_params),
        num_steps=cfg.num_sample_steps,
        burn_ratio=cfg.burn_fraction,
        initial_position=adaptation_results.state.position,
        progress_bar=True,
    )
    return ProInference(particles=states.position[:: cfg.thin], pro_params=pro_params)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_pro")
def main(cfg: DictConfig) -> None:
    log.info("PRO (graph GP): split=%d num_train=%d", cfg.split, cfg.num_train)

    out_dir = pro_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel, gp_sigma_val, scaler_y = load_gp_state(cfg)
    sigma_init = gp_sigma_val if cfg.sigma_init is None else cfg.sigma_init

    split = load_split(cfg, load_pems_graph_data())
    x_train = jnp.asarray(split.x_train)
    y_train = jnp.asarray(scaler_y.transform(split.y_train))
    x_test = jnp.asarray(split.x_test)
    y_test = jnp.asarray(scaler_y.transform(split.y_test))
    log.info("N_train=%d  N_test=%d", x_train.shape[0], x_test.shape[0])

    particles, pro_params = run_pro_inference(
        cfg, jr.PRNGKey(cfg.seed), kernel, x_train, y_train, sigma_init
    )

    test_basis, test_cov = prediction_basis(kernel, x_train, x_test, pro_params)
    metrics = {
        "split": cfg.split,
        "num_train": cfg.num_train,
        "pro_nlpd": float(
            nlpd_pro(y_test, test_basis, test_cov, particles, parameters=pro_params)
        ),
    }
    log.info("PRO  NLPD=%.4f", metrics["pro_nlpd"])

    with open(out_dir / "pro_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "pro_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
