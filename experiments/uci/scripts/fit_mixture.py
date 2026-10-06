"""Bayesian overlapping mixture of GPs (OMGP) baselines, seeded from a saved exact or sparse GP."""

import json
import logging
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import hydra
import jax.numpy as jnp
import jax.random as jr
from omegaconf import DictConfig, OmegaConf
from util import load_gp_state, split_dir

from non_parametric_pro.data.uci.uci import load_uci_regression_dataset
from non_parametric_pro.mixture_gibbs import fit_omgp

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_mixture")
def main(cfg: DictConfig) -> None:
    log.info("OMGP: dataset=%s split=%d inducing=%s", cfg.dataset, cfg.split, cfg.inducing)
    kernel, gp_sigma_val, inducing_basis, scaler_x, scaler_y = load_gp_state(cfg)

    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = jnp.asarray(scaler_x.transform(example.x_train))
    y_train = jnp.asarray(scaler_y.transform(example.y_train))
    x_test = jnp.asarray(scaler_x.transform(example.x_test))
    y_test = jnp.asarray(scaler_y.transform(example.y_test))
    log.info("N_train=%d  N_test=%d", x_train.shape[0], x_test.shape[0])

    results = fit_omgp(
        jr.PRNGKey(cfg.seed), kernel, x_train, y_train, x_test, y_test,
        ks=list(cfg.ks), sigma_init=gp_sigma_val, gamma=cfg.gamma,
        sigma_prior_shape=cfg.sigma_prior_shape, num_steps=cfg.num_steps,
        burn_fraction=cfg.burn_fraction, thin=cfg.thin, inducing_basis=inducing_basis,
    )

    for k, (nlpd, runtime) in results.items():
        metrics = {
            "dataset": cfg.dataset,
            "split": cfg.split,
            "gp_sigma": float(gp_sigma_val),
            "mixture_nlpd": float(jnp.mean(nlpd)),
            "runtime": runtime,
        }
        log.info("K=%d  NLPD=%.4f  (%.1fs)", k, metrics["mixture_nlpd"], runtime)
        out_dir = split_dir(cfg) / f"omgp_shared_k{k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "mixture_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(out_dir / "mixture_config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=2)


if __name__ == "__main__":
    main()
