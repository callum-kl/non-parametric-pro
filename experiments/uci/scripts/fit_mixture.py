"""Bayesian overlapping mixture of GPs (OMGP) baselines, seeded from a saved exact GP."""

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
from non_parametric_pro.mixture_gibbs import fit_omgp_variants

log = logging.getLogger(__name__)

MIXTURE_KEYS = (
    "ks",
    "gamma",
    "sparse_k",
    "sparse_gamma",
    "select_k",
    "val_fraction",
    "sigma_prior_shape",
    "num_steps",
    "burn_fraction",
    "thin",
    "shared_sigma",
)


@hydra.main(version_base=None, config_path="../conf", config_name="fit_mixture")
def main(cfg: DictConfig) -> None:
    log.info("OMGP: dataset=%s split=%d inducing=%s", cfg.dataset, cfg.split, cfg.inducing)
    kernel, gp_sigma_val, inducing_basis, scaler_x, scaler_y = load_gp_state(cfg)

    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test)
    log.info("N_train=%d  N_test=%d", x_train.shape[0], x_test.shape[0])

    mixture_kwargs = {k: OmegaConf.to_object(cfg)[k] for k in MIXTURE_KEYS}
    results = fit_omgp_variants(
        jr.PRNGKey(cfg.seed),
        kernel,
        jnp.asarray(x_train),
        jnp.asarray(y_train),
        jnp.asarray(x_test),
        jnp.asarray(y_test),
        sigma_init=gp_sigma_val,
        inducing_basis=inducing_basis,
        **mixture_kwargs,
    )

    for name, result in results.items():
        metrics = {
            "dataset": cfg.dataset,
            "split": cfg.split,
            "gp_sigma": float(gp_sigma_val),
            "mixture_nlpd": float(jnp.mean(result.nlpd_per_point)),
            "runtime": result.runtime,
            "occupied_components": result.occupied,
            "selected_k": result.selected_k,
        }
        log.info("%s  NLPD=%.4f  (%.1fs)", name, metrics["mixture_nlpd"], result.runtime)
        out_dir = split_dir(cfg) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "mixture_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(out_dir / "mixture_config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=2)


if __name__ == "__main__":
    main()
