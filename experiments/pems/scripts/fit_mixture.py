"""Bayesian overlapping mixture of graph GPs (OMGP), using the kernel of a saved exact graph GP."""

import json
import logging
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import hydra
import jax.numpy as jnp
import jax.random as jr
from omegaconf import DictConfig, OmegaConf
from util import load_gp_state, load_split, split_dir

from non_parametric_pro.data.pems.pems import load_pems_graph_data
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
    log.info("OMGP (graph GP): split=%d num_train=%d", cfg.split, cfg.num_train)
    kernel, gp_sigma_val, scaler_y = load_gp_state(cfg)

    split = load_split(cfg, load_pems_graph_data())
    x_train = jnp.asarray(split.x_train)
    y_train = jnp.asarray(scaler_y.transform(split.y_train))
    x_test = jnp.asarray(split.x_test)
    y_test = jnp.asarray(scaler_y.transform(split.y_test))
    log.info("N_train=%d  N_test=%d", x_train.shape[0], x_test.shape[0])

    config = OmegaConf.to_object(cfg)
    results = fit_omgp_variants(
        jr.PRNGKey(cfg.seed),
        kernel,
        x_train,
        y_train,
        x_test,
        y_test,
        sigma_init=gp_sigma_val,
        **{k: config[k] for k in MIXTURE_KEYS},
    )

    for name, result in results.items():
        metrics = {
            "split": cfg.split,
            "num_train": cfg.num_train,
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
