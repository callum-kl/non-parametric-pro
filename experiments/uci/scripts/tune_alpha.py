"""
PRO-GP over a grid of alpha (= lambda_n / n), with alpha chosen on a validation split,
and the effective-dimension rule alpha = sqrt(d_eff / n), on the exact-GP datasets.

The parametric rate lambda_n ~ sqrt(n) of McLatchie et al. (2025) balances the KL
complexity (~ d log n) against the empirical score; replacing d by the GP's effective
degrees of freedom d_eff = tr(K (K + sigma^2 I)^-1) gives lambda_n ~ sqrt(n d_eff).
With coincident particles alpha acts as a likelihood temperature (alpha = 1 is Bayes),
so both rules give hot posteriors; the grid is centred on the Bayes scale instead.
"""

import json
import logging
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import hydra
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from fit_pro import run_pro
from omegaconf import DictConfig, OmegaConf
from util import load_gp_state, split_dir

from non_parametric_pro.data.uci.uci import load_uci_regression_dataset
from non_parametric_pro.util import train_val_split

log = logging.getLogger(__name__)


def effective_dof(kernel, x, sigma: float) -> float:
    eigvals = np.linalg.eigvalsh(np.asarray(kernel.gram(jnp.asarray(x)).as_matrix()))
    return float(np.sum(eigvals / (eigvals + sigma**2)))


def round_alpha(alpha: float, num_particles: int) -> float:
    return max(1, round(num_particles * alpha)) / num_particles


def save(out_dir, metrics, cfg) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pro_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "pro_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)


@hydra.main(version_base=None, config_path="../conf", config_name="tune_alpha")
def main(cfg: DictConfig) -> None:
    if cfg.inducing:
        msg = "tune_alpha.py supports the exact-GP datasets only."
        raise ValueError(msg)
    log.info("PRO alpha sweep: dataset=%s split=%d", cfg.dataset, cfg.split)
    kernel, gp_sigma, inducing_basis, scaler_x, scaler_y = load_gp_state(cfg)

    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test)
    n = x_train.shape[0]

    d_eff = effective_dof(kernel, x_train, gp_sigma)
    rule_alpha = round_alpha(np.sqrt(d_eff / n), cfg.num_particles)
    log.info("n=%d  d_eff=%.1f  rule alpha=%.2f", n, d_eff, rule_alpha)

    key = jr.PRNGKey(cfg.seed)

    def fit(alpha, x_tr, y_tr, x_te, y_te, fit_key):
        start = time.perf_counter()
        nlpd, sigma = run_pro(
            cfg, fit_key, kernel, inducing_basis, x_tr, y_tr, x_te, y_te, alpha=alpha
        )
        return nlpd, sigma, time.perf_counter() - start

    base = {"dataset": cfg.dataset, "split": cfg.split, "n": n, "d_eff": d_eff,
            "gp_sigma": float(gp_sigma)}
    full = {}
    for alpha in cfg.alphas:
        nlpd, sigma, runtime = fit(alpha, x_train, y_train, x_test, y_test, key)
        full[alpha] = (nlpd, sigma, runtime)
        log.info("alpha=%.2f  NLPD=%.4f  sigma=%.4f  (%.1fs)", alpha, nlpd, sigma, runtime)
        save(split_dir(cfg) / f"pro_gp_alpha{alpha:g}",
             {**base, "alpha": alpha, "pro_nlpd": nlpd, "pro_sigma": sigma,
              "runtime": runtime}, cfg)

    split = train_val_split(
        jr.fold_in(key, 1), x_train, y_train, val_fraction=cfg.alpha_val_fraction
    )
    val_nlpds, val_runtime = {}, 0.0
    for alpha in cfg.alphas:
        nlpd, _, runtime = fit(
            alpha, split.x_train, split.y_train, split.x_val, split.y_val, jr.fold_in(key, 2)
        )
        val_nlpds[alpha], val_runtime = nlpd, val_runtime + runtime
    best = min(val_nlpds, key=val_nlpds.get)
    nlpd, sigma, runtime = full[best]
    log.info("val-selected alpha=%.2f  NLPD=%.4f", best, nlpd)
    save(split_dir(cfg) / "pro_gp_valalpha",
         {**base, "alpha": best, "selected_alpha": best, "pro_nlpd": nlpd, "pro_sigma": sigma,
          "val_nlpds": {f"{a:g}": v for a, v in val_nlpds.items()},
          "runtime": val_runtime + runtime}, cfg)

    if rule_alpha in full:
        nlpd, sigma, runtime = full[rule_alpha]
    else:
        nlpd, sigma, runtime = fit(rule_alpha, x_train, y_train, x_test, y_test, key)
    log.info("d_eff rule alpha=%.2f  NLPD=%.4f", rule_alpha, nlpd)
    save(split_dir(cfg) / "pro_gp_deffalpha",
         {**base, "alpha": rule_alpha, "pro_nlpd": nlpd, "pro_sigma": sigma,
          "runtime": runtime}, cfg)


if __name__ == "__main__":
    main()
