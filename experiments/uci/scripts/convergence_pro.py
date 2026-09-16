"""Run replica Gibbs with a fixed VGP kernel and pre-adapted sigma, recording the training score and wall-clock per chunk."""

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from omegaconf import DictConfig, OmegaConf
from util import load_gp_state, pro_out_dir

from non_parametric_pro.data.uci.uci import load_uci_regression_dataset
from non_parametric_pro.density import ProParameters, pro_logdensity_fn
from non_parametric_pro.inducing import compute_inducing_basis
from non_parametric_pro.replica_gibbs import parametric_replica_gibbs, validate_r

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


@hydra.main(version_base=None, config_path="../conf", config_name="convergence")
def main(cfg: DictConfig) -> None:
    log.info(
        "PrO convergence: dataset=%s split=%d m=%d seed=%d",
        cfg.dataset,
        cfg.split,
        cfg.num_inducing,
        cfg.seed,
    )
    validate_r(cfg.num_particles, cfg.alpha)
    if cfg.num_gibbs_iters % cfg.gibbs_chunk_len:
        raise ValueError("num_gibbs_iters must be a multiple of gibbs_chunk_len.")

    out_dir = pro_out_dir(cfg)
    with open(out_dir / "pro_metrics.json") as f:
        sigma = json.load(f)["pro_sigma"]

    kernel, _, inducing_basis, scaler_x, scaler_y = load_gp_state(cfg)
    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train)
    n_train = x_train.shape[0]

    basis, residual_std = compute_inducing_basis(inducing_basis, kernel, x_train)
    parameters = ProParameters(
        y=y_train,
        basis=basis,
        step_size=None,
        sigma=sigma,
        alpha=cfg.alpha,
        residual_std=residual_std,
    )
    algorithm = parametric_replica_gibbs(pro_logdensity_fn, parameters)

    key = jr.PRNGKey(cfg.seed)
    init_key, run_key = jr.split(key)
    state = algorithm.init(jr.normal(init_key, (basis.shape[1], cfg.num_particles)))

    @jax.jit
    def run_chunk(state, keys):
        return jax.lax.scan(lambda s, k: algorithm.step(k, s), state, keys)

    num_chunks = cfg.num_gibbs_iters // cfg.gibbs_chunk_len
    chunk_keys = jr.split(run_key, cfg.num_gibbs_iters).reshape(
        num_chunks, cfg.gibbs_chunk_len, -1
    )
    scores, chunk_times = [], []
    for i in range(num_chunks):
        start = time.perf_counter()
        state, info = run_chunk(state, chunk_keys[i])
        score = np.asarray(info.score)
        chunk_times.append(time.perf_counter() - start)
        scores.append(score)

    score = np.concatenate(scores)
    avg_score = score / (cfg.num_particles * cfg.alpha * n_train)
    log.info(
        "avg score first=%.4f last=%.4f  median %.4fs/iter",
        avg_score[0],
        avg_score[-1],
        np.median(chunk_times[1:]) / cfg.gibbs_chunk_len,
    )

    np.savez(
        out_dir / "convergence.npz",
        score=score,
        avg_score=avg_score,
        chunk_times=np.array(chunk_times),
        chunk_len=cfg.gibbs_chunk_len,
        n_train=n_train,
        m=cfg.num_inducing,
        sigma=sigma,
    )
    log.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()
