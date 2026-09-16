"""Fit a non-collapsed VGP with a chunked Adam loop, recording the objective and wall-clock per chunk."""

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import equinox as eqx
import gpjax as gpx
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px
from gpjax.fit import get_batch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from util import build_vgp_posterior, gp_state_dir, save_gp_state

from non_parametric_pro.data.uci.uci import load_uci_regression_dataset
from non_parametric_pro.inducing import kmeans_inducing_points
from non_parametric_pro.util import nlpd_gp

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


@hydra.main(version_base=None, config_path="../conf", config_name="convergence")
def main(cfg: DictConfig) -> None:
    log.info(
        "VGP convergence: dataset=%s split=%d m=%d seed=%d",
        cfg.dataset,
        cfg.split,
        cfg.num_inducing,
        cfg.seed,
    )
    if cfg.gp_num_iters % cfg.vgp_chunk_len:
        raise ValueError("gp_num_iters must be a multiple of vgp_chunk_len.")

    key = jr.PRNGKey(cfg.seed)
    out_dir = gp_state_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train = jnp.array(scaler_x.fit_transform(example.x_train))
    y_train = jnp.array(scaler_y.fit_transform(example.y_train))
    x_test = jnp.array(scaler_x.transform(example.x_test))
    y_test = jnp.array(scaler_y.transform(example.y_test))
    data = gpx.Dataset(X=x_train, y=y_train)
    log.info("N_train=%d  D=%d", data.n, x_train.shape[1])

    posterior = build_vgp_posterior(x_train, cfg)
    key, km_key, fit_key = jr.split(key, 3)
    z_init = kmeans_inducing_points(km_key, x_train, cfg.num_inducing).z
    model = gpx.variational_families.VariationalGaussian(
        posterior=posterior, inducing_inputs=z_init
    )

    optim = ox.adam(cfg.kernel_lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def loss(model, batch):
        return -gpx.objectives.elbo(px.unwrap(model), batch)

    # Mirrors the step in gpx.fit, but chunked so each chunk can be timed.
    @eqx.filter_jit
    def run_chunk(model, opt_state, keys):
        def step(carry, key):
            model, opt_state = carry
            batch = get_batch(data, cfg.vgp_batch_size, key)
            loss_val, grads = eqx.filter_value_and_grad(loss)(model, batch)
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            return (eqx.apply_updates(model, updates), opt_state), loss_val

        (model, opt_state), losses = jax.lax.scan(step, (model, opt_state), keys)
        return model, opt_state, losses

    full_loss = eqx.filter_jit(loss)

    num_chunks = cfg.gp_num_iters // cfg.vgp_chunk_len
    chunk_keys = jr.split(fit_key, cfg.gp_num_iters).reshape(
        num_chunks, cfg.vgp_chunk_len, -1
    )
    minibatch_loss, full_neg_elbo, chunk_times = [], [], []
    for i in range(num_chunks):
        start = time.perf_counter()
        model, opt_state, losses = run_chunk(model, opt_state, chunk_keys[i])
        losses = np.asarray(losses)
        chunk_times.append(time.perf_counter() - start)
        minibatch_loss.append(losses)
        full_neg_elbo.append(float(full_loss(model, data)))
        if (i + 1) % 10 == 0:
            log.info(
                "iter %d  full -ELBO/n=%.4f  chunk %.3fs",
                (i + 1) * cfg.vgp_chunk_len,
                full_neg_elbo[-1] / data.n,
                chunk_times[-1],
            )

    np.savez(
        out_dir / "convergence.npz",
        minibatch_loss=np.concatenate(minibatch_loss),
        full_neg_elbo=np.array(full_neg_elbo),
        chunk_times=np.array(chunk_times),
        chunk_len=cfg.vgp_chunk_len,
        n_train=data.n,
        m=cfg.num_inducing,
    )

    predictive = model.posterior.likelihood(model.predict(x_test))
    metrics = {
        "vgp_nlpd": float(
            nlpd_gp(y_test, predictive.mean, jnp.sqrt(predictive.variance))
        ),
        "gp_sigma": float(
            np.array(px.unwrap(model.posterior.likelihood.obs_stddev)).reshape(())
        ),
        "num_inducing": cfg.num_inducing,
        "median_seconds_per_iter": float(
            np.median(chunk_times[1:]) / cfg.vgp_chunk_len
        ),
    }
    log.info("VGP  NLPD=%.4f", metrics["vgp_nlpd"])

    save_gp_state(out_dir, model, scaler_x, scaler_y)
    with open(out_dir / "gp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "gp_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
    log.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()
