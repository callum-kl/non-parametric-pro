"""Graph PrO-GP / exact graph GP posteriors over the PeMS road network, drawn with the
benchmark's own `plot_prediction` / `plot_uncertainty` (vendored in
`pems_regression_plotting.py`)."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")

import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import paramax as px
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_pro import run_pro_inference
from pems_regression_plotting import plot_prediction, plot_uncertainty
from util import load_gp_state

from non_parametric_pro.data.pems.pems import load_pems_graph_data, pems_regression_split
from non_parametric_pro.util import prediction_basis, predictive_moments

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

# Upstream's notebook settings, so these panels are directly comparable to the
# benchmark's published ones.
SPEED_VMIN, SPEED_VMAX = 10, 70
STD_VMIN, STD_VMAX = 0, 18
ALPHA_GLOBAL, ALPHA_LOCAL = 0.6, 0.95

# (n, s, e, w). Upstream frames the network almost exactly -- 26.6 x 26.7 km, so the
# panels come out square; widening the longitude window leaves the geography alone
# (osmnx sets aspect 1/cos(lat)) and just gives a landscape frame with more context.
BBOX = (37.450, 37.210, -121.77, -122.13)


def _pro_posterior(cfg, kernel, x_train, y_train, x_all, gp_sigma):
    sigma_init = gp_sigma if cfg.sigma_init is None else cfg.sigma_init
    fit = run_pro_inference(
        cfg, jr.PRNGKey(cfg.seed), kernel, x_train, y_train, sigma_init
    )
    basis, covariance = prediction_basis(fit.kernel, x_train, x_all, fit.pro_params)
    residual_std = jnp.sqrt(
        jnp.maximum(jnp.diag(covariance) - jnp.sum(basis**2, axis=1), 0.0)
    )
    return predictive_moments(
        basis,
        fit.particles,
        noise_std=px.unwrap(fit.pro_params.sigma),
        residual_std=residual_std,
    )


def _gp_posterior(kernel, x_train, y_train, x_all, gp_sigma):
    """The exact GP conditioned on the same split, with the kernel and noise
    `fit_exact_gp.py` already fitted -- nothing is refitted here."""
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=x_train.shape[0], obs_stddev=jnp.array(gp_sigma)
    )
    posterior = prior * likelihood
    predictive = posterior.likelihood(
        posterior.predict(x_all, train_data=gpx.Dataset(X=x_train, y=y_train))
    )
    return predictive.mean, jnp.sqrt(predictive.variance)


def predict_all_nodes(cfg, graph_data, *, method: str):
    """Posterior at every node of the road graph, in standardised units
    (`plot_PEMS` un-normalises with the scaler's mean/scale itself)."""
    kernel, gp_sigma, scaler_y = load_gp_state(cfg)
    split = pems_regression_split(
        graph_data,
        cfg.split,
        num_train=cfg.num_train,
        seed=cfg.split + cfg.split_seed_offset,
    )
    x_train = jnp.asarray(split.x_train)
    y_train = jnp.asarray(scaler_y.transform(split.y_train))
    x_all = jnp.arange(graph_data.num_nodes, dtype=x_train.dtype).reshape(-1, 1)

    if method == "pro":
        mean, std = _pro_posterior(cfg, kernel, x_train, y_train, x_all, gp_sigma)
    else:
        mean, std = _gp_posterior(kernel, x_train, y_train, x_all, gp_sigma)

    scale, offset = float(scaler_y.scale_[0]), float(scaler_y.mean_[0])
    for name, x, y in (
        ("train", split.x_train, split.y_train),
        ("test", split.x_test, split.y_test),
    ):
        residual = np.asarray(mean)[x.ravel()] * scale + offset - y.ravel()
        print(
            f"{name}: RMSE={np.sqrt(np.mean(residual**2)):.2f} mph, "
            f"mean predictive std={np.asarray(std)[x.ravel()].mean() * scale:.2f} mph"
        )

    return np.asarray(mean), np.asarray(std), split, scaler_y


def _save(out_path: Path) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.gcf().savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"Saved {out_path}")


def plot_road_map(cfg, *, method: str, prefix: str) -> None:
    graph_data = load_pems_graph_data()
    mean, std, split, scaler_y = predict_all_nodes(cfg, graph_data, method=method)

    shared = {
        "nx_graph": graph_data.graph,
        "xs": np.arange(graph_data.num_nodes),
        "xs_train": split.x_train,
        "alpha_global": ALPHA_GLOBAL,
        "alpha_local": ALPHA_LOCAL,
        "bbox": BBOX,
    }

    plot_prediction(
        mean,
        orig_mean=scaler_y.mean_[0],
        orig_std=scaler_y.scale_[0],
        vmin=SPEED_VMIN,
        vmax=SPEED_VMAX,
        **shared,
    )
    _save(FIGURES_DIR / f"{prefix}_prediction.png")

    plot_uncertainty(
        std,
        orig_std=scaler_y.scale_[0],
        vmin=STD_VMIN,
        vmax=STD_VMAX,
        **shared,
    )
    _save(FIGURES_DIR / f"{prefix}_uncertainty.png")


def load_config(name: str, overrides: list[str]):
    cfg = OmegaConf.load(CONF_DIR / f"{name}.yaml")
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="fit_pro_gibbs")
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--num-train", type=int, default=250)
    parser.add_argument(
        "--method",
        default="pro",
        choices=["pro", "gp"],
        help="pro: PrO-GP sampling; gp: the exact graph GP from the saved gp_state",
    )
    parser.add_argument(
        "--prefix", default=None, help="figure name stem (default: <method>_road_map)"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="extra dotlist config overrides, e.g. num_particles=50",
    )
    args = parser.parse_args()

    cfg = load_config(
        args.config_name,
        [f"split={args.split}", f"num_train={args.num_train}", *args.overrides],
    )
    prefix = args.prefix or {"pro": "pro_gp_road_map", "gp": "graph_gp_road_map"}[args.method]
    plot_road_map(cfg, method=args.method, prefix=prefix)
