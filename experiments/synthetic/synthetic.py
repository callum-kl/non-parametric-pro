import argparse
import os
from pathlib import Path

# Must be set before `import jax` (and before any transitive jax import, e.g. via
# gpjax) -- setting it later doesn't reliably take effect before jax's backend
# initializes.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

# This script only ever calls savefig, never show(); force a non-interactive backend
# so it doesn't depend on a GUI toolkit being usable (e.g. Qt's xcb plugin, which
# aborts the whole process if there's no X server -- as under plain WSL).
matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import gpjax as gpx
import optax as ox
import paramax as px

from blackjax.util import run_inference_algorithm
from sklearn.preprocessing import StandardScaler
from scipy import stats

from non_parametric_pro import ula
from non_parametric_pro.density import ProParameters, pro_logdensity_fn, regularised_score
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import prediction_basis, nlpd_gp, nlpd_pro, crps_gp, crps_pro
from non_parametric_pro.data.uci import load_uci_regression_dataset
from non_parametric_pro.parameter_adaptation import parameter_adaptation, cross_validated_parameter_adaptation
from non_parametric_pro.gp import _full_gp_basis
from non_parametric_pro.util import train_val_split, run_inference_algorithm_with_burn_in

from non_parametric_pro.data.heteroskedastic import (
    heteroskedastic_noise_std,
    make_heteroskedastic_instance,
)

DEFAULT_SEED = 2421
NUM_INSTANCES = 20
GRID_SHAPE = (5, 4)  # rows, cols -- rows * cols must equal NUM_INSTANCES
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def _plot_case(ax, data):
    """Plot one HeteroskedasticCase: true curve, +-1sigma/+-2sigma noise bands (both
    evaluated on the full, sorted train+test pool for a smooth curve), and the
    noisy observed training points."""
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_truth_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_truth_sorted = x_full[order], y_truth_full[order]
    sigma_sorted = heteroskedastic_noise_std(x_sorted, data.regions, noise_floor=data.noise_floor)

    ax.plot(x_sorted, y_truth_sorted, color="C0", linewidth=1.5)
    ax.fill_between(
        x_sorted, y_truth_sorted - 2 * sigma_sorted, y_truth_sorted + 2 * sigma_sorted,
        color="C0", alpha=0.15, linewidth=0,
    )
    ax.fill_between(
        x_sorted, y_truth_sorted - sigma_sorted, y_truth_sorted + sigma_sorted,
        color="C0", alpha=0.3, linewidth=0,
    )
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.set_title(
        f"$\\sigma_0$={data.noise_floor:.2f}  $\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}",
        fontsize=9,
    )


def make_heteroscedastic_panel(key):
    keys = jr.split(key, NUM_INSTANCES)
    nrows, ncols = GRID_SHAPE
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for ax, instance_key in zip(axes.flat, keys, strict=True):
        data = make_heteroskedastic_instance(instance_key)
        _plot_case(ax, data)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heteroskedastic_panel.png", dpi=150)


def run_heteroscedastic(key):
    keys = jr.split(key, NUM_INSTANCES)

    fig, ax = plt.subplots()

    all_data = []
    for instance_key in keys:
        all_data.append(make_heteroskedastic_instance(instance_key))

    _plot_case(ax, all_data[12])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heteroskedastic_instance.png", dpi=150)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    key = jr.PRNGKey(args.seed)
    run_heteroscedastic(key)

