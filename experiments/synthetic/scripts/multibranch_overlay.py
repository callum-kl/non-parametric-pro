"""
One three-branch instance fitted by OMGP with K=2, OMGP with K=3 and PrO-GP: the
mixture needs the right K, PrO-GP is never told.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")

import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from example_grid import PRO_BAND_ALPHAS, PRO_COLOR, _pro_density_bands, _style_box
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from omegaconf import OmegaConf
from synthetic import fit_gp, fit_pro, gp_kernel_hyperparameters

from non_parametric_pro.data.synthetic.multibranch import make_multibranch_instance
from non_parametric_pro.mixture_gibbs import fit_mixture_gp

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
CONF = OmegaConf.load(SCRIPT_DIR.parent / "conf" / "synthetic.yaml")
DS_CONF = OmegaConf.load(SCRIPT_DIR.parent / "conf" / "ds" / "multibranch.yaml")

OMGP_COLOR = "#3b6fb6"
DATA_COLOR = "#707070"
NUM_GRID = 200
NUM_COMPONENT_DRAWS = 400


def omgp_bands(ax, data, kernel, sigma_init, num_components, key) -> None:
    mixture = CONF.mixture
    fit = fit_mixture_gp(
        key, kernel, data.x_train, data.y_train, data.x_test, data.y_test,
        num_components=num_components, gamma=mixture.gamma, sigma_init=sigma_init,
        sigma_prior_shape=mixture.sigma_prior_shape, num_steps=mixture.num_steps,
        burn_fraction=mixture.burn_fraction, thin=mixture.thin,
    )
    samples = fit.samples
    projected = np.einsum("td,sdk->tsk", fit.test_basis, samples.w)
    residual_var = np.maximum(
        np.asarray(fit.test_prior_var) - np.sum(np.asarray(fit.test_basis) ** 2, axis=1), 0.0
    )
    sigma = np.sqrt(samples.sigma[None] ** 2 + residual_var[:, None, None])

    # Resample (sample, component) pairs by weight so the mixture is uniform over draws.
    num_samples = samples.w.shape[0]
    weights = np.exp(np.asarray(samples.log_pi)).reshape(-1) / num_samples
    draws = np.random.default_rng(0).choice(
        weights.size, NUM_COMPONENT_DRAWS, p=weights / weights.sum()
    )
    predictions = projected.reshape(len(residual_var), -1)[:, draws]
    sigma_draws = sigma.reshape(len(residual_var), -1)[:, draws]
    _pro_density_bands(
        ax,
        np.asarray(data.x_test[:, 0]),
        predictions.mean(axis=1),
        predictions.std(axis=1) + sigma_draws.mean(axis=1),
        predictions,
        sigma_draws,
        color=OMGP_COLOR,
    )


def main(index: int) -> None:
    key = jr.split(jr.PRNGKey(CONF.seed), CONF.num_instances)[index]
    kwargs = {k: v for k, v in OmegaConf.to_object(DS_CONF).items() if k not in ("source", "param_name")}
    kwargs["num_branches"] = 3
    data = make_multibranch_instance(key, **kwargs)

    gp = fit_gp(data, kernel_lengthscale=CONF.kernel.lengthscale)
    lengthscale, variance = gp_kernel_hyperparameters(gp.kernel)
    kernel = gpx.kernels.RBF(lengthscale=lengthscale, variance=variance)

    x_grid = jnp.linspace(float(data.x_train.min()), float(data.x_train.max()), NUM_GRID)
    plot_data = data._replace(x_test=x_grid[:, None], y_test=jnp.zeros((NUM_GRID, 1)))
    fit_key, omgp_key = jr.split(key)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    titles = ("OMGP, K = 2", "OMGP, K = 3", "PrO-GP (no K)")
    for ax, title, num_components in zip(axes, titles, (2, 3, None), strict=True):
        if num_components is None:
            pro = fit_pro(
                plot_data, fit_key, kernel_lengthscale=lengthscale,
                kernel_variance=variance, **OmegaConf.to_container(CONF.pro),
            )
            _pro_density_bands(
                ax, np.asarray(x_grid), np.asarray(pro.mean), np.asarray(pro.std),
                np.asarray(pro.particle_predictions), np.asarray(pro.sigma_eff),
            )
        else:
            omgp_bands(ax, plot_data, kernel, gp.noise_std, num_components, omgp_key)

        x_all = np.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
        branches = np.concatenate([data.y_branches_train, data.y_branches_test], axis=1)
        order = np.argsort(x_all)
        for branch in branches:
            ax.plot(x_all[order], branch[order], color="black", linestyle="--",
                    linewidth=1.2, alpha=0.7)
        ax.scatter(data.x_train, data.y_train, color=DATA_COLOR, s=10, alpha=0.6, zorder=3)
        _style_box(ax, grid=True, grid_axis="y")
        ax.set_title(title, fontsize=13)
        span = branches.max() - branches.min()
        ax.set_ylim(branches.min() - 0.15 * span, branches.max() + 0.15 * span)
        ax.set_xlabel("x")

    fig.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="--", label="true branches"),
            Line2D([0], [0], marker="o", color=DATA_COLOR, linestyle="None", label="training data"),
            Patch(facecolor=OMGP_COLOR, alpha=PRO_BAND_ALPHAS[1], label="OMGP 50%/95% regions"),
            Patch(facecolor=PRO_COLOR, alpha=PRO_BAND_ALPHAS[1], label="PrO-GP 50%/95% regions"),
        ],
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), frameon=False,
    )
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "multibranch_overlay.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=12)
    main(parser.parse_args().index)
