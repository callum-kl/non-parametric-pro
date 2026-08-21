import argparse
import os
from pathlib import Path
from typing import Callable, NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from non_parametric_pro.data.synthetic.block_outliers import (
    make_block_outlier_instance,
    plot_block_outlier_case,
)
from non_parametric_pro.data.synthetic.heteroskedastic import (
    make_heteroskedastic_instance,
    plot_heteroskedastic_case,
)
from non_parametric_pro.data.synthetic.multimodal import make_multimodal_instance, plot_multimodal_case
from non_parametric_pro.data.synthetic.well_specified import (
    make_well_specified_instance,
    plot_well_specified_case,
)

from synthetic import FitResult, fit_gp, fit_pro

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

GP_COLOR = "#3f8f5f"
PRO_COLOR = "#3a76c4"
GP_CMAP = LinearSegmentedColormap.from_list("gp_density", ["#e8f4ec", GP_COLOR])
PRO_CMAP = LinearSegmentedColormap.from_list("pro_density", ["#e6eef8", PRO_COLOR])
PRO_ALPHA = 0.7

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

class SourceSpec(NamedTuple):
    name: str
    title: str
    make_instance: Callable
    plot_case: Callable
    kwargs: dict
    plot_kwargs: dict
    instance_index: int


_SOURCES = [
    SourceSpec(
        "block_outliers",
        "Block outliers",
        make_block_outlier_instance,
        plot_block_outlier_case,
        {
            "num_regions": 1, "outlier_offset_frac": 1.5,
            "min_width": 0.15, "max_width": 0.3, "ell_range": (0.5, 1.0), "noise_std_frac": 0.15,
        },
        {"show_curve": False, "show_train": False, "color_by_outlier": True, "outlier_subsample_frac": 0.3},
        instance_index=3,
    ),
    SourceSpec(
        "heteroskedastic",
        "Heteroskedastic",
        make_heteroskedastic_instance,
        plot_heteroskedastic_case,
        {"num_regions": 2, "min_width": 0.3, "max_width": 0.8, "ell_range": (0.5, 1.0), "noise_std_frac": 0.15},
        {"show_curve": False, "show_noise_bands": False, "show_train": False},
        instance_index=4,
    ),
    SourceSpec(
        "multimodal",
        "Multimodal",
        make_multimodal_instance,
        plot_multimodal_case,
        {
            "num_regions": 1, "mix_prob": 0.5, "min_width": 0.3, "max_width": 0.8,
            "ell_range": (0.5, 1.0), "noise_std_frac": 0.1, "n": 300,
        },
        {"show_curves": False, "color_by_branch": False, "show_train": False},
        instance_index=3,
    ),
    SourceSpec(
        "well_specified",
        "Well-specified",
        make_well_specified_instance,
        plot_well_specified_case,
        {"train_fraction": 0.7, 'noise_std_frac': 0.2},
        {"show_curve": False, "show_train": False},
        instance_index=2,
    ),
]


_DENSITY_DEFAULTS = {"num_bands": 3, "credible_k": 5.0, "num_y": 150, "min_density_frac": 0.03}

def _masked_density_bands(  # noqa: PLR0913
    ax, x_sorted, mean_sorted, std_sorted, density_fn, *,
    cmap, num_bands, credible_k, num_y, min_density_frac, alpha=1.0, edge_color=None,
) -> None:
    y_lo = float(jnp.min(mean_sorted - credible_k * std_sorted))
    y_hi = float(jnp.max(mean_sorted + credible_k * std_sorted))
    y_grid = jnp.linspace(y_lo, y_hi, num_y)

    density = density_fn(y_grid)
    in_band = jnp.abs(y_grid[None, :] - mean_sorted[:, None]) <= credible_k * std_sorted[:, None]
    density = np.asarray(jnp.where(in_band, density, jnp.nan))

    peak = np.nanmax(density)
    levels = np.linspace(min_density_frac, 1.0, num_bands) * peak

    x_grid, y_mesh = np.meshgrid(np.asarray(x_sorted), np.asarray(y_grid), indexing="ij")
    ax.contourf(x_grid, y_mesh, density, levels=levels, cmap=cmap, alpha=alpha, extend="neither")
    if edge_color is not None:
        ax.contour(x_grid, y_mesh, density, levels=levels, colors=edge_color, linewidths=0.6, alpha=alpha)
    ax.plot(x_sorted, mean_sorted, color=edge_color or cmap(1.0), linewidth=1.4, alpha=alpha)


def _overlay_gp_density(ax, x_test, result: FitResult, *, cmap, alpha=1.0, **density_kwargs) -> None:
    order = jnp.argsort(x_test[:, 0])
    x_sorted = x_test[order, 0]
    mean_sorted, std_sorted = result.mean[order], result.std[order]

    def density_fn(y_grid):
        z = (y_grid[None, :] - mean_sorted[:, None]) / std_sorted[:, None]
        return jnp.exp(-0.5 * z**2) / (std_sorted[:, None] * jnp.sqrt(2 * jnp.pi))

    _masked_density_bands(
        ax, x_sorted, mean_sorted, std_sorted, density_fn, cmap=cmap, alpha=alpha, edge_color=GP_COLOR,
        **{**_DENSITY_DEFAULTS, **density_kwargs},
    )


def _overlay_pro_density(ax, x_test, result: FitResult, *, cmap, alpha=1.0, **density_kwargs) -> None:
    order = jnp.argsort(x_test[:, 0])
    x_sorted = x_test[order, 0]
    mean_sorted = result.mean[order]
    std_sorted = result.std[order]
    particle_predictions_sorted = result.particle_predictions[order]  # (N_x, J)
    sigma_eff_sorted = result.sigma_eff[order]  # (N_x,)

    def density_fn(y_grid):
        z = (y_grid[None, :, None] - particle_predictions_sorted[:, None, :]) / sigma_eff_sorted[:, None, None]
        normal_pdf = jnp.exp(-0.5 * z**2) / (sigma_eff_sorted[:, None, None] * jnp.sqrt(2 * jnp.pi))
        return jnp.mean(normal_pdf, axis=2)

    _masked_density_bands(
        ax, x_sorted, mean_sorted, std_sorted, density_fn, cmap=cmap, alpha=alpha, edge_color=PRO_COLOR,
        **{**_DENSITY_DEFAULTS, **density_kwargs},
    )


def plot_example_panel(ax, spec: SourceSpec, instance_key) -> None:
    """Draw one dataset's fit-overlay panel (data + GP/PRO density bands) into `ax`."""
    data = spec.make_instance(instance_key, **spec.kwargs)
    spec.plot_case(ax, data, **spec.plot_kwargs)
    kernel_type = getattr(data, "kernel_type", "rbf")

    gp_result = fit_gp(data, kernel_type=kernel_type)
    _overlay_gp_density(ax, data.x_test, gp_result, cmap=GP_CMAP)

    fit_key, _ = jr.split(instance_key)
    pro_result = fit_pro(data, fit_key, kernel_type=kernel_type)
    _overlay_pro_density(ax, data.x_test, pro_result, cmap=PRO_CMAP, alpha=PRO_ALPHA)

    ax.set_title(spec.title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_examples(
    axes, seed: int, num_instances: int, index_overrides: dict[str, int] | None = None,
) -> None:
    index_overrides = index_overrides or {}
    keys = jr.split(jr.PRNGKey(seed), num_instances)
    for ax, spec in zip(axes, _SOURCES, strict=True):
        index = index_overrides.get(spec.name, spec.instance_index)
        plot_example_panel(ax, spec, keys[index])


LEGEND_HANDLES = [
    Patch(color=GP_COLOR, label="Standard GP"),
    Patch(color=PRO_COLOR, label="PrO-GP"),
    Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=6, label="Test data"),
    Line2D(
        [0], [0], marker="x", color="maroon", linestyle="None",
        markersize=8, markeredgewidth=1.5, label="Outliers",
    ),
]


def main(seed: int, num_instances: int, index_overrides: dict[str, int]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_examples(axes.flat, seed, num_instances, index_overrides)

    fig.legend(handles=LEGEND_HANDLES, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = FIGURES_DIR / "example_grid.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2421)
    parser.add_argument("--num-instances", type=int, default=20)
    for _spec in _SOURCES:
        parser.add_argument(
            f"--{_spec.name.replace('_', '-')}-index", type=int, default=None,
            help=f"Override this panel's instance index (default from _SOURCES: {_spec.instance_index}).",
        )
    args = parser.parse_args()

    index_overrides = {
        spec.name: getattr(args, f"{spec.name}_index")
        for spec in _SOURCES
        if getattr(args, f"{spec.name}_index") is not None
    }
    main(args.seed, args.num_instances, index_overrides)
