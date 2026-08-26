import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from synthetic import fit_gp, fit_pro

from non_parametric_pro.data.synthetic.block_outliers import (
    make_block_outlier_instance,
    plot_block_outlier_case,
)
from non_parametric_pro.data.synthetic.heteroskedastic import (
    make_heteroskedastic_instance,
    plot_heteroskedastic_case,
)
from non_parametric_pro.data.synthetic.multimodal import (
    make_multimodal_instance,
    plot_multimodal_case,
)
from non_parametric_pro.data.synthetic.well_specified import (
    make_well_specified_instance,
    plot_well_specified_case,
)

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

GP_COLOR = "#e8974e"
PRO_COLOR = "#2ca58d"
GP_LABEL = "Bayes GP"
PRO_LABEL = "PrO-GP"
Z_95 = 1.96
DATA_ALPHA = 0.35

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


class SourceSpec(NamedTuple):
    name: str
    title: str
    make_instance: Callable
    plot_case: Callable
    kwargs: dict
    plot_kwargs: dict
    instance_index: int
    curve_kwarg: str = "show_curve"


_SOURCES = [
    SourceSpec(
        "block_outliers",
        "Block outliers",
        make_block_outlier_instance,
        plot_block_outlier_case,
        {
            "num_regions": 1,
            "outlier_offset_frac": 1.5,
            "min_width": 0.15,
            "max_width": 0.3,
            "ell_range": (0.5, 1.0),
            "noise_std_frac": 0.15,
        },
        {
            "show_train": False,
            "color_by_outlier": True,
            "outlier_subsample_frac": 0.3,
        },
        instance_index=16,
    ),
    SourceSpec(
        "heteroskedastic",
        "Heteroskedastic",
        make_heteroskedastic_instance,
        plot_heteroskedastic_case,
        {
            "num_regions": 2,
            "min_width": 0.3,
            "max_width": 0.8,
            "ell_range": (0.5, 1.0),
            "noise_std_frac": 0.35,
            "amplitude_frac": 1.5,
            "n": 400,
        },
        {"show_noise_bands": False, "show_train": False},
        instance_index=0,
    ),
    SourceSpec(
        "multimodal",
        "Multimodal",
        make_multimodal_instance,
        plot_multimodal_case,
        {
            "num_regions": 1,
            "mix_prob": 0.5,
            "min_width": 0.3,
            "max_width": 0.8,
            "ell_range": (0.5, 1.0),
            "noise_std_frac": 0.1,
            "n": 400,
        },
        {"color_by_branch": False, "show_train": False},
        instance_index=3,
        curve_kwarg="show_curves",
    ),
    SourceSpec(
        "well_specified",
        "Well-specified",
        make_well_specified_instance,
        plot_well_specified_case,
        {"train_fraction": 0.7, "noise_std_frac": 0.2},
        {"show_train": False},
        instance_index=2,
    ),
]


def _style_box(ax, *, grid: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.9)
    ax.grid(grid)
    ax.tick_params(labelsize=9, length=3)


def _hide_ticks(ax, *, x: bool = False, y: bool = False) -> None:
    if x:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if y:
        ax.tick_params(axis="y", left=False, labelleft=False)


def _gp_band(ax, x_sorted, mean_sorted, std_sorted) -> None:
    ax.fill_between(
        x_sorted,
        mean_sorted - Z_95 * std_sorted,
        mean_sorted + Z_95 * std_sorted,
        color=GP_COLOR,
        alpha=0.3,
        linewidth=0,
    )
    ax.plot(x_sorted, mean_sorted, color=GP_COLOR, linewidth=1.6)


def _pro_density_bands(
    ax,
    x_sorted,
    mean_sorted,
    std_sorted,
    particle_predictions_sorted,
    sigma_eff_sorted,
    *,
    credible_k: float = 5.0,
    num_y: int = 150,
    outer_frac: float = 0.05,
    inner_frac: float = 0.3,
) -> None:
    """
    Shade the actual predictive density surface (a per-x mixture of Gaussians, one per
    particle) rather than a per-x quantile envelope -- a quantile-based fill_between
    always draws one contiguous band and so can't show genuine multimodality (e.g. two
    separate branches with a low-density gap between them); contourf over the real 2D
    density can, since each level set is free to split into disconnected regions.
    """
    y_lo = float(np.min(mean_sorted - credible_k * std_sorted))
    y_hi = float(np.max(mean_sorted + credible_k * std_sorted))
    y_grid = np.linspace(y_lo, y_hi, num_y)

    z = (y_grid[None, :, None] - particle_predictions_sorted[:, None, :]) / (
        sigma_eff_sorted[:, None, None]
    )
    normal_pdf = np.exp(-0.5 * z**2) / (sigma_eff_sorted[:, None, None] * np.sqrt(2 * np.pi))
    density = normal_pdf.mean(axis=2)  # (N_x, num_y)

    peak = density.max()
    x_grid, y_mesh = np.meshgrid(x_sorted, y_grid, indexing="ij")
    for frac, alpha in ((outer_frac, 0.25), (inner_frac, 0.45)):
        ax.contourf(
            x_grid,
            y_mesh,
            density,
            levels=[frac * peak, np.inf],
            colors=[PRO_COLOR],
            alpha=alpha,
        )


def _plot_fit_panel(ax, spec: SourceSpec, instance_key, *, method: str, algorithm: str) -> None:
    """Draw one dataset's single-method fit panel (truth + data + GP or PrO band) into `ax`."""
    data = spec.make_instance(instance_key, **spec.kwargs)
    plot_kwargs = {**spec.plot_kwargs, spec.curve_kwarg: True}
    spec.plot_case(ax, data, **plot_kwargs)
    for line in ax.lines:
        line.set_color("black")
        line.set_linestyle("--")
        line.set_linewidth(1.3)
        line.set_alpha(0.7)
    if ax.collections:
        ax.collections[0].set_alpha(DATA_ALPHA)
    kernel_type = getattr(data, "kernel_type", "rbf")

    order = np.argsort(data.x_test[:, 0])
    x_sorted = np.asarray(data.x_test[order, 0])
    gp_key, fit_key = jr.split(instance_key)

    if method == "gp":
        result = fit_gp(data, gp_key, kernel_type=kernel_type)
        _gp_band(
            ax,
            x_sorted,
            np.asarray(result.mean)[order],
            np.asarray(result.std)[order],
        )
    else:
        result = fit_pro(data, fit_key, kernel_type=kernel_type, algorithm=algorithm)
        _pro_density_bands(
            ax,
            x_sorted,
            np.asarray(result.mean)[order],
            np.asarray(result.std)[order],
            np.asarray(result.particle_predictions)[order],
            np.asarray(result.sigma_eff)[order],
        )

    _style_box(ax)


def plot_fit_grid(
    fig,
    gp_axes,
    pro_axes,
    seed: int,
    num_instances: int,
    index_overrides: dict[str, int] | None = None,
    *,
    algorithm: str = "replica_gibbs",
    sources: list[SourceSpec] = _SOURCES,
) -> None:
    """Fill `gp_axes`/`pro_axes` (each a flat list matching `sources`, one axis per
    dataset) with the Bayes-GP row and PrO-GP row of a reference-style fit grid."""
    index_overrides = index_overrides or {}
    keys = jr.split(jr.PRNGKey(seed), num_instances)
    for col, (ax, spec) in enumerate(zip(gp_axes, sources, strict=True)):
        index = index_overrides.get(spec.name, spec.instance_index)
        _plot_fit_panel(ax, spec, keys[index], method="gp", algorithm=algorithm)
        ax.set_title(GP_LABEL, fontsize=16, color=GP_COLOR)
        pos = ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.075,
            spec.title,
            ha="center",
            fontsize=16,
            color="black",
        )
        _hide_ticks(ax, x=True, y=col > 0)
    for col, (ax, spec) in enumerate(zip(pro_axes, sources, strict=True)):
        index = index_overrides.get(spec.name, spec.instance_index)
        _plot_fit_panel(ax, spec, keys[index], method="pro", algorithm=algorithm)
        ax.set_title(PRO_LABEL, fontsize=16, color=PRO_COLOR)
        _hide_ticks(ax, y=col > 0)


TOP_LEGEND_HANDLES = [
    Line2D(
        [0], [0], color="black", linestyle="--", linewidth=1.3, alpha=0.7, label="truth"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        linestyle="None",
        markersize=6,
        label="data",
    ),
    Patch(facecolor=GP_COLOR, alpha=0.3, label=f"{GP_LABEL} (95% central)"),
    Patch(facecolor=PRO_COLOR, alpha=0.4, label=f"{PRO_LABEL} (50%/95% regions)"),
    Line2D(
        [0],
        [0],
        marker="x",
        color="maroon",
        linestyle="None",
        markersize=8,
        markeredgewidth=1.5,
        label="outliers",
    ),
]


def main(
    seed: int,
    num_instances: int,
    index_overrides: dict[str, int],
    *,
    algorithm: str = "replica_gibbs",
) -> None:
    n = len(_SOURCES)
    fig = plt.figure(figsize=(6.5 * n, 8.5))
    gs = fig.add_gridspec(2, n, top=0.78, bottom=0.08, hspace=0.4, wspace=0.08)
    gp_axes = [fig.add_subplot(gs[0, j]) for j in range(n)]
    pro_axes = [fig.add_subplot(gs[1, j]) for j in range(n)]

    plot_fit_grid(
        fig, gp_axes, pro_axes, seed, num_instances, index_overrides, algorithm=algorithm
    )

    fig.legend(
        handles=TOP_LEGEND_HANDLES,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        fontsize=13,
        frameon=False,
    )

    out_path = FIGURES_DIR / "example_grid.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2421)
    parser.add_argument("--num-instances", type=int, default=20)
    parser.add_argument(
        "--algorithm",
        default="replica_gibbs",
        choices=["ula", "replica_gibbs"],
        help="PRO sampler to use for the PrO-GP fits (default: replica_gibbs).",
    )
    for _spec in _SOURCES:
        parser.add_argument(
            f"--{_spec.name.replace('_', '-')}-index",
            type=int,
            default=None,
            help=f"Override this panel's instance index (default from _SOURCES: {_spec.instance_index}).",
        )
    args = parser.parse_args()

    index_overrides = {
        spec.name: getattr(args, f"{spec.name}_index")
        for spec in _SOURCES
        if getattr(args, f"{spec.name}_index") is not None
    }
    main(args.seed, args.num_instances, index_overrides, algorithm=args.algorithm)
