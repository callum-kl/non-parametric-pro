import os
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

from non_parametric_pro.data.synthetic.illustrative import (
    make_illustrative_instance,
    plot_illustrative_case,
    plotted_test_points,
)

GP_COLOR = "#e8974e"
PRO_COLOR = "#2ca58d"
GP_LABEL = "Bayes GP"
PRO_LABEL = "PrO-GP"
Z_90 = 1.96
DATA_ALPHA = 0.45
PANEL_ALPHA = 1.4
PANEL_NUM_PARTICLES = 50
# From the tuning study in ../tuning: kernel adaptation is robust to its initial
# lengthscale only at a large learning rate, with kernel updates spaced a handful of
# sampler iterations apart -- both a tiny lr and a near-every-step schedule fail.
PANEL_ADAPT_STEPS = 400
PANEL_KERNEL_ADAPT_STEPS = 400
PANEL_KERNEL_LR = 0.5
PANEL_KERNEL_STEPS_PER_ADAPT = 1
GP_BAND_ALPHA = 0.22
PRO_BAND_ALPHAS = (0.18, 0.36)
COLUMN_TITLE_PAD_IN = 0.62
X_TICKS = (0.0, 0.5, 1.0)
# Every panel is rescaled onto this window for display (see `_display_affine`), then
# padded slightly so nothing sits on the spines.
DISPLAY_YLIM = (-1.5, 1.0)
DISPLAY_PAD = 0.06

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
    regime: str
    title: str
    instance_index: int


_SOURCES = [
    SourceSpec("block_outliers", "Block outliers", 16),
    SourceSpec("heteroskedastic", "Heteroskedastic", 1),
    SourceSpec("multimodal", "Multimodal", 10),
    SourceSpec("well_specified", "Well-specified", 1),
]


def _style_box(ax, *, grid: bool = False, grid_axis: str = "both") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.9)
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid_axis)
    ax.tick_params(labelsize=9, length=3)


def _hide_ticks(ax, *, x: bool = False, y: bool = False) -> None:
    if x:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if y:
        ax.tick_params(axis="y", left=False, labelleft=False)


def _gp_band(ax, x_sorted, mean_sorted, std_sorted) -> None:
    ax.fill_between(
        x_sorted,
        mean_sorted - Z_90 * std_sorted,
        mean_sorted + Z_90 * std_sorted,
        color=GP_COLOR,
        alpha=GP_BAND_ALPHA,
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
    num_y: int = 100,
    # widest first, so the inner region sits on top
    masses: tuple[float, ...] = (0.95, 0.5),
    alphas: tuple[float, ...] = PRO_BAND_ALPHAS,
) -> None:
    """
    Shade the actual predictive density surface (a per-x mixture of Gaussians, one per
    particle) rather than a per-x quantile envelope -- a quantile-based fill_between
    always draws one contiguous band and so can't show genuine multimodality (e.g. two
    separate branches with a low-density gap between them); contouring over the real 2D
    density can, since each level set is free to split into disconnected regions.

    The field contoured is the *enclosed mass*: `enclosed[i, j]` is the mass of the
    smallest highest-density region at `x_i` that contains `y_j`, so `enclosed <= m`
    is exactly that slice's 100m% HPD region.
    """
    y_lo = float(np.min(mean_sorted - credible_k * std_sorted))
    y_hi = float(np.max(mean_sorted + credible_k * std_sorted))
    y_grid = np.linspace(y_lo, y_hi, num_y)

    z = (y_grid[None, :, None] - particle_predictions_sorted[:, None, :]) / (
        sigma_eff_sorted[:, None, None]
    )
    normal_pdf = np.exp(-0.5 * z**2) / (
        sigma_eff_sorted[:, None, None] * np.sqrt(2 * np.pi)
    )
    density = normal_pdf.mean(axis=2)  # (N_x, num_y)

    order = np.argsort(-density, axis=1)
    cumulative = np.cumsum(np.take_along_axis(density, order, axis=1), axis=1)
    cumulative /= cumulative[:, -1:]  # normalise away the grid's truncated tails
    enclosed = np.empty_like(cumulative)
    np.put_along_axis(enclosed, order, cumulative, axis=1)

    x_grid, y_mesh = np.meshgrid(x_sorted, y_grid, indexing="ij")
    for mass, alpha in zip(masses, alphas, strict=True):
        ax.contourf(
            x_grid,
            y_mesh,
            enclosed,
            levels=[0.0, mass],
            colors=[PRO_COLOR],
            alpha=alpha,
        )


def _display_affine(data) -> tuple[float, float]:
    """`(scale, shift)` mapping this panel's own data onto `DISPLAY_YLIM`.

    Every regime is generated and fitted at the magnitudes it is specified with --
    the block-outlier shift is a genuine +2.2, the heteroskedastic noise really
    reaches 0.48 -- which puts the panels on wildly different y ranges. Rescaling for
    display only, after fitting, lets all eight share one frame without touching what
    the models actually saw. Applied uniformly to data, truth and predictive bands, so
    each panel stays internally consistent.
    """
    # the same points the panel draws: held-out data, truth curves, outlier markers
    is_outlier = np.asarray(data.is_outlier_train)
    values = np.concatenate(
        [
            np.asarray(plotted_test_points(data)[1]).ravel(),
            np.asarray(data.y_curves).ravel(),
            np.asarray(data.y_train).ravel()[is_outlier],
        ]
    )
    lo, hi = float(values.min()), float(values.max())
    display_lo, display_hi = DISPLAY_YLIM
    scale = (display_hi - display_lo) / (hi - lo)
    return scale, display_lo - scale * lo


def _plot_fit_panel(
    ax, spec: SourceSpec, instance_key, *, method: str, algorithm: str
) -> None:
    """Draw one dataset's single-method fit panel (truth + data + GP or PrO band) into `ax`."""
    data = make_illustrative_instance(instance_key, regime=spec.regime)
    scale, shift = _display_affine(data)
    plot_illustrative_case(ax, data, data_alpha=DATA_ALPHA, scale=scale, shift=shift)

    order = np.argsort(data.x_test[:, 0])
    x_sorted = np.asarray(data.x_test[order, 0])
    gp_key, fit_key = jr.split(instance_key)

    if method == "gp":
        result = fit_gp(data, gp_key)
        _gp_band(
            ax,
            x_sorted,
            scale * np.asarray(result.mean)[order] + shift,
            scale * np.asarray(result.std)[order],
        )
    else:
        result = fit_pro(
            data,
            fit_key,
            algorithm=algorithm,
            alpha=PANEL_ALPHA,
            num_particles=PANEL_NUM_PARTICLES,
            num_adapt_steps=PANEL_ADAPT_STEPS,
            kernel_adapt_steps=PANEL_KERNEL_ADAPT_STEPS,
            kernel_lr=PANEL_KERNEL_LR,
            kernel_steps_per_adapt=PANEL_KERNEL_STEPS_PER_ADAPT,
            warmup_steps=4,
            kernel_lengthscale=0.4,
            val_fraction=0.3,
        )
        _pro_density_bands(
            ax,
            x_sorted,
            scale * np.asarray(result.mean)[order] + shift,
            scale * np.asarray(result.std)[order],
            scale * np.asarray(result.particle_predictions)[order] + shift,
            scale * np.asarray(result.sigma_eff)[order],
        )

    _style_box(ax, grid=True, grid_axis="y")


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
        index = index_overrides.get(spec.regime, spec.instance_index)
        _plot_fit_panel(ax, spec, keys[index], method="gp", algorithm=algorithm)
        ax.set_title(GP_LABEL, fontsize=16, color=GP_COLOR)
        pos = ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + COLUMN_TITLE_PAD_IN / fig.get_figheight(),
            spec.title,
            ha="center",
            fontsize=16,
            color="black",
        )
        _hide_ticks(ax, x=True, y=col > 0)
    for col, (ax, spec) in enumerate(zip(pro_axes, sources, strict=True)):
        index = index_overrides.get(spec.regime, spec.instance_index)
        _plot_fit_panel(ax, spec, keys[index], method="pro", algorithm=algorithm)
        ax.set_title(PRO_LABEL, fontsize=16, color=PRO_COLOR)
        ax.set_xlabel("input x")
        _hide_ticks(ax, y=col > 0)

    display_lo, display_hi = DISPLAY_YLIM
    pad = DISPLAY_PAD * (display_hi - display_lo)
    for ax in (*gp_axes, *pro_axes):
        ax.set_ylim(display_lo - pad, display_hi + pad)
        ax.set_xticks(X_TICKS)
    for ax in (gp_axes[0], pro_axes[0]):
        ax.set_ylabel("response y")


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
    Patch(facecolor=GP_COLOR, alpha=GP_BAND_ALPHA, label=f"{GP_LABEL} (90% central)"),
    Patch(
        facecolor=PRO_COLOR,
        alpha=PRO_BAND_ALPHAS[1],
        label=f"{PRO_LABEL} (50%/90% regions)",
    ),
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
