import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from example_grid import _SOURCES as EXAMPLE_SOURCES
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_summary import (
    DIVIDER_COLOR,
    SOURCES,
    load_summary,
    plot_summary_row,
)
from synthetic import fit_gp, fit_pro

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

_EXAMPLE_ORDER = tuple(spec.name for spec in EXAMPLE_SOURCES)
if _EXAMPLE_ORDER != SOURCES:
    msg = (
        f"example_grid.py's dataset order {_EXAMPLE_ORDER} doesn't match "
        f"plot_summary.py's {SOURCES} -- did one change without the other?"
    )
    raise ValueError(msg)

GP_COLOR = "#e8974e"
PRO_COLOR = "#2ca58d"
SUMMARY_COLORS = {"standard_gp": GP_COLOR, "pro_gp": PRO_COLOR}

GP_LABEL = "Bayes GP"
PRO_LABEL = "PrO-GP"
NUM_QUANTILE_DRAWS = 200
Z_90 = 1.6448536269514722
DATA_ALPHA = 0.35

_CURVE_KWARG_NAME = {"multimodal": "show_curves"}

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
    Patch(facecolor=GP_COLOR, alpha=0.3, label=f"{GP_LABEL} (90% central)"),
    Patch(facecolor=PRO_COLOR, alpha=0.4, label=f"{PRO_LABEL} (50%/90% regions)"),
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

TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 13
AXIS_FONTSIZE = 13
FIT_LEFT, FIT_RIGHT = 0.03, 0.68
FIT_NCOLS = 4
FIT_COL_WIDTH = (FIT_RIGHT - FIT_LEFT) / FIT_NCOLS
SUMMARY_LEFT = FIT_RIGHT + 0.035
SUMMARY_RIGHT = SUMMARY_LEFT + FIT_COL_WIDTH
GRID_TOP, GRID_BOTTOM = 0.80, 0.08


def _style_box(ax, *, grid: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.9)
    ax.grid(grid)
    ax.tick_params(labelsize=AXIS_FONTSIZE - 3, length=3)


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
        alpha=0.3,
        linewidth=0,
    )
    ax.plot(x_sorted, mean_sorted, color=GP_COLOR, linewidth=1.6)


def _pro_bands(ax, x_sorted, draws_sorted) -> None:
    lo90, hi90 = np.quantile(draws_sorted, [0.05, 0.95], axis=1)
    lo50, hi50 = np.quantile(draws_sorted, [0.25, 0.75], axis=1)
    ax.fill_between(x_sorted, lo90, hi90, color=PRO_COLOR, alpha=0.25, linewidth=0)
    ax.fill_between(x_sorted, lo50, hi50, color=PRO_COLOR, alpha=0.45, linewidth=0)


def _plot_fit_panel(ax, spec, instance_key, *, method: str) -> None:
    data = spec.make_instance(instance_key, **spec.kwargs)
    curve_kwarg = _CURVE_KWARG_NAME.get(spec.name, "show_curve")
    plot_kwargs = {**spec.plot_kwargs, curve_kwarg: True}
    spec.plot_case(ax, data, **plot_kwargs)
    for line in ax.lines:
        line.set_color("black")
        line.set_linestyle("--")
        line.set_linewidth(1.3)
        line.set_alpha(0.7)
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
        result = fit_pro(
            data, fit_key, kernel_type=kernel_type, num_function_draws=NUM_QUANTILE_DRAWS
        )
        _pro_bands(ax, x_sorted, np.asarray(result.function_draws)[order])

    _style_box(ax)


def plot_fit_grid(
    fig,
    gp_axes,
    pro_axes,
    seed: int,
    num_instances: int,
    index_overrides: dict[str, int] | None = None,
) -> None:
    index_overrides = index_overrides or {}
    keys = jr.split(jr.PRNGKey(seed), num_instances)
    for col, (ax, spec) in enumerate(zip(gp_axes, EXAMPLE_SOURCES, strict=True)):
        index = index_overrides.get(spec.name, spec.instance_index)
        _plot_fit_panel(ax, spec, keys[index], method="gp")
        ax.set_title(GP_LABEL, fontsize=TITLE_FONTSIZE, color=GP_COLOR)
        pos = ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.075,
            spec.title,
            ha="center",
            fontsize=TITLE_FONTSIZE,
            color="black",
        )
        _hide_ticks(ax, x=True, y=col > 0)
    for col, (ax, spec) in enumerate(zip(pro_axes, EXAMPLE_SOURCES, strict=True)):
        index = index_overrides.get(spec.name, spec.instance_index)
        _plot_fit_panel(ax, spec, keys[index], method="pro")
        ax.set_title(PRO_LABEL, fontsize=TITLE_FONTSIZE, color=PRO_COLOR)
        _hide_ticks(ax, y=col > 0)


def _add_dividers(fig, fit_axes, summary_axes) -> None:
    fig.canvas.draw()
    fit_pos = [ax.get_position() for row in fit_axes for ax in row]
    su_pos = [ax.get_position() for ax in summary_axes]

    top = max(*(p.y1 for p in fit_pos), *(p.y1 for p in su_pos))
    bottom = min(*(p.y0 for p in fit_pos), *(p.y0 for p in su_pos))
    v_x = (max(p.x1 for p in fit_pos) + min(p.x0 for p in su_pos)) / 2
    fig.add_artist(Line2D([v_x, v_x], [bottom, top], color=DIVIDER_COLOR, linewidth=1.2))


def main(seed: int, num_instances: int, index_overrides: dict[str, int]) -> None:
    plt.rcParams.update({"font.size": AXIS_FONTSIZE})

    fig = plt.figure(figsize=(26, 8.5))
    fit_gs = fig.add_gridspec(
        2,
        4,
        left=FIT_LEFT,
        right=FIT_RIGHT,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
        hspace=0.4,
        wspace=0.08,
    )
    summary_gs = fig.add_gridspec(
        4,
        1,
        left=SUMMARY_LEFT,
        right=SUMMARY_RIGHT,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
        hspace=0.7,
    )
    gp_axes = [fig.add_subplot(fit_gs[0, j]) for j in range(4)]
    pro_axes = [fig.add_subplot(fit_gs[1, j]) for j in range(4)]
    summary_axes = [fig.add_subplot(summary_gs[i, 0]) for i in range(4)]

    plot_fit_grid(fig, gp_axes, pro_axes, seed, num_instances, index_overrides)

    records = load_summary()
    plot_summary_row(summary_axes, records, SOURCES, colors=SUMMARY_COLORS)
    for ax in summary_axes[:-1]:
        ax.set_xlabel("")
        _hide_ticks(ax, x=True)
    for ax in summary_axes:
        ax.set_title(ax.get_title(), fontsize=TITLE_FONTSIZE)
        ax.xaxis.label.set_fontsize(AXIS_FONTSIZE)
        _style_box(ax, grid=True)

    fig.text(
        (SUMMARY_LEFT + SUMMARY_RIGHT) / 2,
        GRID_TOP + 0.075,
        "Held-out NLPD (lower is better)",
        ha="center",
        fontsize=TITLE_FONTSIZE,
    )

    fig.legend(
        handles=TOP_LEGEND_HANDLES,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )

    _add_dividers(fig, [gp_axes, pro_axes], summary_axes)

    out_path = FIGURES_DIR / "example_grid_and_summary_columns.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2421)
    parser.add_argument("--num-instances", type=int, default=20)
    for _spec in EXAMPLE_SOURCES:
        parser.add_argument(
            f"--{_spec.name.replace('_', '-')}-index",
            type=int,
            default=None,
            help=f"Override this panel's instance index (default from _SOURCES: {_spec.instance_index}).",
        )
    args = parser.parse_args()

    index_overrides = {
        spec.name: getattr(args, f"{spec.name}_index")
        for spec in EXAMPLE_SOURCES
        if getattr(args, f"{spec.name}_index") is not None
    }
    main(args.seed, args.num_instances, index_overrides)
