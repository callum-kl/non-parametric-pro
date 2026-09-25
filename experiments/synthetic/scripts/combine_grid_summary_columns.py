import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from example_grid import _SOURCES as EXAMPLE_SOURCES
from example_grid import (
    COLUMN_TITLE_PAD_IN,
    GP_COLOR,
    LEGEND_HANDLES,
    PRO_COLOR,
    _hide_ticks,
    _style_box,
    plot_fit_grid,
)
from matplotlib.lines import Line2D
from plot_summary import (
    DIVIDER_COLOR,
    SOURCES,
    load_summary,
    plot_summary_row,
)

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

_EXAMPLE_ORDER = tuple(spec.regime for spec in EXAMPLE_SOURCES)
if _EXAMPLE_ORDER != SOURCES:
    msg = (
        f"example_grid.py's dataset order {_EXAMPLE_ORDER} doesn't match "
        f"plot_summary.py's {SOURCES} -- did one change without the other?"
    )
    raise ValueError(msg)

SUMMARY_COLORS = {"standard_gp": GP_COLOR, "pro_gp": PRO_COLOR}

TITLE_FONTSIZE = 19
LEGEND_FONTSIZE = 19
AXIS_FONTSIZE = 16

# Layout is specified in inches and converted to figure fractions below, so panel
# proportions are exact regardless of how many columns the figure ends up with.
FIT_PANEL_W_IN = 3.6
FIT_PANEL_H_IN = 2.8  # slightly wider than tall
FIT_COL_GAP_IN = 0.62  # room for each column's own y tick labels
FIT_ROW_GAP_IN = 0.55  # room for the "PrO-GP" row title
SUMMARY_NCOLS = 1
BLOCK_GAP_IN = 0.95  # between the fit block and the summary column
MARGIN_LEFT_IN, MARGIN_RIGHT_IN = 0.62, 0.25
MARGIN_TOP_IN, MARGIN_BOTTOM_IN = 0.95, 1.05

FIT_NCOLS, FIT_NROWS = 4, 2
SUMMARY_NROWS = 4
# The NLPD column is as wide as one fit column; its panels are short, their height
# falling out of sharing the fit block's total height four ways.
SUMMARY_PANEL_W_IN = FIT_PANEL_W_IN

FIT_BLOCK_W_IN = FIT_NCOLS * FIT_PANEL_W_IN + (FIT_NCOLS - 1) * FIT_COL_GAP_IN
GRID_H_IN = FIT_NROWS * FIT_PANEL_H_IN + (FIT_NROWS - 1) * FIT_ROW_GAP_IN
SUMMARY_BLOCK_W_IN = (
    SUMMARY_NCOLS * SUMMARY_PANEL_W_IN + (SUMMARY_NCOLS - 1) * FIT_COL_GAP_IN
)
SUMMARY_ROW_GAP_IN = 0.42  # room for each NLPD panel's own title
SUMMARY_PANEL_H_IN = (
    GRID_H_IN - (SUMMARY_NROWS - 1) * SUMMARY_ROW_GAP_IN
) / SUMMARY_NROWS

FIG_W = (
    MARGIN_LEFT_IN
    + FIT_BLOCK_W_IN
    + BLOCK_GAP_IN
    + SUMMARY_BLOCK_W_IN
    + MARGIN_RIGHT_IN
)
FIG_H = MARGIN_TOP_IN + GRID_H_IN + MARGIN_BOTTOM_IN

FIT_LEFT = MARGIN_LEFT_IN / FIG_W
FIT_RIGHT = (MARGIN_LEFT_IN + FIT_BLOCK_W_IN) / FIG_W
SUMMARY_LEFT = (MARGIN_LEFT_IN + FIT_BLOCK_W_IN + BLOCK_GAP_IN) / FIG_W
SUMMARY_RIGHT = SUMMARY_LEFT + SUMMARY_BLOCK_W_IN / FIG_W
GRID_TOP = 1.0 - MARGIN_TOP_IN / FIG_H
GRID_BOTTOM = MARGIN_BOTTOM_IN / FIG_H

# gridspec spacing is a fraction of the average panel size
FIT_WSPACE = FIT_COL_GAP_IN / FIT_PANEL_W_IN
FIT_HSPACE = FIT_ROW_GAP_IN / FIT_PANEL_H_IN
SUMMARY_HSPACE = SUMMARY_ROW_GAP_IN / SUMMARY_PANEL_H_IN


def _add_dividers(fig, fit_axes, summary_axes) -> None:
    fig.canvas.draw()
    fit_pos = [ax.get_position() for row in fit_axes for ax in row]
    su_pos = [ax.get_position() for ax in summary_axes]

    top = max(*(p.y1 for p in fit_pos), *(p.y1 for p in su_pos))
    bottom = min(*(p.y0 for p in fit_pos), *(p.y0 for p in su_pos))
    v_x = (max(p.x1 for p in fit_pos) + min(p.x0 for p in su_pos)) / 2
    fig.add_artist(
        Line2D([v_x, v_x], [bottom, top], color=DIVIDER_COLOR, linewidth=1.2)
    )


def main(
    seed: int,
    num_instances: int,
    index_overrides: dict[str, int],
    *,
    algorithm: str = "replica_gibbs",
) -> None:
    plt.rcParams.update({"font.size": AXIS_FONTSIZE})

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fit_gs = fig.add_gridspec(
        FIT_NROWS,
        FIT_NCOLS,
        left=FIT_LEFT,
        right=FIT_RIGHT,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
        hspace=FIT_HSPACE,
        wspace=FIT_WSPACE,
    )
    summary_gs = fig.add_gridspec(
        SUMMARY_NROWS,
        SUMMARY_NCOLS,
        left=SUMMARY_LEFT,
        right=SUMMARY_RIGHT,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
        hspace=SUMMARY_HSPACE,
    )
    gp_axes = [fig.add_subplot(fit_gs[0, j]) for j in range(4)]
    pro_axes = [fig.add_subplot(fit_gs[1, j]) for j in range(4)]
    summary_axes = [
        fig.add_subplot(summary_gs[i, j])
        for i in range(SUMMARY_NROWS)
        for j in range(SUMMARY_NCOLS)
    ]

    plot_fit_grid(
        fig,
        gp_axes,
        pro_axes,
        seed,
        num_instances,
        index_overrides,
        algorithm=algorithm,
    )

    records = load_summary()
    plot_summary_row(summary_axes, records, SOURCES, colors=SUMMARY_COLORS)
    for ax in summary_axes[:-SUMMARY_NCOLS]:
        ax.set_xlabel("")
        _hide_ticks(ax, x=True)
    for ax in summary_axes:
        ax.set_title(ax.get_title(), fontsize=TITLE_FONTSIZE)
        ax.xaxis.label.set_fontsize(AXIS_FONTSIZE)
        _style_box(ax, grid=True, grid_axis="y")

    fig.text(
        (SUMMARY_LEFT + SUMMARY_RIGHT) / 2,
        GRID_TOP + COLUMN_TITLE_PAD_IN / FIG_H,
        "Held-out NLPD",
        ha="center",
        fontsize=TITLE_FONTSIZE,
    )

    fig.legend(
        handles=LEGEND_HANDLES,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
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
    parser.add_argument(
        "--algorithm",
        default="replica_gibbs",
        choices=["ula", "replica_gibbs"],
        help="PRO sampler to use for the PrO-GP fits (default: replica_gibbs).",
    )
    for _spec in EXAMPLE_SOURCES:
        parser.add_argument(
            f"--{_spec.regime.replace('_', '-')}-index",
            type=int,
            default=None,
            help=f"Override this panel's instance index (default from _SOURCES: {_spec.instance_index}).",
        )
    args = parser.parse_args()

    index_overrides = {
        spec.regime: getattr(args, f"{spec.regime}_index")
        for spec in EXAMPLE_SOURCES
        if getattr(args, f"{spec.regime}_index") is not None
    }
    main(args.seed, args.num_instances, index_overrides, algorithm=args.algorithm)
