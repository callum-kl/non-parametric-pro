import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from example_grid import _SOURCES as EXAMPLE_SOURCES
from example_grid import LEGEND_HANDLES as EXAMPLE_LEGEND_HANDLES
from example_grid import plot_examples
from matplotlib.lines import Line2D
from plot_summary import (
    ALGORITHM_COLORS,
    DIVIDER_COLOR,
    SOURCES,
    load_summary,
    plot_summary_row,
)

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

_EXAMPLE_ORDER = tuple(spec.name for spec in EXAMPLE_SOURCES)
if _EXAMPLE_ORDER != SOURCES:
    msg = (
        f"example_grid.py's dataset order {_EXAMPLE_ORDER} doesn't match "
        f"plot_summary.py's {SOURCES} -- did one change without the other?"
    )
    raise ValueError(msg)

SUMMARY_LEGEND_HANDLES = [
    Line2D(
        [0],
        [0],
        marker="o",
        color=ALGORITHM_COLORS["standard_gp"],
        linestyle="-",
        markersize=6,
        label="Standard GP",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color=ALGORITHM_COLORS["pro_gp"],
        linestyle="-",
        markersize=6,
        label="PrO-GP",
    ),
]

TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 14
AXIS_FONTSIZE = 15
EXAMPLE_LEFT, EXAMPLE_RIGHT = 0.02, 0.64
SUMMARY_LEFT, SUMMARY_RIGHT = 0.70, 0.99
GRID_TOP, GRID_BOTTOM = 0.90, 0.16


def _add_dividers(fig, example_axes, summary_axes) -> None:
    fig.canvas.draw()
    ex_pos = [ax.get_position() for ax in example_axes]
    su_pos = [[ax.get_position() for ax in row] for row in summary_axes]

    top = max(*(p.y1 for p in ex_pos), *(p.y1 for row in su_pos for p in row))
    bottom = min(*(p.y0 for p in ex_pos), *(p.y0 for row in su_pos for p in row))

    block_v_x = (
        max(p.x1 for p in ex_pos) + min(p.x0 for row in su_pos for p in row)
    ) / 2
    fig.add_artist(
        Line2D(
            [block_v_x, block_v_x], [bottom, top], color=DIVIDER_COLOR, linewidth=1.2
        )
    )

    su_v_x = (su_pos[0][0].x1 + su_pos[0][1].x0) / 2
    su_h_y = (su_pos[0][0].y0 + su_pos[1][0].y1) / 2
    su_left = min(su_pos[0][0].x0, su_pos[1][0].x0)
    su_right = max(su_pos[0][1].x1, su_pos[1][1].x1)
    su_top = max(su_pos[0][0].y1, su_pos[0][1].y1)
    su_bottom = min(su_pos[1][0].y0, su_pos[1][1].y0)
    fig.add_artist(
        Line2D(
            [su_v_x, su_v_x], [su_bottom, su_top], color=DIVIDER_COLOR, linewidth=1.0
        )
    )
    fig.add_artist(
        Line2D(
            [su_left, su_right], [su_h_y, su_h_y], color=DIVIDER_COLOR, linewidth=1.0
        )
    )


def main(seed: int, num_instances: int) -> None:
    plt.rcParams.update({"font.size": AXIS_FONTSIZE})

    fig = plt.figure(figsize=(24, 9))
    example_gs = fig.add_gridspec(
        2,
        2,
        left=EXAMPLE_LEFT,
        right=EXAMPLE_RIGHT,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
        hspace=0.12,
        wspace=0.06,
    )
    summary_gs = fig.add_gridspec(
        2,
        2,
        left=SUMMARY_LEFT,
        right=SUMMARY_RIGHT,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
        hspace=0.45,
        wspace=0.3,
    )
    example_axes = [
        [fig.add_subplot(example_gs[i, j]) for j in range(2)] for i in range(2)
    ]
    summary_axes = [
        [fig.add_subplot(summary_gs[i, j]) for j in range(2)] for i in range(2)
    ]
    example_flat = [ax for row in example_axes for ax in row]
    summary_flat = [ax for row in summary_axes for ax in row]

    plot_examples(example_flat, seed, num_instances)
    for ax in example_flat:
        ax.set_title(ax.get_title(), fontsize=TITLE_FONTSIZE)

    records = load_summary()
    plot_summary_row(summary_flat, records, SOURCES)
    for ax in summary_flat[1:]:
        ax.sharey(summary_flat[0])
    for row in summary_axes:
        row[0].set_ylabel("NLPD", fontsize=AXIS_FONTSIZE)
    for ax in summary_flat:
        ax.set_title(ax.get_title(), fontsize=TITLE_FONTSIZE)
        ax.tick_params(labelsize=AXIS_FONTSIZE)
        ax.xaxis.label.set_fontsize(AXIS_FONTSIZE)

    fig.legend(
        handles=EXAMPLE_LEGEND_HANDLES,
        loc="lower center",
        bbox_to_anchor=((EXAMPLE_LEFT + EXAMPLE_RIGHT) / 2, 0.0),
        ncol=4,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )
    fig.legend(
        handles=SUMMARY_LEGEND_HANDLES,
        loc="lower center",
        bbox_to_anchor=((SUMMARY_LEFT + SUMMARY_RIGHT) / 2, 0.0),
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )

    _add_dividers(fig, example_flat, summary_axes)

    out_path = FIGURES_DIR / "example_grid_and_summary_columns.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2421)
    parser.add_argument("--num-instances", type=int, default=20)
    args = parser.parse_args()
    main(args.seed, args.num_instances)
