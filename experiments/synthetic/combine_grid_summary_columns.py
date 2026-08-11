"""A genuine 2x4 grid, one column per dataset: `example_grid.py`'s qualitative fit
overlay on top, `plot_summary.py`'s quantitative NLPD error bar directly underneath --
built from real matplotlib subplots (via `example_grid.plot_examples`/
`plot_summary.plot_summary_row`), not a composite of the two saved PNGs, so each
dataset's column lines up exactly and there's one shared legend instead of two.

    python experiments/synthetic/combine_grid_summary_columns.py
    python experiments/synthetic/combine_grid_summary_columns.py --seed 7 --num-instances 30
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from example_grid import LEGEND_HANDLES as EXAMPLE_LEGEND_HANDLES
from example_grid import _SOURCES as EXAMPLE_SOURCES
from example_grid import plot_examples
from plot_summary import ALGORITHM_COLORS, DIVIDER_COLOR, SOURCES, load_summary, plot_summary_row

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# example_grid.py's `_SOURCES` and plot_summary.py's `SOURCES` are independently
# maintained lists that happen to need the same order for this script to make sense
# (each column must be the same dataset in both rows) -- fail loudly rather than
# silently mismatching columns if one changes without the other.
_EXAMPLE_ORDER = tuple(spec.name for spec in EXAMPLE_SOURCES)
if _EXAMPLE_ORDER != SOURCES:
    msg = (
        f"example_grid.py's dataset order {_EXAMPLE_ORDER} doesn't match "
        f"plot_summary.py's {SOURCES} -- columns would line up wrong."
    )
    raise ValueError(msg)

# The summary row's own error-bar markers already use these colors (see
# plot_summary.py's ALGORITHM_COLORS), so this just gives the shared legend a second
# pair of proxies alongside example_grid's density-band Patches -- readers connect the
# two purely by color, which is already consistent across both rows.
SUMMARY_LEGEND_HANDLES = [
    Line2D(
        [0], [0], marker="o", color=ALGORITHM_COLORS["standard_gp"], linestyle="-",
        markersize=6, label="Standard GP (NLPD)",
    ),
    Line2D(
        [0], [0], marker="o", color=ALGORITHM_COLORS["pro_gp"], linestyle="-",
        markersize=6, label="PrO-GP (NLPD)",
    ),
]


def _add_dividers(fig, axes) -> None:
    """One horizontal divider between the example-grid row and the summary row, and
    three vertical dividers between the four dataset columns (spanning both rows) --
    the "no spines" style otherwise leaves nothing separating the eight panels."""
    fig.canvas.draw()  # finalise the tight_layout()-adjusted positions before reading them
    pos = [[ax.get_position() for ax in row] for row in axes]

    left = min(p.x0 for row in pos for p in row)
    right = max(p.x1 for row in pos for p in row)
    top = max(p.y1 for row in pos for p in row)
    bottom = min(p.y0 for row in pos for p in row)

    h_y = (pos[0][0].y0 + pos[1][0].y1) / 2
    fig.add_artist(Line2D([left, right], [h_y, h_y], color=DIVIDER_COLOR, linewidth=1.0))

    for col in range(len(pos[0]) - 1):
        v_x = (pos[0][col].x1 + pos[0][col + 1].x0) / 2
        fig.add_artist(Line2D([v_x, v_x], [bottom, top], color=DIVIDER_COLOR, linewidth=1.0))


TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 14
AXIS_FONTSIZE = 15


def main(seed: int, num_instances: int) -> None:
    # Overrides example_grid.py's/plot_summary.py's own rcParams (already applied at
    # import time) for tick/axis-label text specifically -- larger than either
    # standalone figure needs, since this one gets read at a smaller size once
    # embedded in a wider layout.
    plt.rcParams.update({"font.size": AXIS_FONTSIZE})

    fig, axes = plt.subplots(2, 4, figsize=(24, 9), sharey="row", gridspec_kw={"hspace": 0.5, "wspace": 0.18})

    plot_examples(axes[0], seed, num_instances)
    for ax in axes[0]:
        # `plot_example_panel` hardcodes its own title fontsize; bump it here rather
        # than threading a fontsize param through example_grid.py just for this.
        ax.set_title(ax.get_title(), fontsize=TITLE_FONTSIZE)

    records = load_summary()
    plot_summary_row(axes[1], records, SOURCES)
    axes[1][0].set_ylabel("NLPD", fontsize=AXIS_FONTSIZE)
    for ax in axes[1]:
        # Each column's dataset name is already the top row's title; repeating it
        # under the bottom row too would just be noise.
        ax.set_title("")
        ax.tick_params(labelsize=AXIS_FONTSIZE)
        ax.xaxis.label.set_fontsize(AXIS_FONTSIZE)

    fig.legend(
        handles=[*EXAMPLE_LEGEND_HANDLES, *SUMMARY_LEGEND_HANDLES],
        loc="lower center", ncol=6, fontsize=LEGEND_FONTSIZE, frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _add_dividers(fig, axes)

    out_path = FIGURES_DIR / "example_grid_and_summary_columns.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2421)
    parser.add_argument("--num-instances", type=int, default=20)
    args = parser.parse_args()
    main(args.seed, args.num_instances)
