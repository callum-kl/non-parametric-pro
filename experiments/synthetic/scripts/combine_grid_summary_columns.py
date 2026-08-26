import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from example_grid import _SOURCES as EXAMPLE_SOURCES
from example_grid import (
    GP_COLOR,
    PRO_COLOR,
    TOP_LEGEND_HANDLES,
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

_EXAMPLE_ORDER = tuple(spec.name for spec in EXAMPLE_SOURCES)
if _EXAMPLE_ORDER != SOURCES:
    msg = (
        f"example_grid.py's dataset order {_EXAMPLE_ORDER} doesn't match "
        f"plot_summary.py's {SOURCES} -- did one change without the other?"
    )
    raise ValueError(msg)

SUMMARY_COLORS = {"standard_gp": GP_COLOR, "pro_gp": PRO_COLOR}

TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 13
AXIS_FONTSIZE = 13
FIT_LEFT = 0.03
RIGHT_MARGIN = 0.02
GAP = 0.035
FIT_NCOLS = 4
# Fit columns and the summary column share one width, filling [FIT_LEFT, 1 - RIGHT_MARGIN]
# with `GAP` between the fit block and the summary block.
FIT_COL_WIDTH = (1.0 - RIGHT_MARGIN - FIT_LEFT - GAP) / (FIT_NCOLS + 1)
FIT_RIGHT = FIT_LEFT + FIT_NCOLS * FIT_COL_WIDTH
SUMMARY_LEFT = FIT_RIGHT + GAP
SUMMARY_RIGHT = SUMMARY_LEFT + FIT_COL_WIDTH
GRID_TOP, GRID_BOTTOM = 0.80, 0.08


def _add_dividers(fig, fit_axes, summary_axes) -> None:
    fig.canvas.draw()
    fit_pos = [ax.get_position() for row in fit_axes for ax in row]
    su_pos = [ax.get_position() for ax in summary_axes]

    top = max(*(p.y1 for p in fit_pos), *(p.y1 for p in su_pos))
    bottom = min(*(p.y0 for p in fit_pos), *(p.y0 for p in su_pos))
    v_x = (max(p.x1 for p in fit_pos) + min(p.x0 for p in su_pos)) / 2
    fig.add_artist(Line2D([v_x, v_x], [bottom, top], color=DIVIDER_COLOR, linewidth=1.2))


def main(
    seed: int,
    num_instances: int,
    index_overrides: dict[str, int],
    *,
    algorithm: str = "replica_gibbs",
) -> None:
    plt.rcParams.update({"font.size": AXIS_FONTSIZE})

    fig = plt.figure(figsize=(26, 8.5))
    fit_gs = fig.add_gridspec(
        2,
        4,
        left=FIT_LEFT,
        right=FIT_RIGHT,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
        hspace=0.15,
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

    plot_fit_grid(
        fig, gp_axes, pro_axes, seed, num_instances, index_overrides, algorithm=algorithm
    )

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
    parser.add_argument(
        "--algorithm",
        default="replica_gibbs",
        choices=["ula", "replica_gibbs"],
        help="PRO sampler to use for the PrO-GP fits (default: replica_gibbs).",
    )
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
    main(args.seed, args.num_instances, index_overrides, algorithm=args.algorithm)
