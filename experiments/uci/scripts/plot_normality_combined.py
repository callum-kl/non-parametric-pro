"""Combine the whitened-residual normality panels for several datasets into one figure."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gp_normality import (
    INK,
    INK_MUTED,
    SERIES,
    SURFACE,
    draw_histogram,
    draw_qq,
    draw_residual_scatter,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR.parent / "results"

COLUMN_TITLES = ("Q-Q vs N(0, 1)", "Density vs N(0, 1)", "Residual vs index")


def residuals_path(dataset: str, split: int) -> Path:
    return (
        RESULTS_ROOT / dataset / f"split_{split}" / "exact_gp" / "normality"
    ) / "residuals.npz"


def _union(axes, getter):
    lo = min(getter(ax)[0] for ax in axes)
    hi = max(getter(ax)[1] for ax in axes)
    return lo, hi


def _share_limits(qq_axes, hist_axes, scatter_axes) -> None:
    """One common z range across the Q-Q y, density x and residual y axes."""
    z_lo, z_hi = _union([*qq_axes, *scatter_axes], lambda ax: ax.get_ylim())
    h_lo, h_hi = _union(hist_axes, lambda ax: ax.get_xlim())
    z_lo, z_hi = min(z_lo, h_lo), max(z_hi, h_hi)

    q_lim = _union(qq_axes, lambda ax: ax.get_xlim())
    d_top = max(ax.get_ylim()[1] for ax in hist_axes)

    for ax in qq_axes:
        ax.set_xlim(*q_lim)
        ax.set_ylim(z_lo, z_hi)
    for ax in hist_axes:
        ax.set_xlim(z_lo, z_hi)
        ax.set_ylim(0.0, d_top)
    for ax in scatter_axes:
        ax.set_ylim(z_lo, z_hi)


def plot_combined(datasets: list[str], split: int, path: Path) -> None:
    colour = SERIES["whitened"]
    fig = plt.figure(figsize=(16.5, 4.0 * len(datasets)), facecolor=SURFACE)
    subfigs = fig.subfigures(len(datasets), 1, hspace=0.10)
    if len(datasets) == 1:
        subfigs = [subfigs]

    qq_axes, hist_axes, scatter_axes = [], [], []
    for row, (subfig, dataset) in enumerate(zip(subfigs, datasets, strict=True)):
        subfig.set_facecolor(SURFACE)
        z = np.load(residuals_path(dataset, split))["z_whitened"]
        axes = subfig.subplots(1, 3)

        draw_qq(axes[0], z, colour)
        axes[0].set_xlabel("Theoretical quantile", fontsize=9, color=INK_MUTED)
        axes[0].set_ylabel("Observed quantile", fontsize=9, color=INK_MUTED)
        if row == 0:
            axes[0].legend(
                frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left"
            )

        draw_histogram(axes[1], z, colour)
        axes[1].set_xlabel("z", fontsize=9, color=INK_MUTED)
        axes[1].set_ylabel("Density", fontsize=9, color=INK_MUTED)

        draw_residual_scatter(axes[2], z, np.arange(z.size, dtype=float), colour)
        axes[2].set_xlabel("Training index", fontsize=9, color=INK_MUTED)
        axes[2].set_ylabel("z", fontsize=9, color=INK_MUTED)

        for ax, title in zip(axes, COLUMN_TITLES, strict=True):
            ax.set_title(title, fontsize=10, color=INK_MUTED)

        qq_axes.append(axes[0])
        hist_axes.append(axes[1])
        scatter_axes.append(axes[2])

        subfig.subplots_adjust(top=0.80)
        subfig.suptitle(dataset, fontsize=16, color=INK, y=0.99)

    _share_limits(qq_axes, hist_axes, scatter_axes)

    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="machine,stock")
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    missing = [d for d in datasets if not residuals_path(d, args.split).exists()]
    if missing:
        raise SystemExit(
            f"No residuals for {missing} at split {args.split} -- run gp_normality.py first."
        )

    name = f"normality_panel_{'_'.join(datasets)}_split{args.split}.png"
    out = Path(args.out) if args.out else RESULTS_ROOT / name
    plot_combined(datasets, args.split, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
