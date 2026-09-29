"""Whitened-residual normality panels of the fitted exact GP, one row per dataset."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy import stats
from scipy.linalg import solve_triangular
from util import load_gp_state

from non_parametric_pro.data.uci.uci import load_uci_regression_dataset

UCI_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = UCI_DIR / "figures"

COLUMN_TITLES = ("Q-Q vs N(0, 1)", "Density vs N(0, 1)", "Residual vs index")

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd6"
SERIES = "#2a78d6"
REF = "#8a8984"


def whitened_residuals(dataset: str, split: int, jitter: float = 1e-8) -> np.ndarray:
    """z = L^-1 y with L = chol(K + sigma^2 I): exactly i.i.d. N(0, 1) under the model."""
    cfg = OmegaConf.create(
        {"results_root": str(UCI_DIR / "results"), "dataset": dataset, "split": split, "inducing": False}
    )
    kernel, sigma, _, scaler_x, scaler_y = load_gp_state(cfg)
    example = load_uci_regression_dataset(dataset, split=split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train).squeeze(-1)

    k_ff = np.asarray(kernel.gram(x_train).as_matrix(), dtype=float)
    k_tilde = k_ff + (sigma**2 + jitter) * np.eye(k_ff.shape[0])
    return solve_triangular(np.linalg.cholesky(k_tilde), y_train, lower=True)


def _blom(n: int) -> np.ndarray:
    return (np.arange(1, n + 1) - 0.375) / (n + 0.25)


def _qq_envelope(n: int, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise band for N(0,1) order statistics via their Beta distribution."""
    i = np.arange(1, n + 1)
    lo_p = (1.0 - level) / 2.0
    lo = stats.norm.ppf(stats.beta.ppf(lo_p, i, n - i + 1))
    hi = stats.norm.ppf(stats.beta.ppf(1.0 - lo_p, i, n - i + 1))
    return lo, hi


TICK_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 15
PANEL_TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 13


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=TICK_FONTSIZE, length=4)


def _qq_points(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(z)
    theoretical = stats.norm.ppf(_blom(ordered.size))
    return theoretical, ordered


def draw_qq(ax, z: np.ndarray, colour: str) -> None:
    theoretical, ordered = _qq_points(z)
    lo, hi = _qq_envelope(ordered.size)
    ax.fill_between(
        theoretical,
        lo,
        hi,
        color=REF,
        alpha=0.15,
        linewidth=0,
        label="95% pointwise band",
    )
    ax.plot(theoretical, theoretical, color=REF, linewidth=2, zorder=2, label="N(0, 1)")
    ax.scatter(
        theoretical,
        ordered,
        s=14,
        color=colour,
        edgecolor=SURFACE,
        linewidth=0.5,
        zorder=3,
    )
    _style_axes(ax)


def draw_histogram(ax, z: np.ndarray, colour: str) -> None:
    ax.hist(
        z,
        bins=min(40, max(12, int(np.sqrt(z.size) * 1.5))),
        density=True,
        color=colour,
        alpha=0.55,
        edgecolor=SURFACE,
        linewidth=0.5,
    )
    grid = np.linspace(min(-4.0, z.min()), max(4.0, z.max()), 400)
    ax.plot(grid, stats.norm.pdf(grid), color=REF, linewidth=2, label="N(0, 1)")
    _style_axes(ax)


def draw_residual_scatter(
    ax, z: np.ndarray, covariate: np.ndarray, colour: str
) -> None:
    ax.axhline(0.0, color=REF, linewidth=2, zorder=2)
    for level in (-2.0, 2.0):
        ax.axhline(level, color=REF, linewidth=1, linestyle="--", alpha=0.7, zorder=2)
    ax.scatter(
        covariate,
        z,
        s=14,
        color=colour,
        edgecolor=SURFACE,
        linewidth=0.5,
        zorder=3,
    )
    _style_axes(ax)


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
    colour = SERIES
    fig = plt.figure(figsize=(16.5, 4.6 * len(datasets)), facecolor=SURFACE)
    subfigs = fig.subfigures(len(datasets), 1, hspace=0.10)
    if len(datasets) == 1:
        subfigs = [subfigs]

    qq_axes, hist_axes, scatter_axes = [], [], []
    for row, (subfig, dataset) in enumerate(zip(subfigs, datasets, strict=True)):
        subfig.set_facecolor(SURFACE)
        z = whitened_residuals(dataset, split)
        axes = subfig.subplots(1, 3)

        draw_qq(axes[0], z, colour)
        axes[0].set_xlabel("Theoretical quantile", fontsize=AXIS_LABEL_FONTSIZE, color=INK_MUTED)
        axes[0].set_ylabel("Observed quantile", fontsize=AXIS_LABEL_FONTSIZE, color=INK_MUTED)
        if row == 0:
            axes[0].legend(
                frameon=False, fontsize=LEGEND_FONTSIZE, labelcolor=INK_MUTED, loc="upper left"
            )

        draw_histogram(axes[1], z, colour)
        axes[1].set_xlabel("z", fontsize=AXIS_LABEL_FONTSIZE, color=INK_MUTED)
        axes[1].set_ylabel("Density", fontsize=AXIS_LABEL_FONTSIZE, color=INK_MUTED)

        draw_residual_scatter(axes[2], z, np.arange(z.size, dtype=float), colour)
        axes[2].set_xlabel("Training index", fontsize=AXIS_LABEL_FONTSIZE, color=INK_MUTED)
        axes[2].set_ylabel("z", fontsize=AXIS_LABEL_FONTSIZE, color=INK_MUTED)

        for ax, title in zip(axes, COLUMN_TITLES, strict=True):
            ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE, color=INK_MUTED)

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
    parser.add_argument("--datasets", default="machine,stock,concreteslump")
    parser.add_argument("--split", type=int, default=1)
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / f"normality_panel_{'_'.join(datasets)}_split{args.split}.png"
    plot_combined(datasets, args.split, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
