from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

RESULTS_ROOT = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
SUMMARY_CSV = RESULTS_ROOT / "summary.csv"

SOURCES = ("block_outliers", "heteroskedastic", "multimodal", "well_specified")
SOURCE_TITLES = {
    "multimodal": "Multimodal",
    "heteroskedastic": "Heteroskedastic",
    "block_outliers": "Block outliers",
    "well_specified": "Well-specified",
}

ALGORITHMS = ("pro_gp", "standard_gp")
ALGORITHM_LABELS = {"pro_gp": "PrO-GP", "standard_gp": "Standard GP"}
ALGORITHM_COLORS = {"pro_gp": "#3a76c4", "standard_gp": "#3f8f5f"}

N_MIN = 100
DODGE_FRAC = 0.03

DIVIDER_COLOR = "#cccac0"

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


def load_summary(path: Path = SUMMARY_CSV):
    """{source: {algorithm: [(n, nlpd_mean, nlpd_ci95), ...]}}, each algorithm's list
    sorted by `n`. Only rows whose own `param_name` is `"n"` are kept -- `summary.csv`
    also holds each dataset's own misspecification-severity sweep (`amplitude_frac`,
    `outlier_offset_frac`, `noise_skewness`, ...), which isn't what this plot shows."""
    records: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["param_name"] != "n":
                continue
            n = float(row["param_value"])
            if n < N_MIN:
                continue
            records[row["source"]][row["algorithm"]].append(
                (n, float(row["nlpd_mean"]), float(row["nlpd_ci95"]))
            )

    for by_algorithm in records.values():
        for points in by_algorithm.values():
            points.sort(key=lambda point: point[0])

    return records


def _plot_sweep(ax, by_algorithm: dict[str, list[tuple[float, float, float]]], all_values):
    """Per-algorithm line+error-bar series across the swept `n` values -- dodged apart
    so nearby points' error bars stay legible, log2-scaled (`n` doubles each step) so
    the ticks stay evenly spaced, with the actual tested values as ticks."""
    for algorithm in ALGORITHMS:
        points = by_algorithm.get(algorithm)
        if not points:
            continue
        dodge = 1 - DODGE_FRAC if algorithm == "pro_gp" else 1 + DODGE_FRAC
        x = [point[0] * dodge for point in points]
        mean = [point[1] for point in points]
        ci95 = [point[2] for point in points]
        ax.errorbar(
            x, mean, yerr=ci95, fmt="o-",
            color=ALGORITHM_COLORS[algorithm], label=ALGORITHM_LABELS[algorithm],
            markersize=6, linewidth=1.5, elinewidth=1.5, capsize=4, capthick=1.5,
        )

    ax.set_xlabel("dataset size")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(all_values)


def _plot_categorical(ax, by_algorithm: dict[str, list[tuple[float, float, float]]]):
    xs = list(range(len(ALGORITHMS)))
    for x, algorithm in zip(xs, ALGORITHMS, strict=True):
        (_, mean, ci95), = by_algorithm[algorithm]
        ax.errorbar(
            [x], [mean], yerr=[ci95], fmt="o",
            color=ALGORITHM_COLORS[algorithm], label=ALGORITHM_LABELS[algorithm],
            markersize=6, elinewidth=1.5, capsize=4, capthick=1.5,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)


def _plot_source(ax, source: str, by_algorithm: dict[str, list[tuple[float, float, float]]]):
    all_values = sorted({point[0] for points in by_algorithm.values() for point in points})
    is_sweep = len(all_values) > 1

    if is_sweep:
        _plot_sweep(ax, by_algorithm, all_values)
    else:
        _plot_categorical(ax, by_algorithm)

    ax.yaxis.set_major_locator(mticker.MultipleLocator(base=0.5))

    ax.set_title(SOURCE_TITLES[source], fontsize=13)


def _add_grid_dividers(fig, axes) -> None:
    fig.canvas.draw()
    pos = [[ax.get_position() for ax in row] for row in axes]

    v_x = (pos[0][0].x1 + pos[0][1].x0) / 2
    h_y = (pos[1][0].y1 + pos[0][0].y0) / 2
    left = min(pos[0][0].x0, pos[1][0].x0)
    right = max(pos[0][1].x1, pos[1][1].x1)
    bottom = min(pos[1][0].y0, pos[1][1].y0)
    top = max(pos[0][0].y1, pos[0][1].y1)

    fig.add_artist(Line2D([v_x, v_x], [bottom, top], color=DIVIDER_COLOR, linewidth=1.0))
    fig.add_artist(Line2D([left, right], [h_y, h_y], color=DIVIDER_COLOR, linewidth=1.0))


def plot_summary_row(axes, records, sources=SOURCES) -> None:
    for ax, source in zip(axes, sources, strict=True):
        _plot_source(ax, source, records.get(source, {}))


def plot_summary(records, sources=SOURCES, filename="summary_panel.png"):
    fig, axes = plt.subplots(
        2, 2, figsize=(9, 8.5), sharey=True, gridspec_kw={"hspace": 0.45, "wspace": 0.12}
    )

    plot_summary_row(axes.flat, records, sources)

    for row in axes:
        row[0].set_ylabel("NLPD")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=False)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _add_grid_dividers(fig, axes)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    plot_summary(load_summary())
