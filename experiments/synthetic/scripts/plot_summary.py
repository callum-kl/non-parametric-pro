import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
SUMMARY_CSV = RESULTS_ROOT / "summary.csv"

SOURCES = ("block_outliers", "heteroskedastic", "multimodal", "well_specified")
SOURCE_TITLES = {
    "block_outliers": "Block outliers",
    "heteroskedastic": "Heteroskedastic",
    "multimodal": "Multimodal",
    "well_specified": "Well-specified",
}

ALGORITHMS = ("pro_gp", "standard_gp")
ALGORITHM_LABELS = {"pro_gp": "PrO-GP", "standard_gp": "Bayes GP"}
ALGORITHM_COLORS = {"pro_gp": "#e8974e", "standard_gp": "#4c3a8e"}
ALGORITHM_MARKERS = {"pro_gp": "^", "standard_gp": "o"}

X_PAD_FRAC = 0.14
# horizontal dodge between the two series, as a fraction of one log2 step
DODGE_FRAC = 0.06

DIVIDER_COLOR = "#cccac0"

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


def load_summary(path: Path = SUMMARY_CSV):
    """{source: {algorithm: [(n, nlpd_mean, nlpd_ci95), ...]}}, each list sorted by `n`."""
    records: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open() as f:
        for row in csv.DictReader(f):
            records[row["source"]][row["algorithm"]].append(
                (
                    float(row["param_value"]),
                    float(row["nlpd_mean"]),
                    float(row["nlpd_ci95"]),
                )
            )

    for by_algorithm in records.values():
        for points in by_algorithm.values():
            points.sort(key=lambda point: point[0])

    return records


def _dodge_offsets(n: int) -> list[float]:
    """Symmetric log2-space offsets so the series' error bars sit side by side."""
    return [(i - (n - 1) / 2) * DODGE_FRAC for i in range(n)]


def _plot_sweep(
    ax, by_algorithm: dict[str, list[tuple[float, float, float]]], all_values, colors
):
    """
    Per-algorithm line+error-bar series across the swept `n` values, log2-scaled
    (`n` doubles each step) so the ticks stay evenly spaced, with the actual tested
    values as ticks.
    """
    offsets = _dodge_offsets(len(ALGORITHMS))
    for algorithm, offset in zip(ALGORITHMS, offsets, strict=True):
        points = by_algorithm.get(algorithm)
        if not points:
            continue
        # dodge in log2 space so the shift is even across the axis
        x = [2 ** (math.log2(point[0]) + offset) for point in points]
        mean = [point[1] for point in points]
        ci95 = [point[2] for point in points]
        ax.errorbar(
            x,
            mean,
            yerr=ci95,
            marker=ALGORITHM_MARKERS[algorithm],
            linestyle="-",
            color=colors[algorithm],
            label=ALGORITHM_LABELS[algorithm],
            markersize=6,
            linewidth=1.5,
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
        )

    ax.set_xlabel("training size n")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(all_values)

    lo, hi = math.log2(min(all_values)), math.log2(max(all_values))
    pad = X_PAD_FRAC * (hi - lo)
    ax.set_xlim(2 ** (lo - pad), 2 ** (hi + pad))


def _plot_source(
    ax, source: str, by_algorithm: dict[str, list[tuple[float, float, float]]], colors
):
    all_values = sorted(
        {point[0] for points in by_algorithm.values() for point in points}
    )
    _plot_sweep(ax, by_algorithm, all_values, colors)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))

    ax.set_title(SOURCE_TITLES[source], fontsize=13)


def plot_summary_row(axes, records, sources=SOURCES, colors=ALGORITHM_COLORS) -> None:
    for ax, source in zip(axes, sources, strict=True):
        _plot_source(ax, source, records.get(source, {}), colors)
