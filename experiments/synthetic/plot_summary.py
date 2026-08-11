"""Plot `pro_gp` vs `standard_gp` mean NLPD +/- 95% CI, one subplot per dataset, from
`results/summary.csv` (written by `aggregate_results.py`'s `save_csv`).

A 2x2 error-bar panel -- same grid shape and dataset order (multimodal,
heteroskedastic, block_outliers, well_specified) as `example_grid.py`'s `_SOURCES` --
with one subplot per source. `well_specified` (`n`) and `block_outliers`
(`outlier_offset_frac`) each sweep a real parameter with more than one `param_value` in
`summary.csv`, so each algorithm gets its own dodged, line-connected error bar across
the tested values (`SWEEP_XLABELS`/`SWEEP_LOG2_SOURCES` control each one's x-axis label
and whether it's log2-scaled -- `n` doubles each step so log2 spaces it evenly,
`outlier_offset_frac` doesn't so it stays linear). `heteroskedastic`/`multimodal`
currently only have one `param_value` each, so there's no sweep to put on the x-axis --
instead, `pro_gp`/`standard_gp` themselves are the two x positions, each with its own
error bar and no connecting line (a line only makes sense along a real swept axis, which
two bare categories aren't). Their x-tick text is hidden (redundant with the shared
legend) -- only a genuine sweep's x-axis keeps tick labels. A subplot automatically
falls back to the categorical view whenever it has only one `param_value` per algorithm,
and to the sweep view otherwise -- so a future multi-point sweep for e.g.
`heteroskedastic` would render like the other two without any further code changes
beyond adding it to `SWEEP_XLABELS`.

Colors/style match `example_grid.py`'s convention (GP green / PRO blue, no spines, faint
grid, a shared bottom legend) so figures share one visual language. Thin divider lines
between the four panels (`_add_grid_dividers`) substitute for the spines this style
otherwise removes, since without them the four panels bleed into each other.

`nlpd_ci95` in the CSV is already a 95% CI half-width (1.96 * SEM, see
`aggregate_results.py`'s `_nlpd_ci95`), so it's used directly as `yerr`.
"""

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

# Same order as example_grid.py's `_SOURCES`, laid out row-major over a 2x2 grid (i.e.
# `axes.flat`) so a given dataset lands in the same cell in both figures.
SOURCES = ("multimodal", "heteroskedastic", "block_outliers", "well_specified")
SOURCE_TITLES = {
    # Exact strings from example_grid.py's SourceSpec.title, for consistency.
    "multimodal": "Multimodal",
    "heteroskedastic": "Heteroskedastic",
    "block_outliers": "Block outliers",
    "well_specified": "Well-specified",
}

# x-axis label for sources with a real sweep (more than one `param_value`); a source
# with only one `param_value` doesn't need an entry (it renders via the categorical
# view instead -- see `_plot_categorical`).
SWEEP_XLABELS = {
    "well_specified": "dataset size",
    "block_outliers": "outlier offset (x signal std)",
}
# `n` doubles each step, so log2 spacing keeps the ticks evenly spaced;
# `outlier_offset_frac`'s steps aren't multiplicative, so it stays linear.
SWEEP_LOG2_SOURCES = {"well_specified"}

ALGORITHMS = ("pro_gp", "standard_gp")
ALGORITHM_LABELS = {"pro_gp": "PrO-GP", "standard_gp": "Standard GP"}
# Same GP green / PRO blue as example_grid.py's GP_COLOR/PRO_COLOR -- kept as literal
# duplicates rather than an import, since example_grid.py pulls in the whole fitting
# stack (synthetic.fit_gp/fit_pro) that this CSV-only script has no other need for.
ALGORITHM_COLORS = {"pro_gp": "#3a76c4", "standard_gp": "#3f8f5f"}

WELL_SPECIFIED_N_MIN = 50  # drop the n=50 row -- "n increasing from 100 to 800"
DODGE_FRAC = 0.03  # multiplicative x-offset between the two algorithms' error bars

DIVIDER_COLOR = "#cccac0"

# Paper-figure formatting, matching example_grid.py: no spines, a faint grid for scale
# reference instead. Unlike example_grid.py's qualitative density overlays, this plot is
# meant to be read quantitatively, so (unlike there) ticks/tick-labels stay on.
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
    """{source: {algorithm: [(param_value, nlpd_mean, nlpd_ci95), ...]}}, each
    algorithm's list sorted by `param_value`."""
    records: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open() as f:
        for row in csv.DictReader(f):
            source = row["source"]
            param_value = float(row["param_value"])
            if source == "well_specified" and param_value < WELL_SPECIFIED_N_MIN:
                continue
            records[source][row["algorithm"]].append(
                (param_value, float(row["nlpd_mean"]), float(row["nlpd_ci95"]))
            )

    for by_algorithm in records.values():
        for points in by_algorithm.values():
            points.sort(key=lambda point: point[0])

    return records


def _plot_sweep(
    ax, by_algorithm: dict[str, list[tuple[float, float, float]]], all_values, *, xlabel, use_log2,
):
    """Per-algorithm line+error-bar series across a real swept `param_value` -- dodged
    apart so nearby points' error bars stay legible, with the actual swept values as
    ticks. `use_log2` only makes sense for a multiplicatively-spaced sweep (e.g. `n`
    doubling); a linearly-spaced one (e.g. an offset fraction) stays on a linear axis."""
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

    ax.set_xlabel(xlabel)
    ax.set_xticks(all_values)
    if use_log2:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _plot_categorical(ax, by_algorithm: dict[str, list[tuple[float, float, float]]]):
    """`pro_gp` vs `standard_gp` as the two x positions (one tested `param_value`, so
    there's no sweep to put on the x-axis instead) -- no connecting line (nothing sits
    "between" two bare categories), and no x-tick text (redundant with the shared
    legend, which is what actually identifies each color here)."""
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
        _plot_sweep(
            ax, by_algorithm, all_values,
            xlabel=SWEEP_XLABELS.get(source, "param value"),
            use_log2=source in SWEEP_LOG2_SOURCES,
        )
    else:
        _plot_categorical(ax, by_algorithm)

    ax.set_title(SOURCE_TITLES[source], fontsize=13)


def _add_grid_dividers(fig, axes) -> None:
    """Thin lines through the midpoints between the 2x2 grid's rows/columns -- the
    "no spines" style otherwise leaves nothing separating the four panels."""
    fig.canvas.draw()  # finalise the tight_layout()-adjusted positions before reading them
    pos = [[ax.get_position() for ax in row] for row in axes]

    v_x = (pos[0][0].x1 + pos[0][1].x0) / 2
    h_y = (pos[1][0].y1 + pos[0][0].y0) / 2
    left = min(pos[0][0].x0, pos[1][0].x0)
    right = max(pos[0][1].x1, pos[1][1].x1)
    bottom = min(pos[1][0].y0, pos[1][1].y0)
    top = max(pos[0][0].y1, pos[0][1].y1)

    fig.add_artist(Line2D([v_x, v_x], [bottom, top], color=DIVIDER_COLOR, linewidth=1.0))
    fig.add_artist(Line2D([left, right], [h_y, h_y], color=DIVIDER_COLOR, linewidth=1.0))


def plot_summary(records, sources=SOURCES, filename="summary_panel.png"):
    fig, axes = plt.subplots(
        2, 2, figsize=(9, 8.5), sharey=True, gridspec_kw={"hspace": 0.45, "wspace": 0.12}
    )

    for ax, source in zip(axes.flat, sources, strict=True):
        _plot_source(ax, source, records.get(source, {}))

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
