"""
Figures and tables comparing PrO-GP with the OMGP baselines:
K-sensitivity on the four regimes (E1), the multi-branch heatmap (E2) and
particles vs components (E3).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from plot_summary import ALGORITHM_COLORS, SOURCE_TITLES, SOURCES

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

PRO_COLOR = ALGORITHM_COLORS["pro_gp"]
GP_COLOR = ALGORITHM_COLORS["standard_gp"]
OMGP_COLOR = "#3b6fb6"

E1_KS = (1, 2, 3, 5, 10)
E3_KS = (1, 2, 3, 5, 10, 20, 50)
E3_NS = (5, 10, 20, 50, 100)
E3_N = 100
E3_SOURCES = ("multimodal", "well_specified")

PREFIX = "omgp"


def load(source: str, param_dir: str) -> dict[str, dict]:
    root = RESULTS_ROOT / source / param_dir
    if not root.exists():
        return {}
    return {
        path.parent.name: json.loads(path.read_text())
        for path in root.glob("*/metrics.json")
    }


def mean_se(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(se)


def paired(metrics: dict, algorithm: str, reference: str = "pro_gp"):
    """Mean ± SE of per-instance NLPD(algorithm) - NLPD(reference), and win-rate of reference."""
    diff = np.array(metrics[algorithm]["nlpds"]) - np.array(metrics[reference]["nlpds"])
    return (*mean_se(diff), float(np.mean(diff > 0)))


def _hline(ax, metrics, algorithm, color, label, linestyle):
    if algorithm not in metrics:
        return
    mean, se = mean_se(metrics[algorithm]["nlpds"])
    ax.axhline(mean, color=color, linestyle=linestyle, linewidth=1.5, label=label)
    ax.axhspan(mean - se, mean + se, color=color, alpha=0.12, linewidth=0)


def _omgp_curve(ax, metrics, ks, label="OMGP (K components)"):
    points = [
        (k, *mean_se(metrics[f"{PREFIX}_k{k}"]["nlpds"]))
        for k in ks
        if f"{PREFIX}_k{k}" in metrics
    ]
    if not points:
        return
    k, mean, se = zip(*points, strict=True)
    ax.errorbar(
        k, mean, yerr=se, color=OMGP_COLOR, marker="o", markersize=6,
        linewidth=1.5, capsize=3, label=label,
    )


def _log_k_axis(ax, ticks, label):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(label)


def plot_k_sensitivity(n: int) -> None:
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(4 * len(SOURCES), 3.4))
    for ax, source in zip(axes, SOURCES, strict=True):
        metrics = load(source, f"n_{n}")
        _omgp_curve(ax, metrics, E1_KS)
        if f"{PREFIX}_sparse" in metrics:
            mean, se = mean_se(metrics[f"{PREFIX}_sparse"]["nlpds"])
            ax.errorbar(
                [14], [mean], yerr=[se], color=OMGP_COLOR, marker="D",
                markerfacecolor="white", markersize=7, capsize=3,
                linestyle="none", label="OMGP sparse (K≤10)",
            )
        _hline(ax, metrics, "pro_gp", PRO_COLOR, "PrO-GP (no K)", "-")
        _hline(ax, metrics, "standard_gp", GP_COLOR, "Bayes GP", "--")
        _log_k_axis(ax, [*E1_KS, 14], "mixture components K")
        ax.set_xticklabels([*map(str, E1_KS), "sp."])
        ax.set_title(SOURCE_TITLES[source], fontsize=13)
    axes[0].set_ylabel("test NLPD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    path = FIGURES_DIR / f"{PREFIX}_k_sensitivity_n{n}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


def k_sensitivity_table(ns) -> list[dict]:
    rows = []
    for source in SOURCES:
        for n in ns:
            metrics = load(source, f"n_{n}")
            if "pro_gp" not in metrics:
                continue
            ks = [k for k in E1_KS if f"{PREFIX}_k{k}" in metrics]
            if not ks:
                continue
            oracle_k = min(ks, key=lambda k: np.mean(metrics[f"{PREFIX}_k{k}"]["nlpds"]))
            row = {"source": source, "n": n, "oracle_k": oracle_k}
            for name, algorithm in [
                ("gp", "standard_gp"),
                ("oracle", f"{PREFIX}_k{oracle_k}"),
                ("valk", f"{PREFIX}_valk"),
                ("sparse", f"{PREFIX}_sparse"),
                *[(f"k{k}", f"{PREFIX}_k{k}") for k in ks],
            ]:
                if algorithm in metrics:
                    row[f"{name}_delta"], row[f"{name}_se"], row[f"{name}_pro_wins"] = (
                        paired(metrics, algorithm)
                    )
            row["pro_nlpd"], row["pro_se"] = mean_se(metrics["pro_gp"]["nlpds"])
            if f"{PREFIX}_valk" in metrics:
                row["valk_selected"] = json.dumps(metrics[f"{PREFIX}_valk"]["selected_k"])
                row["valk_runtime"] = float(np.median(metrics[f"{PREFIX}_valk"]["runtimes"]))
            if f"{PREFIX}_sparse" in metrics:
                row["sparse_occupied"] = float(
                    np.mean(metrics[f"{PREFIX}_sparse"]["occupied_components"])
                )
            rows.append(row)
    return rows


def plot_multibranch_heatmap(branches=(1, 2, 3, 4)) -> list[dict]:
    algorithms = [
        *(f"{PREFIX}_k{k}" for k in E1_KS), f"{PREFIX}_sparse", f"{PREFIX}_valk", "standard_gp",
    ]
    labels = [*(f"OMGP K={k}" for k in E1_KS), "OMGP sparse", "OMGP val-K", "Bayes GP"]
    delta = np.full((len(algorithms), len(branches)), np.nan)
    rows = []
    for j, num_branches in enumerate(branches):
        metrics = load("multibranch", f"num_branches_{num_branches}")
        if "pro_gp" not in metrics:
            continue
        for i, algorithm in enumerate(algorithms):
            if algorithm in metrics:
                mean, se, wins = paired(metrics, algorithm)
                delta[i, j] = mean
                rows.append({
                    "num_branches": num_branches, "algorithm": algorithm,
                    "delta_vs_pro": mean, "se": se, "pro_wins": wins,
                })
        if f"{PREFIX}_sparse" in metrics:
            rows.append({
                "num_branches": num_branches, "algorithm": f"{PREFIX}_sparse_occupied",
                "delta_vs_pro": float(np.mean(metrics[f"{PREFIX}_sparse"]["occupied_components"])),
            })
    if np.all(np.isnan(delta)):
        return rows

    table = np.column_stack([delta, np.nanmean(delta, axis=1)])
    bound = np.nanmax(np.abs(table))
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    image = ax.imshow(
        table, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -bound, bound), aspect="auto"
    )
    for (i, j), value in np.ndenumerate(table):
        if np.isfinite(value):
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 0.6 * bound else "#222222")
    ax.set_xticks(range(len(branches) + 1), [*map(str, branches), "mean"])
    ax.set_yticks(range(len(labels)), labels)
    ax.axvline(len(branches) - 0.5, color="white", linewidth=3)
    ax.set_xlabel("true number of branches K*")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, label="ΔNLPD vs PrO-GP (>0: PrO better)")
    fig.tight_layout()
    path = FIGURES_DIR / "multibranch_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    return rows


def plot_particles_vs_components() -> None:
    if not any(
        (RESULTS_ROOT / source / f"n_{E3_N}" / f"pro_gp_n{n}").exists()
        for source in E3_SOURCES
        for n in E3_NS
    ):
        return
    fig, axes = plt.subplots(1, len(E3_SOURCES), figsize=(4.6 * len(E3_SOURCES), 3.6))
    for ax, source in zip(axes, E3_SOURCES, strict=True):
        metrics = load(source, f"n_{E3_N}")
        _omgp_curve(ax, metrics, E3_KS, label="OMGP, K components")
        points = []
        for num_particles in E3_NS:
            name = "pro_gp" if num_particles == 50 else f"pro_gp_n{num_particles}"
            if name in metrics:
                points.append((num_particles, *mean_se(metrics[name]["nlpds"])))
        if points:
            x, mean, se = zip(*points, strict=True)
            ax.errorbar(
                x, mean, yerr=se, color=PRO_COLOR, marker="^", markersize=7,
                linewidth=1.5, capsize=3, label="PrO-GP, N particles",
            )
        _hline(ax, metrics, "standard_gp", GP_COLOR, "Bayes GP", "--")
        _log_k_axis(ax, sorted({*E3_KS, *E3_NS}), "K (OMGP) or N (PrO-GP)")
        ax.set_title(f"{SOURCE_TITLES[source]} (n={E3_N})", fontsize=13)
    axes[0].set_ylabel("test NLPD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    path = FIGURES_DIR / "particles_vs_components.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def print_k_table(rows: list[dict]) -> None:
    columns = ("gp", "oracle", "valk", "sparse", *(f"k{k}" for k in E1_KS))
    print("\nΔNLPD vs PrO-GP (mean±SE over instances; >0 means PrO better)")
    print(f"{'regime':<16}{'n':>5}{'PrO':>14}" + "".join(f"{c:>15}" for c in columns))
    for row in rows:
        line = f"{row['source']:<16}{row['n']:>5}{row['pro_nlpd']:>8.3f}±{row['pro_se']:.3f}"
        for c in columns:
            cell = (
                f"{row[f'{c}_delta']:+.3f}±{row[f'{c}_se']:.3f}"
                if f"{c}_delta" in row else "—"
            )
            line += f"{cell:>15}"
        print(line + f"   oracle K={row['oracle_k']}")


def main(n: int, prefix: str) -> None:
    global PREFIX  # noqa: PLW0603
    PREFIX = prefix
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_k_sensitivity(n)
    rows = k_sensitivity_table((50, 100, 200))
    print_k_table(rows)
    write_csv(rows, RESULTS_ROOT / f"{PREFIX}_k_sensitivity.csv")
    write_csv(plot_multibranch_heatmap(), RESULTS_ROOT / "mixture_multibranch.csv")
    plot_particles_vs_components()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--prefix", default="omgp", choices=("omgp", "omgp_shared"))
    args = parser.parse_args()
    main(args.n, args.prefix)
