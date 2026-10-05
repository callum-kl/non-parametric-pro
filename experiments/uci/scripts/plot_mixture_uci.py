"""PrO-GP against the OMGP baselines on the exact-GP UCI datasets: table and figure."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from run_uci import EXACT_DATASETS

from non_parametric_pro.data.uci.uci import UCI_REGRESSION_DATASET_SIZES

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

KS = (1, 2, 3, 5)
PREFIX = "omgp"


def methods() -> dict[str, tuple[str, str]]:
    return {
        "exact_gp": ("gp_metrics.json", "gp_nlpd"),
        "pro_gp_gibbs": ("pro_metrics.json", "pro_nlpd"),
        **{
            f"{PREFIX}_{v}": ("mixture_metrics.json", "mixture_nlpd")
            for v in (*(f"k{k}" for k in KS), "sparse", "valk")
        },
    }


def labels() -> dict[str, str]:
    return {
        "exact_gp": "GP",
        "pro_gp_gibbs": "PrO-GP",
        **{f"{PREFIX}_k{k}": f"OMGP K={k}" for k in KS},
        f"{PREFIX}_sparse": "OMGP sparse",
        f"{PREFIX}_valk": "OMGP val-K",
    }


OMGP_COLOR = "#3b6fb6"
GP_COLOR = "#4c3a8e"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def collect(dataset: str) -> dict[str, dict[int, float]]:
    """{method: {split: nlpd}}, restricted to splits where every method has run."""
    records = {m: {} for m in methods()}
    for split_dir in (RESULTS_ROOT / dataset).glob("split_*"):
        split = int(split_dir.name.split("_")[1])
        for method, (fname, key) in methods().items():
            path = split_dir / method / fname
            if path.exists():
                records[method][split] = json.loads(path.read_text())[key]
    common = set.intersection(*(set(v) for v in records.values()))
    return {m: {s: v[s] for s in sorted(common)} for m, v in records.items()}


def selected_ks(dataset: str) -> list[int]:
    paths = sorted((RESULTS_ROOT / dataset).glob(f"split_*/{PREFIX}_valk/mixture_metrics.json"))
    return [json.loads(p.read_text())["selected_k"] for p in paths]


def mean_se(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(se)


def main(prefix: str) -> None:
    global PREFIX  # noqa: PLW0603
    PREFIX = prefix
    datasets = sorted(EXACT_DATASETS, key=lambda d: UCI_REGRESSION_DATASET_SIZES[d][0])
    rows, ranks = [], {m: [] for m in methods()}
    deltas = {}
    for dataset in datasets:
        records = collect(dataset)
        splits = list(records["pro_gp_gibbs"])
        if not splits:
            continue
        pro = np.array(list(records["pro_gp_gibbs"].values()))
        row = {"dataset": dataset, "num_splits": len(splits)}
        means = {}
        for method, by_split in records.items():
            values = np.array(list(by_split.values()))
            means[method], row[f"{method}_se"] = mean_se(values)
            row[f"{method}_mean"] = means[method]
            row[f"{method}_delta_vs_pro"], row[f"{method}_delta_se"] = mean_se(values - pro)
        row["oracle_k"] = min(KS, key=lambda k: means[f"{PREFIX}_k{k}"])
        row["valk_selected"] = json.dumps(selected_ks(dataset))
        order = sorted(means, key=means.get)
        for method in methods():
            ranks[method].append(order.index(method) + 1)
        deltas[dataset] = {
            k: (row[f"{PREFIX}_k{k}_delta_vs_pro"], row[f"{PREFIX}_k{k}_delta_se"]) for k in KS
        } | {
            v: (row[f"{PREFIX}_{v}_delta_vs_pro"], row[f"{PREFIX}_{v}_delta_se"])
            for v in ("sparse", "valk")
        } | {"gp": (row["exact_gp_delta_vs_pro"], row["exact_gp_delta_se"])}
        rows.append(row)

    if not rows:
        print("No OMGP results found; run run_uci.py --mixture-only first.")
        return

    print(f"{'NLPD':<14}" + "".join(f"{labels()[m]:>16}" for m in methods()) + "   oracle K")
    for row in rows:
        line = f"{row['dataset']:<14}"
        for method in methods():
            line += f"{row[f'{method}_mean']:>9.3f}±{row[f'{method}_se']:.3f}"
        print(line + f"   {row['oracle_k']}")
    print(f"{'mean rank':<14}" + "".join(f"{np.mean(ranks[m]):>16.2f}" for m in methods()))

    path = RESULTS_ROOT / f"{PREFIX}_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for ax, dataset in zip(axes.flat, deltas, strict=False):
        by = deltas[dataset]
        mean, se = zip(*(by[k] for k in KS), strict=True)
        ax.errorbar(KS, mean, yerr=se, color=OMGP_COLOR, marker="o", capsize=3,
                    linewidth=1.5, label="OMGP, K components")
        for x, key, marker, label in ((7, "sparse", "D", "OMGP sparse"),
                                      (10, "valk", "s", "OMGP val-K")):
            ax.errorbar([x], [by[key][0]], yerr=[by[key][1]], color=OMGP_COLOR,
                        marker=marker, markerfacecolor="white", capsize=3,
                        linestyle="none", label=label)
        ax.axhline(0.0, color="#e8974e", linewidth=1.5, label="PrO-GP (reference)")
        ax.axhline(by["gp"][0], color=GP_COLOR, linestyle="--", linewidth=1.2, label="Bayes GP")
        ax.set_xscale("log")
        ax.set_xticks([*KS, 7, 10], [*map(str, KS), "sp.", "val"])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(dataset, fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("ΔNLPD vs PrO-GP\n(>0: PrO better)")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{PREFIX}_uci.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="omgp", choices=("omgp", "omgp_shared"))
    main(parser.parse_args().prefix)
