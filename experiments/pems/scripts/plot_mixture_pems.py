"""PrO-GP against the OMGP baselines on PeMS, one panel per training size."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

NUM_TRAINS = (200, 225, 250, 275)
KS = (1, 2, 3, 5)
OMGP_COLOR = "#3b6fb6"
PRO_COLOR = "#e8974e"
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


def nlpd(split_dir: Path, method: str) -> float | None:
    fname, key = {
        "exact_gp": ("gp_metrics.json", "gp_nlpd"),
        "pro_gp_gibbs": ("pro_metrics.json", "pro_nlpd"),
    }.get(method, ("mixture_metrics.json", "mixture_nlpd"))
    path = split_dir / method / fname
    return json.loads(path.read_text())[key] if path.exists() else None


def paired_deltas(num_train: int, prefix: str) -> tuple[dict[str, tuple[float, float]], int]:
    """({variant: (mean, se)} of NLPD(variant) - NLPD(PrO), number of splits), over splits where all have run."""
    variants = {
        **{f"k{k}": f"{prefix}_k{k}" for k in KS},
        "sparse": f"{prefix}_sparse",
        "valk": f"{prefix}_valk",
        "gp": "exact_gp",
    }
    diffs = {v: [] for v in variants}
    for split_dir in (RESULTS_ROOT / f"num_train_{num_train}").glob("split_*"):
        pro = nlpd(split_dir, "pro_gp_gibbs")
        values = {v: nlpd(split_dir, m) for v, m in variants.items()}
        if pro is None or any(x is None for x in values.values()):
            continue
        for v, x in values.items():
            diffs[v].append(x - pro)
    return {
        v: (float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(len(d))))
        for v, d in diffs.items()
        if len(d) > 1
    }, len(diffs["gp"])


def main(prefix: str) -> None:
    fig, axes = plt.subplots(1, len(NUM_TRAINS), figsize=(3.6 * len(NUM_TRAINS), 3.4), sharey=True)
    for ax, num_train in zip(axes, NUM_TRAINS, strict=True):
        by, num_splits = paired_deltas(num_train, prefix)
        if not by:
            continue
        mean, se = zip(*(by[f"k{k}"] for k in KS), strict=True)
        ax.errorbar(KS, mean, yerr=se, color=OMGP_COLOR, marker="o", capsize=3,
                    linewidth=1.5, label="OMGP, K components")
        for x, key, marker, label in ((7, "sparse", "D", "OMGP sparse"),
                                      (10, "valk", "s", "OMGP val-K")):
            ax.errorbar([x], [by[key][0]], yerr=[by[key][1]], color=OMGP_COLOR,
                        marker=marker, markerfacecolor="white", capsize=3,
                        linestyle="none", label=label)
        ax.axhline(0.0, color=PRO_COLOR, linewidth=1.5, label="PrO-GP (reference)")
        ax.axhline(by["gp"][0], color=GP_COLOR, linestyle="--", linewidth=1.2, label="graph GP")
        ax.set_xscale("log")
        ax.set_xticks([*KS, 7, 10], [*map(str, KS), "sp.", "val"])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(f"n_train = {num_train} ({num_splits} splits)", fontsize=12)
    axes[0].set_ylabel("ΔNLPD vs PrO-GP\n(>0: PrO better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{prefix}_pems.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="omgp_shared", choices=("omgp", "omgp_shared"))
    main(parser.parse_args().prefix)
