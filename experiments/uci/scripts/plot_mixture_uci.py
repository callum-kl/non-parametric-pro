"""
Test NLPD of the OMGP baselines against the number of mixture components M, with PrO-GP
and the GP for reference: exact-GP datasets on top, inducing-point datasets below.
Each panel uses the splits on which every method has run.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from non_parametric_pro.data.uci.uci import UCI_REGRESSION_DATASET_SIZES

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
FIGURE_PATH = ROOT / "figures" / "omgp_shared_uci_raw_combined.png"

KS = (1, 2, 3, 5, 10, 20)
EXACT = ("concreteslump", "autos", "machine", "autompg", "housing", "stock", "energy", "concrete")
INDUCING = ("wine", "skillcraft", "abalone", "parkinsons", "airquality", "elevators", "kin40k",
            "protein")

# (method dir, metrics file, NLPD key, label) for the GP and PrO-GP of each row.
EXACT_BASELINES = (
    ("exact_gp", "gp_metrics.json", "gp_nlpd", "Bayes GP"),
    ("pro_gp_gibbs", "pro_metrics.json", "pro_nlpd", "PrO-GP"),
)
INDUCING_BASELINES = (
    ("vgp_noncollapsed", "gp_metrics.json", "vgp_nlpd", "SVGP"),
    ("inducing_pro_gp_gibbs", "pro_metrics.json", "pro_nlpd", "Inducing PrO-GP"),
)

# Same PrO-GP / GP colours as the other figures (plot_convergence.py, example_grid.py).
PRO_COLOR = "#2ca58d"
GP_COLOR = "#e8974e"
OMGP_COLOR = "#3b6fb6"

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


def collect(dataset: str, baselines) -> dict[str, np.ndarray]:
    """{"gp" | "pro" | K: per-split NLPDs}, over the splits where every method has run."""
    files = {
        "gp": baselines[0][:3],
        "pro": baselines[1][:3],
        **{k: (f"omgp_shared_k{k}", "mixture_metrics.json", "mixture_nlpd") for k in KS},
    }
    by_split = {}
    for split_dir in (RESULTS_ROOT / dataset).glob("split_*"):
        paths = {name: split_dir / run / fname for name, (run, fname, _) in files.items()}
        if all(p.exists() for p in paths.values()):
            by_split[int(split_dir.name.split("_")[1])] = {
                name: json.loads(paths[name].read_text())[files[name][2]] for name in files
            }
    splits = sorted(by_split)
    return {name: np.array([by_split[s][name] for s in splits]) for name in files}


def mean_se(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def plot_row(subfig, datasets, baselines) -> None:
    gp_label, pro_label = baselines[0][3], baselines[1][3]
    datasets = sorted(datasets, key=lambda d: UCI_REGRESSION_DATASET_SIZES[d][0])
    axes = subfig.subplots(2, 4, sharex=True)
    for ax, dataset in zip(axes.flat, datasets, strict=True):
        nlpds = collect(dataset, baselines)
        mean, se = zip(*(mean_se(nlpds[k]) for k in KS), strict=True)
        ax.errorbar(KS, mean, yerr=se, color=OMGP_COLOR, marker="o", capsize=3,
                    linewidth=1.5, label="OMGP, M components")
        pro_mean, pro_se = mean_se(nlpds["pro"])
        ax.axhline(pro_mean, color=PRO_COLOR, linewidth=1.5, label=pro_label)
        ax.axhspan(pro_mean - pro_se, pro_mean + pro_se, color=PRO_COLOR, alpha=0.12, linewidth=0)
        ax.axhline(nlpds["gp"].mean(), color=GP_COLOR, linestyle="--", linewidth=1.2,
                   label=gp_label)
        ax.set_xscale("log")
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(labelsize=13)
        ax.set_title(dataset, fontsize=16)
    for ax in axes[:, 0]:
        ax.set_ylabel("NLPD")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    subfig.legend(handles, labels, loc="lower center", ncol=3, fontsize=16,
                  bbox_to_anchor=(0.5, -0.07))


def main() -> None:
    fig = plt.figure(figsize=(14, 13), layout="constrained")
    top, bottom = fig.subfigures(2, 1, hspace=0.08)
    plot_row(top, EXACT, EXACT_BASELINES)
    plot_row(bottom, INDUCING, INDUCING_BASELINES)
    FIGURE_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
