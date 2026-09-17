"""
Run gp_normality.py across datasets and splits, then summarise across splits.

Writes a per-split table (normality_summary.csv) and one overlay figure per
dataset showing every split's Q-Q curve on shared axes.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR.parent / "results"

DEFAULT_DATASETS = ["machine", "forest"]
RESIDUAL_TYPES = ["whitened", "loo", "test"]
SERIES = {"whitened": "#2a78d6", "loo": "#eb6834", "test": "#1baf7a"}
LABELS = {
    "whitened": "Whitened train",
    "loo": "Leave-one-out train",
    "test": "Held-out test",
}
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd6"
REF = "#8a8984"


def normality_dir(dataset: str, split: int) -> Path:
    return RESULTS_ROOT / dataset / f"split_{split}" / "exact_gp" / "normality"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def plot_split_overlay(dataset: str, splits: list[int], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURFACE, sharey=True)
    for ax, kind in zip(axes, RESIDUAL_TYPES, strict=True):
        colour = SERIES[kind]
        drawn = 0
        for split in splits:
            npz = normality_dir(dataset, split) / "residuals.npz"
            if not npz.exists():
                continue
            drawn += 1
            z = np.sort(np.load(npz)[f"z_{kind}"])
            theoretical = stats.norm.ppf(
                (np.arange(1, z.size + 1) - 0.375) / (z.size + 0.25)
            )
            ax.plot(
                theoretical,
                z,
                color=colour,
                alpha=0.75,
                linewidth=1.6,
                marker="o",
                markersize=2.4,
                markeredgewidth=0,
            )
        lim = np.array([-4.0, 4.0])
        ax.plot(lim, lim, color=REF, linewidth=2, zorder=1)
        ax.set_xlim(*lim)
        ax.set_title(
            f"{LABELS[kind]}  ({drawn} splits overlaid)", fontsize=10.5, color=INK
        )
        ax.set_xlabel("Theoretical N(0, 1) quantile", fontsize=9, color=INK_MUTED)
        _style_axes(ax)
    axes[0].set_ylabel("Observed quantile", fontsize=9, color=INK_MUTED)
    fig.suptitle(
        f"{dataset} -- exact GP Q-Q across splits (grey line = perfect normality)",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default="1,2,3,4,5")
    parser.add_argument(
        "--skip-run", action="store_true", help="only re-summarise existing outputs"
    )
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    splits = [int(s) for s in args.splits.split(",")]
    script = str(SCRIPT_DIR / "gp_normality.py")

    if not args.skip_run:
        for dataset in datasets:
            cmd = [
                sys.executable,
                script,
                "-m",
                f"ds@_global_={dataset}",
                f"split={','.join(str(s) for s in splits)}",
            ]
            print(f"\n$ {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)

    rows = []
    for dataset in datasets:
        for split in splits:
            stats_path = normality_dir(dataset, split) / "normality_stats.json"
            if not stats_path.exists():
                continue
            payload = json.loads(stats_path.read_text())
            for kind in RESIDUAL_TYPES:
                s = payload["residuals"][kind]
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "residuals": kind,
                        "n": s["n"],
                        "std": round(s["std"], 4),
                        "skew": round(s["skew"], 4),
                        "excess_kurtosis": round(s["excess_kurtosis"], 4),
                        "shapiro_p": f"{s['shapiro_p']:.3e}",
                        "ks_p": f"{s['ks_standard_normal_p']:.3e}",
                        "jarque_bera_p": f"{s['jarque_bera_p']:.3e}",
                        "frac_beyond_2sigma": round(s["frac_beyond_2sigma"], 4),
                        "target_modal_fraction": round(
                            payload["target"]["modal_fraction"], 4
                        ),
                    }
                )
        plot_split_overlay(
            dataset, splits, RESULTS_ROOT / dataset / "normality_qq_splits.png"
        )

    if not rows:
        print("No normality outputs found.")
        return

    out_csv = RESULTS_ROOT / "normality_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")

    header = f"{'dataset':9s} {'split':>5s} {'residuals':10s} {'n':>5s} {'skew':>7s} {'ex.kurt':>8s} {'shapiro_p':>11s} {'ks_p':>11s}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['dataset']:9s} {r['split']:5d} {r['residuals']:10s} {r['n']:5d} "
            f"{r['skew']:+7.3f} {r['excess_kurtosis']:+8.3f} {r['shapiro_p']:>11s} {r['ks_p']:>11s}"
        )


if __name__ == "__main__":
    main()
