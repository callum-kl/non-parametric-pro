"""Overlaid PIT density histograms, exact GP vs PrO-GP, one panel per dataset size."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate_results import collect

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

NUM_TRAINS = [200, 250, 300]
GP_METHOD = "exact_gp"
PRO_METHOD = "pro_gp_gibbs"

METHOD_COLORS = {"gp": "#4c3a8e", "pro": "#e8974e"}
METHOD_LABELS = {"gp": "Exact GP", "pro": "PrO-GP"}

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def plot_pit_overlay(pit_records, out_path: Path, *, num_bins: int = 10):
    fig, axes = plt.subplots(1, len(NUM_TRAINS), figsize=(15, 4.2), sharey=True)
    bins = [i / num_bins for i in range(num_bins + 1)]

    for ax, num_train in zip(axes, NUM_TRAINS, strict=True):
        ax.axhline(1.0, color="#999891", linewidth=1.0, linestyle="--", zorder=1)
        for method, key in (("gp", GP_METHOD), ("pro", PRO_METHOD)):
            values = pit_records.get((num_train, key))
            if not values:
                continue
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=METHOD_COLORS[method],
                label=f"{METHOD_LABELS[method]} (n={len(values)})",
            )
        ax.set_title(f"num_train={num_train}")
        ax.set_xlabel("PIT value")
        ax.set_xlim(0.0, 1.0)
        ax.legend(loc="upper right", fontsize=9)

    axes[0].set_ylabel("Density")
    fig.suptitle("PIT calibration: exact GP vs PrO-GP")
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    _, _, pit_records = collect()
    plot_pit_overlay(pit_records, FIGURES_DIR / "pit_overlay.png")
