"""Overlaid PIT density histograms, exact GP vs PrO, one panel per condition."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate_results import collect

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

CONDITIONS = [
    ("Baseline", "exact_gp", "pro_gp_gibbs"),
    (
        "Regime shift (679, 680)",
        "exact_gp_regime_shift_679_680",
        "pro_gp_regime_shift_679_680",
    ),
    (
        "Regime shift (343, 346)",
        "exact_gp_regime_shift_343_346",
        "pro_gp_regime_shift_343_346",
    ),
]

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
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(15, 4.2), sharey=True)
    bins = [i / num_bins for i in range(num_bins + 1)]

    for ax, (title, gp_key, pro_key) in zip(axes, CONDITIONS, strict=True):
        ax.axhline(1.0, color="#999891", linewidth=1.0, linestyle="--", zorder=1)
        for method, key in (("gp", gp_key), ("pro", pro_key)):
            values = pit_records.get(key)
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
        ax.set_title(title)
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
