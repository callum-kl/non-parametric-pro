"""NLPD vs. training set size, exact GP vs PrO-GP, with the gap between them made explicit."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate_results import collect, summarise

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

NUM_TRAINS = [200, 225, 250, 275]
GP_METHOD = "exact_gp"
PRO_METHOD = "pro_gp_gibbs_50"

# Same palette as the synthetic misspecification figures (experiments/synthetic/scripts/example_grid.py).
GP_COLOR = "#e8974e"
PRO_COLOR = "#2ca58d"
GP_LABEL = "Graph GP"
PRO_LABEL = "Graph PrO-GP"
GAP_COLOR = "#999891"

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


def plot_nlpd_by_num_train(summary, out_path: Path) -> None:
    gp_mean = [summary[(n, GP_METHOD)]["nlpd"]["mean"] for n in NUM_TRAINS]
    gp_se = [summary[(n, GP_METHOD)]["nlpd"]["se"] for n in NUM_TRAINS]
    pro_mean = [summary[(n, PRO_METHOD)]["nlpd"]["mean"] for n in NUM_TRAINS]
    pro_se = [summary[(n, PRO_METHOD)]["nlpd"]["se"] for n in NUM_TRAINS]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.fill_between(NUM_TRAINS, gp_mean, pro_mean, color=GAP_COLOR, alpha=0.15, zorder=1)

    ax.errorbar(
        NUM_TRAINS,
        gp_mean,
        yerr=gp_se,
        color=GP_COLOR,
        marker="o",
        markersize=7,
        linewidth=2.2,
        capsize=4,
        zorder=3,
    )
    ax.errorbar(
        NUM_TRAINS,
        pro_mean,
        yerr=pro_se,
        color=PRO_COLOR,
        marker="o",
        markersize=7,
        linewidth=2.2,
        capsize=4,
        zorder=3,
    )

    for n, gp, pro in zip(NUM_TRAINS, gp_mean, pro_mean, strict=True):
        ax.annotate(
            f"Δ={gp - pro:.3f}",
            xy=(n, (gp + pro) / 2),
            fontsize=14,
            color="#5a5850",
            ha="center",
            va="center",
        )

    end_offset = (NUM_TRAINS[-1] - NUM_TRAINS[0]) * 0.03
    ax.text(
        NUM_TRAINS[-1] + end_offset,
        gp_mean[-1],
        GP_LABEL,
        color=GP_COLOR,
        fontsize=12,
        fontweight="bold",
        va="center",
    )
    ax.text(
        NUM_TRAINS[-1] + end_offset,
        pro_mean[-1],
        PRO_LABEL,
        color=PRO_COLOR,
        fontsize=12,
        fontweight="bold",
        va="center",
    )

    ax.set_xlim(NUM_TRAINS[0] - 15, NUM_TRAINS[-1] + end_offset + 20)
    ax.set_xticks(NUM_TRAINS)
    ax.set_xlabel("Number of training sensors")
    ax.set_ylabel("Mean NLPD ± 1 SE")

    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    records, _, _ = collect()
    summary = summarise(records)
    plot_nlpd_by_num_train(summary, FIGURES_DIR / "nlpd_by_num_train.png")
