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

    fig, ax = plt.subplots(figsize=(7.0, 2.6))

    for n, gp, pro in zip(NUM_TRAINS, gp_mean, pro_mean, strict=True):
        ax.plot([n, n], [pro, gp], color=GAP_COLOR, linewidth=1.0, alpha=0.6, zorder=1)
        ax.annotate(
            f"Δ = {gp - pro:.3f}",
            xy=(n, (gp + pro) / 2),
            xytext=(7, 0),
            textcoords="offset points",
            fontsize=12,
            color="#5a5850",
            ha="left",
            va="center",
        )

    for mean, se, color in [(gp_mean, gp_se, GP_COLOR), (pro_mean, pro_se, PRO_COLOR)]:
        ax.errorbar(
            NUM_TRAINS,
            mean,
            yerr=se,
            color=color,
            marker="o",
            markersize=6,
            linewidth=2.0,
            capsize=3,
            zorder=3,
        )

    for mean, color, label in [(gp_mean, GP_COLOR, GP_LABEL), (pro_mean, PRO_COLOR, PRO_LABEL)]:
        ax.annotate(
            label,
            xy=(NUM_TRAINS[-1], mean[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=11,
            fontweight="bold",
            va="center",
            annotation_clip=False,
        )

    ax.set_xlim(NUM_TRAINS[0] - 6, NUM_TRAINS[-1] + 6)
    ax.set_xticks(NUM_TRAINS)
    ax.set_ylim(0.7, 1.2)
    ax.set_yticks([0.8, 1.0, 1.2])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Training sensors")
    ax.set_ylabel("NLPD ± 1 SE")

    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    records, _, _ = collect()
    summary = summarise(records)
    plot_nlpd_by_num_train(summary, FIGURES_DIR / "nlpd_by_num_train.png")
