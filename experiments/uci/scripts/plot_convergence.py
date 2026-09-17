"""Convergence traces and per-iteration wall-clock vs number of inducing points, VGP vs inducing PrO-GP."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

UCI_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = UCI_DIR / "figures"

MS = [100, 250, 500, 1000]
# m=1000 has no full-length Adam trace, only the short run behind its timing point.
VGP_MS = [100, 250, 500]
TIME_MS = [100, 250, 500, 1000]
SEEDS = [0, 1, 2]

# Filled in by eye after the first render; None means no marker.
CONVERGENCE_ITERS = {
    "vgp": {100: None, 250: None, 500: None, 1000: None},
    "pro": {100: None, 250: None, 500: None, 1000: None},
}

VGP_COLOR = "#e8974e"
PRO_COLOR = "#2ca58d"
MARKER_COLOR = "#d62728"

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


def ramp(color: str, n: int) -> list:
    cmap = LinearSegmentedColormap.from_list("ramp", ["#f2efe6", color, "#1f1f1f"])
    return [cmap(t) for t in np.linspace(0.3, 0.75, n)]


def load_runs(split_dir: Path, prefix: str, m: int) -> list:
    paths = [split_dir / f"{prefix}_conv_m{m}_s{s}" / "convergence.npz" for s in SEEDS]
    return [np.load(p) for p in paths if p.exists()]


def stack(arrays: list[np.ndarray]) -> np.ndarray:
    length = min(len(a) for a in arrays)
    return np.stack([a[:length] for a in arrays])


def seconds_per_iter(run) -> float:
    return float(np.median(run["chunk_times"][1:]) / run["chunk_len"])


def plot_trace(ax, x, traces, color, label):
    mean = traces.mean(axis=0)
    ax.plot(x, mean, color=color, lw=2, label=label)
    if len(traces) > 1:
        ax.fill_between(x, traces.min(axis=0), traces.max(axis=0), color=color, alpha=0.2, lw=0)
    return mean


def mark_convergence(ax, x, mean, iteration):
    if iteration is None:
        return
    ax.plot(iteration, np.interp(iteration, x, mean), "x", color=MARKER_COLOR, ms=10, mew=2.5, zorder=5)


def draw_pro_traces(ax, split_dir: Path) -> None:
    for m, color in zip(MS, ramp(PRO_COLOR, len(MS))):
        runs = load_runs(split_dir, "inducing_pro_gp", m)
        if not runs:
            continue
        scores = -stack([r["avg_score"] for r in runs])
        x = np.arange(1, scores.shape[1] + 1)
        mean = plot_trace(ax, x, scores, color, f"m = {m}")
        mark_convergence(ax, x, mean, CONVERGENCE_ITERS["pro"][m])
    ax.set(xlabel="Gibbs iteration", ylabel="Average training score")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=10)


def draw_time_per_iteration(ax, split_dir: Path) -> None:
    for prefix, color, label in [
        ("vgp_noncollapsed", VGP_COLOR, "SVGP"),
        ("inducing_pro_gp", PRO_COLOR, "I-PRO-GP"),
    ]:
        ms, means, lows, highs = [], [], [], []
        for m in TIME_MS:
            times = [seconds_per_iter(r) for r in load_runs(split_dir, prefix, m)]
            if times:
                ms.append(m)
                means.append(np.mean(times))
                lows.append(np.mean(times) - min(times))
                highs.append(max(times) - np.mean(times))
        if not ms:
            continue
        ax.errorbar(ms, means, yerr=[lows, highs], color=color, lw=2, marker="o", ms=8, capsize=3)
        ax.annotate(label, (ms[-1], means[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10, color="#3d3d3a")

    ax.set_yscale("log")
    ax.set_xticks(TIME_MS)
    ax.set(xlabel="Number of inducing points", ylabel="Seconds per iteration")


def save(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined(split_dir: Path, out: Path) -> None:
    fig, (ax_time, ax_pro) = plt.subplots(1, 2, figsize=(11, 4.2))
    draw_time_per_iteration(ax_time, split_dir)
    draw_pro_traces(ax_pro, split_dir)
    ax_time.set_title("Cost per iteration")
    ax_pro.set_title("I-PRO-GP convergence")
    save(fig, out)


def plot_iterations(split_dir: Path, out: Path) -> None:
    fig, (ax_vgp, ax_pro) = plt.subplots(1, 2, figsize=(11, 4.2))
    vgp_tops, vgp_bottoms = [], []

    for m, vgp_color in zip(MS, ramp(VGP_COLOR, len(MS))):
        vgp_runs = load_runs(split_dir, "vgp_noncollapsed", m) if m in VGP_MS else []
        if not vgp_runs:
            continue
        n = float(vgp_runs[0]["n_train"])
        chunk_len = int(vgp_runs[0]["chunk_len"])
        minibatch = stack([r["minibatch_loss"] for r in vgp_runs]).mean(axis=0) / n
        ax_vgp.plot(np.arange(1, len(minibatch) + 1), minibatch, color=vgp_color, lw=0.5, alpha=0.25)
        full = stack([r["full_neg_elbo"] for r in vgp_runs]) / n
        x = chunk_len * np.arange(1, full.shape[1] + 1)
        mean = plot_trace(ax_vgp, x, full, vgp_color, f"m = {m}")
        mark_convergence(ax_vgp, x, mean, CONVERGENCE_ITERS["vgp"][m])
        vgp_tops.append(mean[min(4, len(mean) - 1)])
        vgp_bottoms.append(full.min())

    if vgp_tops:
        # The first few hundred Adam steps are orders of magnitude above the plateau.
        bottom, top = min(vgp_bottoms), max(vgp_tops)
        ax_vgp.set_ylim(bottom - 0.05 * (top - bottom), top)

    draw_pro_traces(ax_pro, split_dir)
    ax_vgp.set(title="VGP", xlabel="Adam iteration", ylabel="Negative ELBO per training point")
    ax_pro.set_title("I-PRO-GP")
    if ax_vgp.get_legend_handles_labels()[0]:
        ax_vgp.legend(fontsize=10)
    save(fig, out)


def plot_time_per_iteration(split_dir: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    draw_time_per_iteration(ax, split_dir)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=UCI_DIR / "results")
    parser.add_argument("--dataset", default="parkinsons")
    parser.add_argument("--split", default=1, type=int)
    args = parser.parse_args()

    split_dir = args.results_root / args.dataset / f"split_{args.split}"
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_combined(split_dir, FIGURES_DIR / f"time_and_convergence_{args.dataset}.pdf")
    plot_iterations(split_dir, FIGURES_DIR / f"convergence_iterations_{args.dataset}.pdf")
    plot_time_per_iteration(split_dir, FIGURES_DIR / f"time_per_iteration_{args.dataset}.pdf")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
