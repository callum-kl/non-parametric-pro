import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from example_grid import (
    GP_COLOR,
    PANEL_ADAPT_STEPS,
    PANEL_ALPHA,
    PANEL_KERNEL_ADAPT_STEPS,
    PANEL_KERNEL_LR,
    PANEL_KERNEL_STEPS_PER_ADAPT,
    PANEL_NUM_PARTICLES,
    PRO_BAND_ALPHAS,
    PRO_COLOR,
    PRO_LABEL,
    X_TICKS,
    DATA_ALPHA,
    _display_affine,
    _pro_density_bands,
    _style_box,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from synthetic import fit_gp, fit_pro

from non_parametric_pro.data.synthetic.illustrative import (
    make_illustrative_instance,
    plot_illustrative_case,
)

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
REGIME = "multimodal"
DEFAULT_INDEX = 10
# tighter than the shared grid's frame, which is sized for four panels at once
YLIM = (-1.3, 1.3)
FIGSIZE = (7.4, 4.6)
DATA_MARKER_SIZE = 22
TICK_FONTSIZE = 15

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def main(seed: int, num_instances: int, index: int, *, algorithm: str) -> None:
    key = jr.split(jr.PRNGKey(seed), num_instances)[index]
    data = make_illustrative_instance(key, regime=REGIME)
    scale, shift = _display_affine(data)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_illustrative_case(ax, data, data_alpha=DATA_ALPHA, scale=scale, shift=shift)
    for collection in ax.collections:
        collection.set_sizes([DATA_MARKER_SIZE])

    order = np.argsort(data.x_test[:, 0])
    x_sorted = np.asarray(data.x_test[order, 0])
    gp_key, fit_key = jr.split(key)

    pro = fit_pro(
        data,
        fit_key,
        algorithm=algorithm,
        alpha=PANEL_ALPHA,
        num_particles=PANEL_NUM_PARTICLES,
        num_adapt_steps=PANEL_ADAPT_STEPS,
        kernel_adapt_steps=PANEL_KERNEL_ADAPT_STEPS,
        kernel_lr=PANEL_KERNEL_LR,
        kernel_steps_per_adapt=PANEL_KERNEL_STEPS_PER_ADAPT,
        warmup_steps=4,
        kernel_lengthscale=0.4,
        val_fraction=0.3,
    )
    _pro_density_bands(
        ax,
        x_sorted,
        scale * np.asarray(pro.mean)[order] + shift,
        scale * np.asarray(pro.std)[order],
        scale * np.asarray(pro.particle_predictions)[order] + shift,
        scale * np.asarray(pro.sigma_eff)[order],
    )

    gp = fit_gp(data, gp_key)
    ax.plot(
        x_sorted,
        scale * np.asarray(gp.mean)[order] + shift,
        color=GP_COLOR,
        linewidth=2.0,
        zorder=5,
    )

    _style_box(ax, grid=True, grid_axis="y")
    ax.set_ylim(*YLIM)
    ax.set_xticks(X_TICKS)
    ax.tick_params(labelsize=TICK_FONTSIZE)

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.3,
                alpha=0.7,
                label="truth",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="None",
                markersize=6,
                label="data",
            ),
            Line2D([0], [0], color=GP_COLOR, linewidth=2.0, label="GP mean"),
            Patch(
                facecolor=PRO_COLOR,
                alpha=PRO_BAND_ALPHAS[1],
                label=f"{PRO_LABEL} (50%/95% regions)",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        fontsize=12,
    )

    out_path = FIGURES_DIR / "multimodal_overlay.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2421)
    parser.add_argument("--num-instances", type=int, default=20)
    parser.add_argument("--index", type=int, default=DEFAULT_INDEX)
    parser.add_argument(
        "--algorithm", default="replica_gibbs", choices=["ula", "replica_gibbs"]
    )
    args = parser.parse_args()
    main(args.seed, args.num_instances, args.index, algorithm=args.algorithm)
