import argparse
import os
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from blackjax.types import PRNGKey
from example_grid import TOP_LEGEND_HANDLES, SourceSpec, plot_fit_grid

from non_parametric_pro.util import train_val_split

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


class PaperCase(NamedTuple):
    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_truth_test: jax.Array
    y_test: jax.Array
    y_truth_alt_train: jax.Array | None = None
    y_truth_alt_test: jax.Array | None = None
    is_outlier_train: jax.Array | None = None


def _smooth_truth(x: jax.Array) -> jax.Array:
    """Shared single-dip truth curve underlying the well-specified, block-outliers,
    and heteroskedastic examples -- these differ only in their noise process."""
    return jnp.cos(jnp.pi * (x + 0.15)) - 0.25


def _mixture_truth_a(x: jax.Array) -> jax.Array:
    return 0.6 * jnp.cos(jnp.pi * (x + 0.1)) - 0.1


def _mixture_truth_b(x: jax.Array) -> jax.Array:
    return -_mixture_truth_a(x)


def _split_case(
    key: PRNGKey,
    x: jax.Array,
    y_truth: jax.Array,
    y_obs: jax.Array,
    *,
    test_fraction: float,
    y_truth_alt: jax.Array | None = None,
    is_outlier: jax.Array | None = None,
) -> PaperCase:
    split = train_val_split(key, x, y_obs, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx
    return PaperCase(
        x_train=x[train_idx].reshape(-1, 1),
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_alt_train=y_truth_alt[train_idx].reshape(-1, 1)
        if y_truth_alt is not None
        else None,
        y_truth_alt_test=y_truth_alt[test_idx].reshape(-1, 1)
        if y_truth_alt is not None
        else None,
        is_outlier_train=is_outlier[train_idx] if is_outlier is not None else None,
    )


def make_well_specified_instance(
    key: PRNGKey, *, n: int = 100, test_fraction: float = 0.3, noise_std: float = 0.08
) -> PaperCase:
    x_key, noise_key, split_key = jr.split(key, 3)
    x = jr.uniform(x_key, (n,), minval=0.0, maxval=1.0)
    y_truth = _smooth_truth(x)
    y_obs = y_truth + noise_std * jr.normal(noise_key, (n,))
    return _split_case(split_key, x, y_truth, y_obs, test_fraction=test_fraction)


def make_block_outliers_instance(
    key: PRNGKey,
    *,
    n: int = 100,
    test_fraction: float = 0.3,
    noise_std: float = 0.06,
    outlier_region: tuple[float, float] = (0.35, 0.55),
    outlier_offset: float = 1.0,
    outlier_frac: float = 0.6,
) -> PaperCase:
    x_key, noise_key, outlier_key, split_key = jr.split(key, 4)
    x = jr.uniform(x_key, (n,), minval=0.0, maxval=1.0)
    y_truth = _smooth_truth(x)
    y_obs = y_truth + noise_std * jr.normal(noise_key, (n,))

    in_region = (x >= outlier_region[0]) & (x <= outlier_region[1])
    is_outlier = in_region & (jr.uniform(outlier_key, (n,)) < outlier_frac)
    y_obs = jnp.where(is_outlier, y_truth + outlier_offset, y_obs)

    return _split_case(
        split_key, x, y_truth, y_obs, test_fraction=test_fraction, is_outlier=is_outlier
    )


def make_heteroskedastic_instance(
    key: PRNGKey,
    *,
    n: int = 100,
    test_fraction: float = 0.3,
    noise_std: float = 0.05,
    noisy_region: tuple[float, float] = (0.0, 0.22),
    noisy_std: float = 0.3,
) -> PaperCase:
    x_key, noise_key, split_key = jr.split(key, 3)
    x = jr.uniform(x_key, (n,), minval=0.0, maxval=1.0)
    y_truth = _smooth_truth(x)
    in_region = (x >= noisy_region[0]) & (x <= noisy_region[1])
    sigma = jnp.where(in_region, noisy_std, noise_std)
    y_obs = y_truth + sigma * jr.normal(noise_key, (n,))
    return _split_case(split_key, x, y_truth, y_obs, test_fraction=test_fraction)


def make_merging_mixture_instance(
    key: PRNGKey,
    *,
    n: int = 100,
    test_fraction: float = 0.3,
    noise_std: float = 0.07,
    mix_prob: float = 0.5,
) -> PaperCase:
    x_key, branch_key, noise_key, split_key = jr.split(key, 4)
    x = jr.uniform(x_key, (n,), minval=0.0, maxval=1.0)
    y_truth_a = _mixture_truth_a(x)
    y_truth_b = _mixture_truth_b(x)
    use_a = jr.bernoulli(branch_key, mix_prob, (n,))
    y_truth = jnp.where(use_a, y_truth_a, y_truth_b)
    y_obs = y_truth + noise_std * jr.normal(noise_key, (n,))
    return _split_case(
        split_key,
        x,
        y_truth_a,
        y_obs,
        test_fraction=test_fraction,
        y_truth_alt=y_truth_b,
    )


def plot_paper_case(
    ax,
    data: PaperCase,
    *,
    show_curve: bool = True,
    show_train: bool = False,
    color_by_outlier: bool = False,
) -> None:
    if show_curve:
        x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
        y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
        order = jnp.argsort(x_full)
        ax.plot(x_full[order], y_full[order], color="C0", linewidth=1.5)
        if data.y_truth_alt_train is not None:
            y_alt_full = jnp.concatenate(
                [data.y_truth_alt_train[:, 0], data.y_truth_alt_test[:, 0]]
            )
            ax.plot(x_full[order], y_alt_full[order], color="C1", linewidth=1.5)

    if show_train:
        ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    else:
        ax.scatter(data.x_test, data.y_test, color="black", s=8, zorder=3)

    if color_by_outlier and data.is_outlier_train is not None:
        idx = jnp.where(data.is_outlier_train)[0]
        ax.scatter(
            data.x_train[idx],
            data.y_train[idx],
            color="maroon",
            marker="x",
            s=40,
            linewidths=1.5,
            zorder=4,
        )


_SOURCES = [
    SourceSpec(
        "well_specified",
        "Well-specified",
        make_well_specified_instance,
        plot_paper_case,
        {},
        {"show_train": False},
        instance_index=0,
    ),
    SourceSpec(
        "block_outliers",
        "Block outliers",
        make_block_outliers_instance,
        plot_paper_case,
        {},
        {"show_train": False, "color_by_outlier": True},
        instance_index=0,
    ),
    SourceSpec(
        "heteroskedastic",
        "Heteroskedastic",
        make_heteroskedastic_instance,
        plot_paper_case,
        {},
        {"show_train": False},
        instance_index=0,
    ),
    SourceSpec(
        "merging_mixture",
        "Merging two-mode mixture",
        make_merging_mixture_instance,
        plot_paper_case,
        {},
        {"show_train": False},
        instance_index=0,
    ),
]


def main(
    seed: int,
    num_instances: int,
    index_overrides: dict[str, int],
    *,
    algorithm: str = "replica_gibbs",
) -> None:
    n = len(_SOURCES)
    fig = plt.figure(figsize=(6.5 * n, 8.5))
    gs = fig.add_gridspec(2, n, top=0.78, bottom=0.08, hspace=0.4, wspace=0.08)
    gp_axes = [fig.add_subplot(gs[0, j]) for j in range(n)]
    pro_axes = [fig.add_subplot(gs[1, j]) for j in range(n)]

    plot_fit_grid(
        fig,
        gp_axes,
        pro_axes,
        seed,
        num_instances,
        index_overrides,
        algorithm=algorithm,
        sources=_SOURCES,
    )

    fig.legend(
        handles=TOP_LEGEND_HANDLES,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        fontsize=13,
        frameon=False,
    )

    out_path = FIGURES_DIR / "paper_examples.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-instances", type=int, default=1)
    parser.add_argument(
        "--algorithm",
        default="replica_gibbs",
        choices=["ula", "replica_gibbs"],
        help="PRO sampler to use for the PrO-GP fits (default: replica_gibbs).",
    )
    for _spec in _SOURCES:
        parser.add_argument(
            f"--{_spec.name.replace('_', '-')}-index",
            type=int,
            default=None,
            help=f"Override this panel's instance index (default from _SOURCES: {_spec.instance_index}).",
        )
    args = parser.parse_args()

    index_overrides = {
        spec.name: getattr(args, f"{spec.name}_index")
        for spec in _SOURCES
        if getattr(args, f"{spec.name}_index") is not None
    }
    main(args.seed, args.num_instances, index_overrides, algorithm=args.algorithm)
