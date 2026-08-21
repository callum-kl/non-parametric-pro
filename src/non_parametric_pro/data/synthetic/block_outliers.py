from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class OutlierBlockRegions(NamedTuple):
    centers: jax.Array
    widths: jax.Array


def sample_outlier_block_regions(
    key: PRNGKey,
    *,
    x_min: float = -1.0,
    x_max: float = 1.0,
    min_width: float = 0.05,
    max_width: float = 0.3,
    num_regions: int = 1,
) -> OutlierBlockRegions:
    center_key, width_key = jr.split(key)

    centers = jr.uniform(center_key, (num_regions,), minval=x_min, maxval=x_max)
    widths = jr.uniform(width_key, (num_regions,), minval=min_width, maxval=max_width)

    return OutlierBlockRegions(centers=centers, widths=widths)


def block_outlier_region_mask(x: jax.Array, regions: OutlierBlockRegions) -> jax.Array:

    def in_one_region(center, width):
        return jnp.abs(x - center) <= width

    in_any_region = jax.vmap(in_one_region)(regions.centers, regions.widths)
    return jnp.any(in_any_region, axis=0)


class BlockOutlierCase(NamedTuple):
    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    is_outlier_train: jax.Array
    noise_std: float
    outlier_offset_frac: float
    outlier_sign: float
    ell: float
    alpha: float
    regions: OutlierBlockRegions


def make_block_outlier_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    outlier_offset_frac: float = 2.0,
    min_width: float = 0.15,
    max_width: float = 0.4,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
    num_regions: int = 1,
) -> BlockOutlierCase:
    (
        x_key,
        ell_key,
        alpha_key,
        latent_key,
        region_key,
        sign_key,
        noise_key,
        split_key,
    ) = jr.split(key, 8)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    regions = sample_outlier_block_regions(
        region_key,
        x_min=x_min,
        x_max=x_max,
        min_width=min_width,
        max_width=max_width,
        num_regions=num_regions,
    )

    noise_std = noise_std_frac * signal_std
    noise = noise_std * jr.normal(noise_key, y_truth.shape)

    outlier_sign = jnp.where(jr.bernoulli(sign_key, 0.5), 1.0, -1.0)
    offset = outlier_sign * outlier_offset_frac * signal_std

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    in_block = block_outlier_region_mask(x[:, 0], regions)
    is_train = jnp.zeros((n,), dtype=bool).at[train_idx].set(True)
    is_outlier = in_block & is_train

    y_obs = y_truth + noise + jnp.where(is_outlier, offset, 0.0)

    return BlockOutlierCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        is_outlier_train=is_outlier[train_idx],
        noise_std=float(noise_std),
        outlier_offset_frac=float(outlier_offset_frac),
        outlier_sign=float(outlier_sign),
        ell=float(ell),
        alpha=float(alpha),
        regions=regions,
    )


def plot_block_outlier_case(
    ax,
    data: BlockOutlierCase,
    *,
    show_curve: bool = True,
    show_train: bool = True,
    color_by_outlier: bool = True,
    outlier_subsample_frac: float = 1.0,
) -> None:
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    if show_curve:
        ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    is_outlier = data.is_outlier_train
    if show_train:
        ax.scatter(
            data.x_train[~is_outlier],
            data.y_train[~is_outlier],
            color="black",
            s=8,
            zorder=3,
        )
    else:
        ax.scatter(data.x_test, data.y_test, color="black", s=8, zorder=3)

    if color_by_outlier:
        outlier_idx = jnp.where(is_outlier)[0]
        if outlier_subsample_frac < 1.0 and outlier_idx.size > 0:
            stride = max(1, round(1.0 / outlier_subsample_frac))
            outlier_idx = outlier_idx[::stride]
        ax.scatter(
            data.x_train[outlier_idx],
            data.y_train[outlier_idx],
            color="maroon",
            marker="x",
            s=40,
            linewidths=1.5,
            zorder=4,
        )
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  "
        f"offset={data.outlier_sign * data.outlier_offset_frac:+.1f}",
        fontsize=9,
    )


BLOCK_OUTLIERS_KWARGS = (
    "n",
    "test_fraction",
    "x_min",
    "x_max",
    "noise_std_frac",
    "outlier_offset_frac",
    "min_width",
    "max_width",
    "ell_range",
    "alpha_range",
    "num_regions",
)
