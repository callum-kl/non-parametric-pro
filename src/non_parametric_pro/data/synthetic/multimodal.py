from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split

_SHAPE_GAUSSIAN, _SHAPE_RAMP = 0, 1
_NUM_SHAPES = 2


class MixtureRegions(NamedTuple):
    centers: jax.Array
    widths: jax.Array
    shapes: jax.Array


class MultimodalCase(NamedTuple):
    x_train: jax.Array
    y_train: jax.Array
    z_train: jax.Array
    y_truth_shared_train: jax.Array
    y_truth_alt_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    z_test: jax.Array
    y_truth_shared_test: jax.Array
    y_truth_alt_test: jax.Array
    noise_std: float
    ell: float
    alpha: float
    regions: MixtureRegions


def sample_mixture_regions(
    key: PRNGKey,
    *,
    x_min: float = -1.0,
    x_max: float = 1.0,
    min_width: float = 0.05,
    max_width: float = 0.3,
    num_regions: int = 1,
) -> MixtureRegions:
    center_key, width_key, shape_key = jr.split(key, 3)

    centers = jr.uniform(center_key, (num_regions,), minval=x_min, maxval=x_max)
    widths = jr.uniform(width_key, (num_regions,), minval=min_width, maxval=max_width)
    shapes = jr.randint(shape_key, (num_regions,), 0, _NUM_SHAPES)

    return MixtureRegions(centers=centers, widths=widths, shapes=shapes)


def _region_bump(x: jax.Array, center: float, width: float, shape: int) -> jax.Array:
    gaussian = jnp.exp(-0.5 * ((x - center) / width) ** 2)
    ramp = jnp.clip(1.0 - jnp.abs(x - center) / width, 0.0, 1.0)
    return jnp.select(
        [shape == _SHAPE_GAUSSIAN, shape == _SHAPE_RAMP], [gaussian, ramp]
    )


def _region_bumps(x: jax.Array, regions: MixtureRegions) -> jax.Array:
    return jax.vmap(lambda c, w, s: _region_bump(x, c, w, s))(
        regions.centers, regions.widths, regions.shapes
    )


def mixture_probability(
    x: jax.Array, regions: MixtureRegions, *, mix_prob: float
) -> jax.Array:
    return mix_prob * jnp.max(_region_bumps(x, regions), axis=0)


def multimodal_region_mask(x: jax.Array, regions: MixtureRegions) -> jax.Array:

    def in_one_region(center, width):
        return jnp.abs(x - center) <= width

    in_any_region = jax.vmap(in_one_region)(regions.centers, regions.widths)
    return jnp.any(in_any_region, axis=0)


def make_multimodal_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    mix_prob: float = 0.5,
    min_width: float = 0.15,
    max_width: float = 0.5,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
    num_regions: int = 1,
) -> MultimodalCase:
    (
        x_key,
        ell_key,
        alpha_key,
        shared_key,
        region_key,
        divergence_key,
        mode_key,
        split_key,
        noise_key,
    ) = jr.split(key, 9)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_shared = prior.predict(x).sample(shared_key)

    regions = sample_mixture_regions(
        region_key,
        x_min=x_min,
        x_max=x_max,
        min_width=min_width,
        max_width=max_width,
        num_regions=num_regions,
    )

    divergence_keys = jr.split(divergence_key, num_regions)
    divergence_per_region = jnp.stack(
        [prior.predict(x).sample(k) for k in divergence_keys], axis=0
    )

    bumps = _region_bumps(x[:, 0], regions)
    y_alt = y_shared + jnp.sum(bumps * divergence_per_region, axis=0)

    p_alt = mix_prob * jnp.max(bumps, axis=0)
    z = jr.bernoulli(mode_key, p_alt)

    y_truth = jnp.where(z, y_alt, y_shared)

    noise_std = noise_std_frac * signal_std
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return MultimodalCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        z_train=z[train_idx],
        y_truth_shared_train=y_shared[train_idx].reshape(-1, 1),
        y_truth_alt_train=y_alt[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        z_test=z[test_idx],
        y_truth_shared_test=y_shared[test_idx].reshape(-1, 1),
        y_truth_alt_test=y_alt[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        ell=float(ell),
        alpha=float(alpha),
        regions=regions,
    )


def plot_multimodal_case(
    ax,
    data: MultimodalCase,
    *,
    show_curves: bool = True,
    color_by_branch: bool = True,
    show_train: bool = True,
) -> None:
    if show_curves:
        x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
        y_shared_full = jnp.concatenate(
            [data.y_truth_shared_train[:, 0], data.y_truth_shared_test[:, 0]]
        )
        y_alt_full = jnp.concatenate(
            [data.y_truth_alt_train[:, 0], data.y_truth_alt_test[:, 0]]
        )
        order = jnp.argsort(x_full)
        x_sorted = x_full[order]
        y_shared_sorted = y_shared_full[order]
        y_alt_sorted = y_alt_full[order]

        ax.plot(x_sorted, y_shared_sorted, color="C0", linewidth=1.5)
        ax.plot(x_sorted, y_alt_sorted, color="C1", linewidth=1.5)

    x_points, y_points, z_points = (
        (data.x_train, data.y_train, data.z_train)
        if show_train
        else (data.x_test, data.y_test, data.z_test)
    )
    if color_by_branch:
        ax.scatter(x_points[~z_points], y_points[~z_points], color="C0", s=8, zorder=3)
        ax.scatter(x_points[z_points], y_points[z_points], color="C1", s=8, zorder=3)
    else:
        ax.scatter(x_points, y_points, color="black", s=8, zorder=3)
    ax.set_title(f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}", fontsize=9)


MULTIMODAL_KWARGS = (
    "n",
    "test_fraction",
    "x_min",
    "x_max",
    "noise_std_frac",
    "mix_prob",
    "min_width",
    "max_width",
    "ell_range",
    "alpha_range",
    "num_regions",
)
