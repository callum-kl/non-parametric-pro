
from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split

_SHAPE_GAUSSIAN, _SHAPE_RAMP = 0, 1
_NUM_SHAPES = 2


class RegimeRegions(NamedTuple):

    centers: jax.Array
    widths: jax.Array
    shapes: jax.Array


class RegimeSwitchCase(NamedTuple):

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    noise_std: float
    ell: float
    alpha: float
    ell_region: float
    regions: RegimeRegions


def sample_regime_regions(
    key: PRNGKey,
    *,
    x_min: float = -1.0,
    x_max: float = 1.0,
    min_width: float = 0.05,
    max_width: float = 0.3,
    num_regions: int = 1,
) -> RegimeRegions:
    center_key, width_key, shape_key = jr.split(key, 3)

    centers = jr.uniform(center_key, (num_regions,), minval=x_min, maxval=x_max)
    widths = jr.uniform(width_key, (num_regions,), minval=min_width, maxval=max_width)
    shapes = jr.randint(shape_key, (num_regions,), 0, _NUM_SHAPES)

    return RegimeRegions(centers=centers, widths=widths, shapes=shapes)


def _region_bump(x: jax.Array, center: float, width: float, shape: int) -> jax.Array:
    gaussian = jnp.exp(-0.5 * ((x - center) / width) ** 2)
    ramp = jnp.clip(1.0 - jnp.abs(x - center) / width, 0.0, 1.0)
    return jnp.select([shape == _SHAPE_GAUSSIAN, shape == _SHAPE_RAMP], [gaussian, ramp])


def _region_bumps(x: jax.Array, regions: RegimeRegions) -> jax.Array:
    return jax.vmap(lambda c, w, s: _region_bump(x, c, w, s))(
        regions.centers, regions.widths, regions.shapes
    )


def regime_switch_region_mask(x: jax.Array, regions: RegimeRegions) -> jax.Array:

    def in_one_region(center, width):
        return jnp.abs(x - center) <= width

    in_any_region = jax.vmap(in_one_region)(regions.centers, regions.widths)
    return jnp.any(in_any_region, axis=0)


def make_regime_switch_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    roughness_factor: float = 3.0,
    min_width: float = 0.15,
    max_width: float = 0.5,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
    num_regions: int = 1,
) -> RegimeSwitchCase:
    (
        x_key,
        ell_key,
        alpha_key,
        base_key,
        region_key,
        perturb_key,
        noise_key,
        split_key,
    ) = jr.split(key, 8)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)
    ell_region = ell / roughness_factor

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    region_kernel = gpx.kernels.RBF(lengthscale=ell_region, variance=alpha)
    region_prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=region_kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_base = prior.predict(x).sample(base_key)

    regions = sample_regime_regions(
        region_key,
        x_min=x_min,
        x_max=x_max,
        min_width=min_width,
        max_width=max_width,
        num_regions=num_regions,
    )

    perturb_keys = jr.split(perturb_key, num_regions)
    perturbation_per_region = jnp.stack(
        [region_prior.predict(x).sample(k) for k in perturb_keys], axis=0
    )

    bumps = _region_bumps(x[:, 0], regions)
    y_truth = y_base + jnp.sum(bumps * perturbation_per_region, axis=0)

    noise_std = noise_std_frac * signal_std
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return RegimeSwitchCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        ell=float(ell),
        alpha=float(alpha),
        ell_region=float(ell_region),
        regions=regions,
    )


def plot_regime_switch_case(ax, data: RegimeSwitchCase) -> None:
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    for center, width in zip(data.regions.centers, data.regions.widths, strict=True):
        ax.axvspan(float(center - width), float(center + width), color="C1", alpha=0.15, linewidth=0)

    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\ell_r$={data.ell_region:.2f}  $\\alpha$={data.alpha:.2f}",
        fontsize=9,
    )


REGIME_SWITCH_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "roughness_factor", "min_width", "max_width", "ell_range", "alpha_range", "num_regions",
)
