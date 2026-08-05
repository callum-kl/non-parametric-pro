"""Synthetic heteroscedastic regression instances with randomised noise structure.

The latent function is one draw from a GP prior (via gpjax); the observation noise
std then varies with `x` according to a *randomly parameterised* profile -- number of
"noisy regions" (1 or 2), and each region's location/width/amplitude/shape -- so that
different keys give genuinely different heteroscedasticity patterns, not just
different noisy draws of one fixed pattern.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split

MAX_NOISE_REGIONS = 3
_SHAPE_GAUSSIAN, _SHAPE_BOXCAR, _SHAPE_RAMP = 0, 1, 2
_NUM_SHAPES = 3


class NoiseRegions(NamedTuple):
    """Randomly drawn parameters for up to ``MAX_NOISE_REGIONS`` local noise-elevation
    regions -- see `sample_noise_regions`/`heteroskedastic_noise_std`."""

    centers: jax.Array  # (max_regions,)
    widths: jax.Array  # (max_regions,)
    amplitudes: jax.Array  # (max_regions,)
    shapes: jax.Array  # (max_regions,) int in {0, 1, 2}
    active: jax.Array  # (max_regions,) bool -- 1 or 2 of these are True


class HeteroskedasticCase(NamedTuple):
    """A synthetic regression instance with randomised input-dependent observation
    noise, built from a GP-prior latent function."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_truth_test: jax.Array
    y_test: jax.Array
    sigma_train: jax.Array
    sigma_test: jax.Array
    noise_floor: float
    ell: float
    alpha: float
    regions: NoiseRegions


def sample_noise_regions(  # noqa: PLR0913
    key: PRNGKey,
    *,
    x_min: float = -1.0,
    x_max: float = 1.0,
    min_width: float = 0.05,
    max_width: float = 0.3,
    min_amplitude: float = 0.15,
    max_amplitude: float = 0.6,
    max_regions: int = MAX_NOISE_REGIONS,
) -> NoiseRegions:
    """Randomly draw 1 or 2 active noise-elevation regions.

    Region count is randomised but kept at a fixed array width (`max_regions`) via an
    `active` mask rather than a variable-length list -- JAX needs static shapes, and
    this keeps the whole thing `jit`/`vmap`-friendly. Each active region also gets an
    independent shape family (Gaussian bump / boxcar / linear ramp): that's what gives
    "different types" of heteroscedasticity rather than one fixed functional form
    repeated at random places.
    """
    count_key, center_key, width_key, amp_key, shape_key = jr.split(key, 5)

    n_active = jr.randint(count_key, (), 1, max_regions + 1)
    active = jnp.arange(max_regions) < n_active

    centers = jr.uniform(center_key, (max_regions,), minval=x_min, maxval=x_max)
    widths = jr.uniform(width_key, (max_regions,), minval=min_width, maxval=max_width)
    amplitudes = jr.uniform(amp_key, (max_regions,), minval=min_amplitude, maxval=max_amplitude)
    shapes = jr.randint(shape_key, (max_regions,), 0, _NUM_SHAPES)

    return NoiseRegions(
        centers=centers, widths=widths, amplitudes=amplitudes, shapes=shapes, active=active
    )


def _region_bump(x: jax.Array, center: float, width: float, shape: int) -> jax.Array:
    """Evaluate one region's noise bump at `x`, roughly unit height at the center."""
    gaussian = jnp.exp(-0.5 * ((x - center) / width) ** 2)
    boxcar = (jnp.abs(x - center) <= width).astype(x.dtype)
    ramp = jnp.clip(1.0 - jnp.abs(x - center) / width, 0.0, 1.0)
    return jnp.select(
        [shape == _SHAPE_GAUSSIAN, shape == _SHAPE_BOXCAR, shape == _SHAPE_RAMP],
        [gaussian, boxcar, ramp],
    )


def heteroskedastic_noise_std(x: jax.Array, regions: NoiseRegions, *, noise_floor: float) -> jax.Array:
    """Evaluate the noise standard deviation implied by `regions` at each point in `x`.

    Regions combine additively in *variance* (independent noise sources add in
    variance, not std), not via e.g. a max/clip across regions -- so overlapping
    regions compound smoothly instead of needing an ad hoc tie-break.
    """

    def one_region(center, width, amplitude, shape, is_active):
        return is_active * (amplitude * _region_bump(x, center, width, shape)) ** 2

    per_region_variance = jax.vmap(one_region)(
        regions.centers, regions.widths, regions.amplitudes, regions.shapes, regions.active
    )
    total_variance = noise_floor**2 + jnp.sum(per_region_variance, axis=0)
    return jnp.sqrt(total_variance)


def make_heteroskedastic_instance(  # noqa: PLR0913
    key: PRNGKey,
    *,
    n: int = 200,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_floor_frac_range: tuple[float, float] = (0.05, 0.15),
    min_amplitude_frac: float = 0.2,
    max_amplitude_frac: float = 0.9,
    min_width: float = 0.15,
    max_width: float = 0.6,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> HeteroskedasticCase:
    """Draw a synthetic heteroscedastic regression instance.

    The latent function is one draw from a mean-zero RBF-kernel GP prior (via gpjax),
    with lengthscale (`ell`) and variance (`alpha`) themselves randomised per
    instance -- so different keys give genuinely different-looking latent functions,
    not just different noise on the same curve. Train/test are a random split of one
    pool of `n` points (rather than separate grids), so both see the same noise
    structure by construction.

    Noise-region amplitude/floor are specified as *fractions of the draw's own signal
    std* (`sqrt(alpha)`) rather than absolute units: since `alpha` itself varies per
    draw, fixed absolute noise levels would make some draws look barely noisy and
    others look like pure noise purely because of which `alpha` got sampled, rather
    than because of the heteroscedasticity pattern itself.
    """
    x_key, ell_key, alpha_key, latent_key, region_key, floor_key, split_key, noise_key = jr.split(
        key, 8
    )

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    regions = sample_noise_regions(
        region_key,
        x_min=x_min,
        x_max=x_max,
        min_width=min_width,
        max_width=max_width,
        min_amplitude=min_amplitude_frac * signal_std,
        max_amplitude=max_amplitude_frac * signal_std,
    )
    noise_floor = (
        jr.uniform(floor_key, (), minval=noise_floor_frac_range[0], maxval=noise_floor_frac_range[1])
        * signal_std
    )

    sigma = heteroskedastic_noise_std(x[:, 0], regions, noise_floor=noise_floor)
    y_obs = y_truth + sigma * jr.normal(noise_key, y_truth.shape)

    # `train_val_split` only splits one (x, y) pair -- reuse its train/val indices to
    # split y_truth/y_obs/sigma consistently rather than calling it three times.
    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return HeteroskedasticCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        y_test=y_obs[test_idx].reshape(-1, 1),
        sigma_train=sigma[train_idx],
        sigma_test=sigma[test_idx],
        noise_floor=float(noise_floor),
        ell=float(ell),
        alpha=float(alpha),
        regions=regions,
    )
