"""Synthetic regime-switching regression instances: a single, well-defined latent
function whose local smoothness (kernel lengthscale) changes within local regions,
while everywhere else it follows one globally-stationary process.

Unlike `multimodal.py` (a hidden binary latent variable causing genuine, irreducible
ambiguity about which of two branches produced a given point), this is a *single-valued*,
unambiguous function throughout -- there is no hidden switch, no bimodality, and any
method that gets the smoothness right could in principle fit this perfectly. The
misspecification is purely about *roughness*: a stationary-kernel GP (one global
lengthscale) cannot simultaneously represent a smooth background and a locally much
rougher patch, and will either oversmooth the rough region (missing real structure) or
undersmooth the background (hallucinating spurious wiggle there) trying to compromise.
This mirrors `heteroskedastic.py`'s "one global noise level can't fit both quiet and
noisy regions" story, but for roughness instead of noise -- a third, complementary
misspecification axis alongside heteroskedastic noise and hidden multimodality.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split

# Boxcar is deliberately not offered (see multimodal.py's identical note): it jumps
# from 1 to 0 at the region boundary, which would make the perturbation below --
# and therefore y_truth itself -- discontinuous right at the edge of the region.
_SHAPE_GAUSSIAN, _SHAPE_RAMP = 0, 1
_NUM_SHAPES = 2


class RegimeRegions(NamedTuple):
    """Randomly drawn parameters for `num_regions` local roughness-switch regions --
    see `sample_regime_regions`."""

    centers: jax.Array  # (num_regions,)
    widths: jax.Array  # (num_regions,)
    shapes: jax.Array  # (num_regions,) int in {0, 1}


class RegimeSwitchCase(NamedTuple):
    """A synthetic regression instance whose latent function is locally rougher
    (shorter lengthscale) inside `regions` and follows one smooth, stationary process
    everywhere else -- see the module docstring."""

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
    """Randomly draw `num_regions` roughness-switch regions."""
    center_key, width_key, shape_key = jr.split(key, 3)

    centers = jr.uniform(center_key, (num_regions,), minval=x_min, maxval=x_max)
    widths = jr.uniform(width_key, (num_regions,), minval=min_width, maxval=max_width)
    shapes = jr.randint(shape_key, (num_regions,), 0, _NUM_SHAPES)

    return RegimeRegions(centers=centers, widths=widths, shapes=shapes)


def _region_bump(x: jax.Array, center: float, width: float, shape: int) -> jax.Array:
    """Evaluate one region's bump at `x`, roughly unit height at the center, tapering
    to zero at the boundary."""
    gaussian = jnp.exp(-0.5 * ((x - center) / width) ** 2)
    ramp = jnp.clip(1.0 - jnp.abs(x - center) / width, 0.0, 1.0)
    return jnp.select([shape == _SHAPE_GAUSSIAN, shape == _SHAPE_RAMP], [gaussian, ramp])


def _region_bumps(x: jax.Array, regions: RegimeRegions) -> jax.Array:
    """(num_regions, len(x)) bump values."""
    return jax.vmap(lambda c, w, s: _region_bump(x, c, w, s))(
        regions.centers, regions.widths, regions.shapes
    )


def regime_switch_region_mask(x: jax.Array, regions: RegimeRegions) -> jax.Array:
    """Boolean mask, True where `x` falls within any region's span
    (``|x - center| <= width``) -- mirrors `heteroskedastic_region_mask`'s/
    `multimodal_region_mask`'s membership rule, for splitting held-out points into
    "region" (locally rough) vs "background" (smooth, stationary) subsets.
    """

    def in_one_region(center, width):
        return jnp.abs(x - center) <= width

    in_any_region = jax.vmap(in_one_region)(regions.centers, regions.widths)
    return jnp.any(in_any_region, axis=0)


def make_regime_switch_instance(  # noqa: PLR0913
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
    """Draw a synthetic regime-switching regression instance.

    A mean-zero RBF-kernel GP prior with lengthscale `ell` and variance `alpha`
    (both randomised per instance) is drawn once for the smooth background
    (`y_base`, valid over the whole domain). Independently, for each of `num_regions`
    local regions, a "perturbation" is drawn from an RBF prior with the *same*
    variance `alpha` but a shorter lengthscale ``ell_region = ell / roughness_factor``
    -- same marginal scale, just packed with more oscillation per unit `x`. Each
    perturbation is windowed by its region's bump (zero at the region's edges, full
    value at its center) before being added to the background:
    ``y_truth = y_base + sum_regions(window * perturbation)``. Because the window is
    exactly (or, for the Gaussian shape, in the limit) zero at the boundary, `y_truth`
    transitions continuously from smooth to rough and back -- no discontinuity, and no
    hidden switch or ambiguity: this is a single, well-defined function throughout
    (contrast `multimodal.py`, which uses the same windowing trick but *also* has a
    hidden Bernoulli choosing between the background and a region's alternate, giving
    genuine bimodality; here every point simply *is* `y_truth`).

    `roughness_factor` is the swept severity variable: 1.0 means the region isn't
    actually any rougher than the background (no real misspecification), and larger
    values mean a shorter, more sharply mismatched region lengthscale -- consistent
    with the "larger swept value = more severe" convention used by `amplitude_frac`
    (`heteroskedastic.py`) and `mix_prob` (`multimodal.py`).

    `noise_std_frac` is a fraction of the draw's own signal std (`sqrt(alpha)`) rather
    than an absolute unit, for the same reason `heteroskedastic.py` normalises its
    noise levels this way: since `alpha` varies per draw, a fixed absolute noise level
    would make some draws look barely noisy and others look like pure noise for reasons
    unrelated to the phenomenon being studied.
    """
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

    # One perturbation draw per region, from the short-lengthscale prior, windowed by
    # that region's own bump before being added to the background -- see the docstring.
    perturb_keys = jr.split(perturb_key, num_regions)
    perturbation_per_region = jnp.stack(
        [region_prior.predict(x).sample(k) for k in perturb_keys], axis=0
    )  # (num_regions, n)

    bumps = _region_bumps(x[:, 0], regions)  # (num_regions, n)
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
