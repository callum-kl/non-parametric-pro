"""
Synthetic heteroskedastic regression instances with randomised noise structure.

The latent function is one draw from a GP prior (via gpjax); the observation noise
std then varies with `x` according to a *randomly parameterised* profile -- a fixed
number of "noisy regions", each with its own location/width/amplitude/shape -- so that
different keys give genuinely different heteroskedasticity patterns, not just
different noisy draws of one fixed pattern.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split

_SHAPE_GAUSSIAN, _SHAPE_BOXCAR, _SHAPE_RAMP = 0, 1, 2
_NUM_SHAPES = 3


class NoiseRegions(NamedTuple):
    """
    Randomly drawn parameters for `num_regions` local noise-elevation regions --
    see `sample_noise_regions`/`heteroskedastic_noise_std`.
    """

    centers: jax.Array  # (num_regions,)
    widths: jax.Array  # (num_regions,)
    amplitudes: jax.Array  # (num_regions,)
    shapes: jax.Array  # (num_regions,) int in {0, 1, 2}


class HeteroskedasticCase(NamedTuple):
    """
    A synthetic regression instance with randomised input-dependent observation
    noise, built from a GP-prior latent function.
    """

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
    amplitude: float = 0.3,
    num_regions: int = 3,
) -> NoiseRegions:
    """
    Randomly draw `num_regions` noise-elevation regions.

    Each region gets an independent shape family (Gaussian bump / boxcar / linear
    ramp): that's what gives "different types" of heteroskedasticity rather than one
    fixed functional form repeated at random places. `amplitude` is fixed (not
    sampled) across regions and draws -- it's meant to be swept directly as an
    experiment variable rather than adding its own randomised range on top.
    """
    center_key, width_key, shape_key = jr.split(key, 3)

    centers = jr.uniform(center_key, (num_regions,), minval=x_min, maxval=x_max)
    widths = jr.uniform(width_key, (num_regions,), minval=min_width, maxval=max_width)
    amplitudes = jnp.full((num_regions,), amplitude)
    shapes = jr.randint(shape_key, (num_regions,), 0, _NUM_SHAPES)

    return NoiseRegions(centers=centers, widths=widths, amplitudes=amplitudes, shapes=shapes)


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
    """
    Evaluate the noise standard deviation implied by `regions` at each point in `x`.

    Regions combine additively in *variance* (independent noise sources add in
    variance, not std), not via e.g. a max/clip across regions -- so overlapping
    regions compound smoothly instead of needing an ad hoc tie-break.
    """

    def one_region(center, width, amplitude, shape):
        return (amplitude * _region_bump(x, center, width, shape)) ** 2

    per_region_variance = jax.vmap(one_region)(
        regions.centers, regions.widths, regions.amplitudes, regions.shapes
    )
    total_variance = noise_floor**2 + jnp.sum(per_region_variance, axis=0)
    return jnp.sqrt(total_variance)


def heteroskedastic_region_mask(x: jax.Array, regions: NoiseRegions) -> jax.Array:
    """Boolean mask, True where `x` falls within any noise region's span
    (``|x - center| <= width``).

    This is the same span each region's bump tapers to exactly zero at for the
    boxcar/ramp shapes, and roughly one std for the Gaussian bump -- a single
    consistent membership rule across shapes, with no extra threshold to tune.
    Used to split held-out points into "heteroskedastic-region" vs
    "background" subsets for region-conditional evaluation (e.g. checking
    whether a method pays a tax in well-specified regions in exchange for
    doing better in misspecified ones).
    """

    def in_one_region(center, width):
        return jnp.abs(x - center) <= width

    in_any_region = jax.vmap(in_one_region)(regions.centers, regions.widths)
    return jnp.any(in_any_region, axis=0)


def make_heteroskedastic_instance(  # noqa: PLR0913
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_floor_frac_range: tuple[float, float] = (0.2, 0.3),
    amplitude_frac: float = 0.4,
    min_width: float = 0.15,
    max_width: float = 0.5,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
    num_regions: int = 3,
) -> HeteroskedasticCase:
    """
    Draw a synthetic heteroskedastic regression instance.

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
    than because of the heteroskedasticity pattern itself. `amplitude_frac` is a single
    fixed value (not a sampled range) so it can be swept directly as an experiment
    variable without an extra layer of per-instance randomness obscuring the trend.
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
        amplitude=amplitude_frac * signal_std,
        num_regions=num_regions,
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


def plot_heteroskedastic_case(
    ax, data: HeteroskedasticCase, *, show_curve: bool = True, show_noise_bands: bool = True
) -> None:
    """Plot one HeteroskedasticCase: true curve, +-1sigma/+-2sigma noise bands (both
    evaluated on the full, sorted train+test pool for a smooth curve), and the noisy
    observed training points. `show_curve=False`/`show_noise_bands=False` skip the
    truth-curve line / true-variance bands respectively -- e.g. when overlaying a
    fitted model's own predictive band on the same axes, where the true curve/bands
    would otherwise clutter/compete with it."""
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_truth_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_truth_sorted = x_full[order], y_truth_full[order]

    if show_curve:
        ax.plot(x_sorted, y_truth_sorted, color="C0", linewidth=1.5)
    if show_noise_bands:
        sigma_sorted = heteroskedastic_noise_std(x_sorted, data.regions, noise_floor=data.noise_floor)
        ax.fill_between(
            x_sorted, y_truth_sorted - 2 * sigma_sorted, y_truth_sorted + 2 * sigma_sorted,
            color="C0", alpha=0.15, linewidth=0,
        )
        ax.fill_between(
            x_sorted, y_truth_sorted - sigma_sorted, y_truth_sorted + sigma_sorted,
            color="C0", alpha=0.3, linewidth=0,
        )
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.set_title(
        f"$\\sigma_0$={data.noise_floor:.2f}  $\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}",
        fontsize=9,
    )


# Allowed keys a `ds` config (experiments/synthetic/conf/ds/*.yaml) can set on top of
# `source` itself; only these are forwarded to `make_heteroskedastic_instance`, so a
# config can override any subset without a code change in synthetic.py.
HETEROSKEDASTIC_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_floor_frac_range",
    "amplitude_frac", "min_width", "max_width",
    "ell_range", "alpha_range", "num_regions",
)
