"""Synthetic multimodal regression instances with a hidden mode switch.

A single GP-prior draw ("background") governs the whole domain. Inside a fixed number
of local "mixture regions", the background is locally perturbed by a second, independent
draw *from the same prior* (same kernel hyperparameters), windowed so the perturbation
is exactly zero at the region's edges and grows toward its center -- i.e. the alternate
branch necessarily *touches* the background curve at the region boundary and diverges
from it only inside the region, giving a genuine bifurcating shape (two branches forking
from, and merging back into, a single curve) rather than an unrelated curve that merely
becomes locally visible.

A *hidden* binary indicator `z` -- never given to a fitted model -- then decides, point
by point, whether that point actually came from the background branch or the (locally
perturbed) alternate branch. Outside all regions `P(z=1|x) ~= 0` and the two branches
coincide anyway, so the data is unambiguously single-valued there; only within a region
does it visibly split into two strands (irreducible bimodality that no function of `x`
alone, however flexible, can resolve -- see `heteroskedastic.py`'s module docstring for
the analogous local-region design this mirrors).
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split

# Boxcar is deliberately not offered here (unlike heteroskedastic.py's three shapes):
# it jumps from 1 to 0 at the region boundary, which would make the perturbation below
# *discontinuous* right where the branches are supposed to touch -- a visible "teleport"
# rather than a fork. Gaussian/ramp both taper continuously to (exactly, for ramp; in the
# limit, for Gaussian) zero at the boundary.
_SHAPE_GAUSSIAN, _SHAPE_RAMP = 0, 1
_NUM_SHAPES = 2


class MixtureRegions(NamedTuple):
    """Randomly drawn parameters for `num_regions` local mixture-ambiguity regions --
    see `sample_mixture_regions`/`mixture_probability`."""

    centers: jax.Array  # (num_regions,)
    widths: jax.Array  # (num_regions,)
    shapes: jax.Array  # (num_regions,) int in {0, 1}


class MultimodalCase(NamedTuple):
    """A synthetic regression instance with a hidden binary mode switch between a
    shared background function and (per-region) alternate functions -- see the module
    docstring."""

    x_train: jax.Array
    y_train: jax.Array
    z_train: jax.Array  # hidden mode indicator (bool); NOT visible to a fitted model
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
    """Randomly draw `num_regions` mixture-ambiguity regions.

    No per-region amplitude here (unlike `heteroskedastic.py`'s `NoiseRegions`): peak
    ambiguity is a single fixed `mix_prob`, supplied externally by `mixture_probability`
    -- same "fixed, not sampled" reasoning as `amplitude_frac` there, so it can be swept
    directly as an experiment variable.
    """
    center_key, width_key, shape_key = jr.split(key, 3)

    centers = jr.uniform(center_key, (num_regions,), minval=x_min, maxval=x_max)
    widths = jr.uniform(width_key, (num_regions,), minval=min_width, maxval=max_width)
    shapes = jr.randint(shape_key, (num_regions,), 0, _NUM_SHAPES)

    return MixtureRegions(centers=centers, widths=widths, shapes=shapes)


def _region_bump(x: jax.Array, center: float, width: float, shape: int) -> jax.Array:
    """Evaluate one region's bump at `x`, roughly unit height at the center, tapering
    to zero at the boundary (see the module-level note on why boxcar is excluded)."""
    gaussian = jnp.exp(-0.5 * ((x - center) / width) ** 2)
    ramp = jnp.clip(1.0 - jnp.abs(x - center) / width, 0.0, 1.0)
    return jnp.select([shape == _SHAPE_GAUSSIAN, shape == _SHAPE_RAMP], [gaussian, ramp])


def _region_bumps(x: jax.Array, regions: MixtureRegions) -> jax.Array:
    """(num_regions, len(x)) bump values -- shared by `mixture_probability` and
    `make_multimodal_instance` (which windows each region's perturbation by it)."""
    return jax.vmap(lambda c, w, s: _region_bump(x, c, w, s))(
        regions.centers, regions.widths, regions.shapes
    )


def mixture_probability(x: jax.Array, regions: MixtureRegions, *, mix_prob: float) -> jax.Array:
    """Evaluate `P(z=1|x)` -- the probability of drawing from the (locally perturbed)
    alternate branch instead of the background -- implied by `regions` at each point.

    Regions combine via *max*, not sum (unlike `heteroskedastic_noise_std`'s additive
    variance): this is a probability, not a variance, so overlapping regions shouldn't
    push it past `mix_prob`. `mix_prob` is the peak ambiguity: 0 means never ambiguous
    (background everywhere), 0.5 means a true 50/50 coin flip at a region's center.
    """
    return mix_prob * jnp.max(_region_bumps(x, regions), axis=0)


def multimodal_region_mask(x: jax.Array, regions: MixtureRegions) -> jax.Array:
    """Boolean mask, True where `x` falls within any mixture region's span
    (``|x - center| <= width``) -- mirrors `heteroskedastic_region_mask`'s membership
    rule, for splitting held-out points into "region" (genuinely ambiguous) vs
    "background" (single, deterministic mode) subsets.
    """

    def in_one_region(center, width):
        return jnp.abs(x - center) <= width

    in_any_region = jax.vmap(in_one_region)(regions.centers, regions.widths)
    return jnp.any(in_any_region, axis=0)


def make_multimodal_instance(  # noqa: PLR0913
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
    """Draw a synthetic multimodal regression instance.

    One mean-zero RBF-kernel GP prior (via gpjax), with lengthscale/variance (`ell`,
    `alpha`) randomised per instance, is drawn from repeatedly with the *same* kernel
    hyperparameters throughout (unlike an earlier version of this function, which
    independently randomised a whole separate kernel per mode and so produced two
    curves that differed everywhere, not just locally):

    - once for the shared background `y_shared`, valid over the whole domain;
    - once per region for a "divergence" draw, which is *windowed* by that region's bump
      (zero at the region's edges, up to its full value at the center) and added to the
      background to get that region's alternate branch: ``y_alt = y_shared + window *
      divergence``. Because the window is exactly (or, for the Gaussian shape, in the
      limit) zero at the boundary, `y_alt` is forced to coincide with `y_shared` there --
      a real fork, not two unrelated curves that happen to overlap in `x`.

    The hidden mode `z ~ Bernoulli(mixture_probability(x))` picks, per point, whether
    that point's true value comes from the background branch or the (locally perturbed)
    alternate branch (`y_truth = where(z, y_alt, y_shared)`); `z` is returned only for
    diagnostics (e.g. plotting) and must never be given to a fitted model.

    `noise_std_frac` is a fraction of the draw's own signal std (`sqrt(alpha)`) rather
    than an absolute unit, for the same reason `heteroskedastic.py` normalises its noise
    levels this way: since `alpha` varies per draw, a fixed absolute noise level would
    make some draws look barely noisy and others look like pure noise for reasons
    unrelated to the phenomenon being studied.
    """
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

    # One divergence draw per region, from the *same* prior, windowed by that region's
    # own bump before being added to the background -- see the docstring above.
    divergence_keys = jr.split(divergence_key, num_regions)
    divergence_per_region = jnp.stack(
        [prior.predict(x).sample(k) for k in divergence_keys], axis=0
    )  # (num_regions, n)

    bumps = _region_bumps(x[:, 0], regions)  # (num_regions, n)
    y_alt = y_shared + jnp.sum(bumps * divergence_per_region, axis=0)

    p_alt = mix_prob * jnp.max(bumps, axis=0)
    z = jr.bernoulli(mode_key, p_alt)

    y_truth = jnp.where(z, y_alt, y_shared)

    noise_std = noise_std_frac * signal_std
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    # `train_val_split` only splits one (x, y) pair -- reuse its train/val indices to
    # split z/y_shared/y_alt/y_obs consistently rather than calling it repeatedly.
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


def plot_multimodal_case(ax, data: MultimodalCase) -> None:
    """Plot one MultimodalCase: the shared background curve (blue) and the alternate
    branch (orange) -- `y_alt` is constructed to already coincide with `y_shared`
    outside a region (see `make_multimodal_instance`), so plotting both over their full
    extent shows a genuine fork: the two curves visibly touch at each region's edges
    and diverge only toward its center, rather than needing to be cut off/hidden away
    from the region. Training points are colored by which branch actually generated
    them (recovering that split is exactly what a model *can't* do, since `z` is
    hidden -- this is a diagnostic only)."""
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_shared_full = jnp.concatenate(
        [data.y_truth_shared_train[:, 0], data.y_truth_shared_test[:, 0]]
    )
    y_alt_full = jnp.concatenate([data.y_truth_alt_train[:, 0], data.y_truth_alt_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted = x_full[order]
    y_shared_sorted = y_shared_full[order]
    y_alt_sorted = y_alt_full[order]

    ax.plot(x_sorted, y_shared_sorted, color="C0", linewidth=1.5)
    ax.plot(x_sorted, y_alt_sorted, color="C1", linewidth=1.5)

    z_train = data.z_train
    ax.scatter(data.x_train[~z_train], data.y_train[~z_train], color="C0", s=8, zorder=3)
    ax.scatter(data.x_train[z_train], data.y_train[z_train], color="C1", s=8, zorder=3)
    ax.set_title(f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}", fontsize=9)


# Allowed keys a `ds` config (experiments/synthetic/conf/ds/*.yaml) can set on top of
# `source` itself; only these are forwarded to `make_multimodal_instance`, so a
# config can override any subset without a code change in synthetic.py.
MULTIMODAL_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "mix_prob", "min_width", "max_width", "ell_range", "alpha_range", "num_regions",
)
