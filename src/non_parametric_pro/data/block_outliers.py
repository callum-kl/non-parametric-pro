"""Synthetic localized corrupted-block regression instances.

A contiguous block of *training* observations -- everything whose `x` falls inside one
of `regions` -- is shifted by a fixed additive offset away from the latent truth (plus
the usual observation noise on top), simulating a batch of bad readings from a
mis-calibrated sensor, a corrupted data-collection run, or similar: a *contiguous*
stretch of training data that's wrong in a coherent, biased way, not just noisier.

This is a third flavor of "outlier" alongside `huber.py`'s and `heavy_tailed.py`'s:
those are global and i.i.d. (independently, at every point, with some probability or
heavy-tailed density) -- there's no structure `x` could exploit to say "outliers are
more likely here". `heteroskedastic.py`/`regime_switch.py` do have spatial structure
(a region), but that structure is a property of the *whole domain*, equally present for
a training point or a test point that happens to land there. This dataset combines both
ideas: outliers are regional (a contiguous block, like the two other regional
datasets) *and* they are a purely training-time artifact (like a batch of readings a
sensor happened to be broken for, before it was fixed) -- so, deliberately, test points
inside the exact same block are **not** corrupted; only whether a point ended up in the
*training* split determines whether the offset applies, on top of whether its `x` falls
inside a region. This is the one dataset in this package where train and test aren't
exchangeable by construction (everywhere else, a point's generative process depends only
on `x`, never on which split it lands in).

The interesting question this poses: a standard GP's posterior mean gets dragged toward
a coherent block of corrupted training points (there is no larger family of nearby
observations to average it against, the way i.i.d. contamination gets averaged out) and
produces a visible spurious bump there, which then also biases predictions at genuinely
clean, nearby *test* points -- not just inside the block, but at its edges too. Whether
PRO's particle-based posterior resists this better, or is dragged just as badly, is what
`region_nlpds["region"]` (test points inside the block's span, all clean) vs.
`region_nlpds["background"]` lets you check directly.

`outlier_offset_frac` is the swept severity variable -- how far the block is shifted, as
a fraction of the draw's own signal std (`sqrt(alpha)`) -- larger means a more severely
"outlying" block, consistent with the "larger swept value = worse" convention used by
`amplitude_frac`/`mix_prob`/`roughness_factor`/`contamination_prob` elsewhere in this
package. The direction of the shift (up or down) is randomised per instance, like
`skewed.py`'s `skew_sign`.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class OutlierBlockRegions(NamedTuple):
    """Randomly drawn parameters for `num_regions` corrupted-block regions -- see
    `sample_outlier_block_regions`."""

    centers: jax.Array  # (num_regions,)
    widths: jax.Array  # (num_regions,)


def sample_outlier_block_regions(
    key: PRNGKey,
    *,
    x_min: float = -1.0,
    x_max: float = 1.0,
    min_width: float = 0.05,
    max_width: float = 0.3,
    num_regions: int = 1,
) -> OutlierBlockRegions:
    """Randomly draw `num_regions` corrupted-block regions."""
    center_key, width_key = jr.split(key)

    centers = jr.uniform(center_key, (num_regions,), minval=x_min, maxval=x_max)
    widths = jr.uniform(width_key, (num_regions,), minval=min_width, maxval=max_width)

    return OutlierBlockRegions(centers=centers, widths=widths)


def block_outlier_region_mask(x: jax.Array, regions: OutlierBlockRegions) -> jax.Array:
    """Boolean mask, True where `x` falls within any region's span (``|x - center| <=
    width``) -- a hard block boundary (unlike `heteroskedastic.py`'s/`regime_switch.py`'s
    tapered bumps), since the offset below is applied to the *observation*, not to
    `y_truth`, so there's no continuity requirement forcing a smooth edge. Used both to
    decide which points are eligible for corruption during generation, and (in
    `experiments/synthetic/synthetic.py`'s `_REGION_MASK_FNS`) to split held-out test
    points into "region" (near the block, but never itself corrupted) vs. "background"
    for region-conditional evaluation."""

    def in_one_region(center, width):
        return jnp.abs(x - center) <= width

    in_any_region = jax.vmap(in_one_region)(regions.centers, regions.widths)
    return jnp.any(in_any_region, axis=0)


class BlockOutlierCase(NamedTuple):
    """A synthetic regression instance with a contiguous block of corrupted *training*
    observations -- see the module docstring."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    is_outlier_train: jax.Array  # bool; True for train points inside a region (corrupted)
    noise_std: float
    outlier_offset_frac: float
    outlier_sign: float  # +1.0 (block shifted up) or -1.0 (shifted down)
    ell: float
    alpha: float
    regions: OutlierBlockRegions


def make_block_outlier_instance(  # noqa: PLR0913
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
    """Draw a synthetic localized corrupted-block regression instance.

    A single mean-zero RBF-kernel GP prior, with lengthscale/variance (`ell`, `alpha`)
    randomised per instance, is drawn once for the latent function -- an ordinary,
    unperturbed stationary process; `y_truth` itself is never touched by the
    corruption below.

    Every observation gets ordinary homoscedastic noise (``noise_std = noise_std_frac *
    sqrt(alpha)``). On top of that, training points whose `x` falls inside one of
    `regions` (see `block_outlier_region_mask`) additionally get a fixed offset
    ``outlier_sign * outlier_offset_frac * sqrt(alpha)`` -- the whole block shifted
    coherently in one direction, not independently per point. Test points keep ordinary
    noise everywhere, *including* inside the region's own span: the offset is gated on
    "is this a training point", not just "is this `x` inside the block" (see the module
    docstring for why).

    `noise_std_frac` is a fraction of the draw's own signal std rather than an absolute
    unit, for the same reason `heteroskedastic.py` normalises its noise levels this way.
    """
    (
        x_key, ell_key, alpha_key, latent_key, region_key, sign_key, noise_key, split_key,
    ) = jr.split(key, 8)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    regions = sample_outlier_block_regions(
        region_key, x_min=x_min, x_max=x_max, min_width=min_width, max_width=max_width,
        num_regions=num_regions,
    )

    noise_std = noise_std_frac * signal_std
    noise = noise_std * jr.normal(noise_key, y_truth.shape)

    outlier_sign = jnp.where(jr.bernoulli(sign_key, 0.5), 1.0, -1.0)
    offset = outlier_sign * outlier_offset_frac * signal_std

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    # Gate the offset on *both* "inside a region" and "ended up a training point" -- a
    # test point inside the block's span still gets ordinary noise only (see the module
    # docstring: train/test aren't exchangeable here, unlike every other dataset).
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
    ax, data: BlockOutlierCase, *,
    show_curve: bool = True, show_train: bool = True, color_by_outlier: bool = True,
) -> None:
    """Plot one BlockOutlierCase: the single latent truth curve, and observed points --
    training points by default (`show_train=True`), or (for comparing a fitted model's
    predictive band against genuinely held-out data rather than the data it was fit to)
    the always-uncorrupted test points instead (`show_train=False`). Corrupted training
    points are marked with a red X (`color_by_outlier=True`, the default) *regardless*
    of `show_train` -- they're the ground-truth explanation for whatever pull a fitted
    model shows near the block, a structurally different thing from "training data" in
    general, so hiding them under `show_train=False` would defeat this dataset's whole
    diagnostic point. `show_curve=False` skips the truth-curve line."""
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    if show_curve:
        ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    is_outlier = data.is_outlier_train
    if show_train:
        ax.scatter(data.x_train[~is_outlier], data.y_train[~is_outlier], color="black", s=8, zorder=3)
    else:
        ax.scatter(data.x_test, data.y_test, color="black", s=8, zorder=3)

    if color_by_outlier:
        ax.scatter(
            data.x_train[is_outlier], data.y_train[is_outlier],
            color="red", marker="x", s=40, linewidths=1.5, zorder=4,
        )
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  "
        f"offset={data.outlier_sign * data.outlier_offset_frac:+.1f}",
        fontsize=9,
    )


# Allowed keys a `ds` config (experiments/synthetic/conf/ds/*.yaml) can set on top of
# `source` itself; only these are forwarded to `make_block_outlier_instance`, so a
# config can override any subset without a code change in synthetic.py.
BLOCK_OUTLIERS_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "outlier_offset_frac", "min_width", "max_width", "ell_range", "alpha_range", "num_regions",
)
