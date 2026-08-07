"""Synthetic extrapolation regression instances.

A "standard" setup -- one ordinary stationary-GP draw, homoscedastic Gaussian noise, no
other misspecification layered on top (see `heteroskedastic.py`/`multimodal.py`/etc.
for that) -- but with a non-standard split. Unlike `interpolation_gap.py` (a held-out
window strictly *inside* the training range, surrounded by data on both sides), the
held-out "region" here sits at one edge of the domain (randomised left/right per
instance): genuine extrapolation beyond the convex hull of the training `x` values, not
interpolation through a gap. The rest of the domain still gets an ordinary random
train/holdout split, exactly like `interpolation_gap.py`, so `region_nlpds["region"]` =
extrapolation performance and `region_nlpds["background"]` = ordinary generalization
performance, directly comparable via the same `aggregate_results.py` machinery.

Whether to expect PRO to win here: probably not, for the same reason as
`regime_switch.py`/`matern_mismatch.py`. A stationary kernel's covariance decays to
~0 with distance, so far from any training point *both* an exact GP and PRO's
particle-based approximation revert toward the same prior-implied predictive (mean 0,
variance `alpha`) -- the true predictive stays Gaussian everywhere, so there's no
non-Gaussian structure for PRO's mixture-of-particles predictive to exploit. If
anything, expect the exact GP to have a slight edge: its reversion to the prior is
exact and closed-form, while PRO approximates the same reversion through a finite
particle set and a Cholesky basis built only from training points, with no data out
there to correct any approximation error. The more interesting question this dataset
answers is a calibration one, not a "who wins" one: does each method's *uncertainty*
grow appropriately with distance from data, not just its point predictions.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class ExtrapolationCase(NamedTuple):
    """A synthetic regression instance whose test set is split into a held-out edge
    (genuine extrapolation) and an ordinary random holdout -- see the module
    docstring."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    noise_std: float
    ell: float
    alpha: float
    boundary: float  # x-value beyond which points are held out for extrapolation
    extrapolate_right: bool  # True: held-out edge is x > boundary; False: x < boundary


def extrapolation_region_mask(x: jax.Array, boundary: float, extrapolate_right: bool) -> jax.Array:
    """Boolean mask, True where `x` falls in the held-out extrapolation edge -- mirrors
    the other datasets' `*_region_mask` membership rule, for region-conditional
    evaluation (here: extrapolation vs. ordinary holdout)."""
    return jnp.where(extrapolate_right, x > boundary, x < boundary)


def make_extrapolation_instance(  # noqa: PLR0913
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    extrapolation_frac: float = 0.2,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> ExtrapolationCase:
    """Draw a synthetic extrapolation regression instance.

    A single mean-zero RBF-kernel GP prior, with lengthscale/variance (`ell`, `alpha`)
    randomised per instance, is drawn once for the latent function -- an ordinary,
    unperturbed stationary process; homoscedastic Gaussian noise throughout.

    `extrapolation_frac` sets the held-out edge's width as a fraction of the domain
    (``boundary = x_max - extrapolation_frac * (x_max - x_min)`` for a right-edge
    instance, mirrored for a left-edge one); which edge is held out is randomised per
    instance. `extrapolation_frac` is the swept severity variable: larger means more of
    the domain (and thus a greater maximum distance from any training point) is held
    out -- consistent with the "larger swept value = worse" convention used by
    `amplitude_frac`/`mix_prob`/`roughness_factor`/`gap_frac`/`contamination_prob`/
    `noise_skewness`/`matern_mix` elsewhere in this package.

    `test_fraction` applies only to the points *outside* the held-out edge (the
    "ordinary holdout" portion) -- points inside the edge are always held out,
    regardless of this fraction.

    `noise_std_frac` is a fraction of the draw's own signal std rather than an absolute
    unit, for the same reason `heteroskedastic.py` normalises its noise levels this way.
    """
    x_key, ell_key, alpha_key, latent_key, side_key, noise_key, split_key = jr.split(key, 7)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    noise_std = noise_std_frac * signal_std
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    extrapolate_right = jr.bernoulli(side_key, 0.5)
    domain_width = x_max - x_min
    boundary = jnp.where(
        extrapolate_right,
        x_max - extrapolation_frac * domain_width,
        x_min + extrapolation_frac * domain_width,
    )
    in_extrap = extrapolation_region_mask(x[:, 0], boundary, extrapolate_right)

    extrap_idx = jnp.where(in_extrap)[0]
    non_extrap_idx = jnp.where(~in_extrap)[0]

    # The non-extrapolation points get the usual random train/holdout split;
    # extrapolation-edge points are always held out. `train_val_split`'s indices are
    # into the non-extrapolation subset, so map them back to the full arrays.
    split = train_val_split(
        split_key, x[non_extrap_idx], y_truth[non_extrap_idx], val_fraction=test_fraction
    )
    train_idx = non_extrap_idx[split.train_idx]
    background_test_idx = non_extrap_idx[split.val_idx]
    test_idx = jnp.concatenate([extrap_idx, background_test_idx])

    return ExtrapolationCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        ell=float(ell),
        alpha=float(alpha),
        boundary=float(boundary),
        extrapolate_right=bool(extrapolate_right),
    )
