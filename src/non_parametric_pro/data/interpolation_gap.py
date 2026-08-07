"""Synthetic "standard" GP regression instances whose test set is a single contiguous
gap in `x`, rather than a random scatter of held-out points -- for testing
*interpolation* specifically (predicting through a region with no nearby training
observations) as distinct from the "typical local density" generalization that a random
train/test split tests.

The data-generating process itself is deliberately plain: one ordinary stationary-GP
draw with homoscedastic Gaussian noise, no regional misspecification (see
`heteroskedastic.py`/`multimodal.py`/`regime_switch.py`/`heavy_tailed.py` for that) --
only *where the test points come from* varies here. Points inside a single window
(`gap_center` +/- `gap_width/2`) are always held out (the interpolation test); the
remaining points get the usual random train/test split, giving a second, "typical"
held-out set for comparison via the same region_nlpds machinery every other dataset in
this package uses (`region` = inside the gap, `background` = the ordinary random
holdout) -- e.g. to check how much worse interpolation-through-a-gap is than ordinary
generalization, and whether PRO's particle-based uncertainty handles the gap better than
a standard GP's analytic predictive.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class InterpolationCase(NamedTuple):
    """A synthetic regression instance whose test set is split into a contiguous
    interpolation gap and an ordinary random holdout -- see the module docstring."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    noise_std: float
    ell: float
    alpha: float
    gap_center: float
    gap_width: float


def interpolation_gap_mask(x: jax.Array, gap_center: float, gap_width: float) -> jax.Array:
    """Boolean mask, True where `x` falls inside the gap (``|x - gap_center| <=
    gap_width / 2``) -- mirrors the other datasets' `*_region_mask` membership rule, for
    region-conditional evaluation (here: interpolation-gap vs. ordinary holdout)."""
    return jnp.abs(x - gap_center) <= gap_width / 2


def make_interpolation_instance(  # noqa: PLR0913
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    gap_frac: float = 0.2,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> InterpolationCase:
    """Draw a synthetic interpolation-gap regression instance.

    A single mean-zero RBF-kernel GP prior, with lengthscale/variance (`ell`, `alpha`)
    randomised per instance, is drawn once for the latent function -- an ordinary,
    unperturbed stationary process; homoscedastic Gaussian noise throughout.

    `gap_frac` sets the gap's width as a fraction of the domain (``gap_width = gap_frac
    * (x_max - x_min)``); its center is drawn uniformly subject to the whole gap fitting
    inside ``[x_min, x_max]``, so there's always some training coverage on both sides --
    a genuine interpolation test, not extrapolation. `gap_frac` is the swept severity
    variable: larger means a wider gap, i.e. the gap's center is farther from the
    nearest training point, a harder interpolation task -- consistent with the "larger
    swept value = worse" convention used by `amplitude_frac`/`mix_prob`/
    `roughness_factor` elsewhere in this package.

    `test_fraction` applies only to the points *outside* the gap (the "ordinary holdout"
    portion) -- points inside the gap are always held out, regardless of this fraction.

    `noise_std_frac` is a fraction of the draw's own signal std rather than an absolute
    unit, for the same reason `heteroskedastic.py` normalises its noise levels this way.
    """
    x_key, ell_key, alpha_key, latent_key, gap_key, noise_key, split_key = jr.split(key, 7)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    noise_std = noise_std_frac * signal_std
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    gap_width = gap_frac * (x_max - x_min)
    gap_center = jr.uniform(
        gap_key, (), minval=x_min + gap_width / 2, maxval=x_max - gap_width / 2
    )
    in_gap = interpolation_gap_mask(x[:, 0], gap_center, gap_width)

    gap_idx = jnp.where(in_gap)[0]
    non_gap_idx = jnp.where(~in_gap)[0]

    # The non-gap points get the usual random train/holdout split; gap points are
    # always held out. `train_val_split`'s indices are into the non-gap subset, so map
    # them back to indices into the full `x`/`y_obs`/`y_truth` arrays.
    split = train_val_split(split_key, x[non_gap_idx], y_truth[non_gap_idx], val_fraction=test_fraction)
    train_idx = non_gap_idx[split.train_idx]
    background_test_idx = non_gap_idx[split.val_idx]
    test_idx = jnp.concatenate([gap_idx, background_test_idx])

    return InterpolationCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        ell=float(ell),
        alpha=float(alpha),
        gap_center=float(gap_center),
        gap_width=float(gap_width),
    )
