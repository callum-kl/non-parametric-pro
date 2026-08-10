"""Synthetic sensor-saturation regression instances.

Observations are censored (clipped) to a fixed range -- ``y_obs = clip(y_truth + noise,
-threshold, threshold)`` -- applied over the whole domain, like `heavy_tailed.py`'s/
`huber.py`'s/`skewed.py`'s final forms (saturation isn't spatially correlated with `x`
any more than those are). The latent function and noise are otherwise ordinary: a
single stationary-GP draw with homoscedastic Gaussian noise, clipped only *after* the
noise is added -- modelling a sensor whose output circuitry saturates at a fixed range
regardless of the (noisy) signal underneath, rather than the underlying process itself
being bounded.

This is the classical Tobit / censored-regression setup (Tobin, 1958): the true
predictive distribution at any `x` is neither symmetric-heavy-tailed, a discrete
mixture, nor skewed, but a *point mass at each threshold* (every value that would have
exceeded it collapses onto it) plus a continuous density strictly between them -- a
fifth, structurally distinct "shape of the observation distribution" alongside
`heteroskedastic.py` (variance), `heavy_tailed.py` (kurtosis), `huber.py` (discrete
mixture), and `skewed.py` (asymmetry). No single Gaussian, however well its
hyperparameters are tuned, can place a genuine point mass at a boundary; whether PRO's
finite mixture of Gaussians (`nlpd_pro`) approximates that any better -- e.g. by several
particles' predictions collapsing near the threshold -- is exactly what this tests.

`threshold_frac` is the swept severity variable, but -- like `heavy_tailed.py`'s
`noise_df` -- *smaller* is more severe here (a tighter clip range censors more of the
distribution): this breaks the "larger swept value = worse" convention used elsewhere
deliberately, for the same reason `noise_df` does -- reporting results by the actual
clip threshold is the natural, immediately-recognisable quantity, and inverting it into
some artificial "severity" score would only obscure that.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class SaturationCase(NamedTuple):
    """A synthetic regression instance whose observations are censored to a fixed
    range everywhere -- see the module docstring."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    is_censored_train: jax.Array  # hidden censoring indicator (bool); NOT visible to a fitted model
    noise_std: float
    threshold: float
    ell: float
    alpha: float


def make_saturation_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    threshold_frac: float = 1.5,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> SaturationCase:
    """Draw a synthetic sensor-saturation regression instance.

    A single mean-zero RBF-kernel GP prior, with lengthscale/variance (`ell`, `alpha`)
    randomised per instance, is drawn once for the latent function; homoscedastic
    Gaussian noise is added as usual, and *then* the whole observation is clipped to
    ``[-threshold, threshold]`` where ``threshold = threshold_frac * sqrt(alpha)``.

    `noise_std_frac` is a fraction of the draw's own signal std rather than an absolute
    unit, for the same reason `heteroskedastic.py` normalises its noise levels this way.
    """
    x_key, ell_key, alpha_key, latent_key, noise_key, split_key = jr.split(key, 6)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    noise_std = noise_std_frac * signal_std
    threshold = threshold_frac * signal_std

    y_raw = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)
    y_obs = jnp.clip(y_raw, -threshold, threshold)
    is_censored = jnp.abs(y_raw) > threshold

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return SaturationCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        is_censored_train=is_censored[train_idx],
        noise_std=float(noise_std),
        threshold=float(threshold),
        ell=float(ell),
        alpha=float(alpha),
    )


def plot_saturation_case(ax, data: SaturationCase) -> None:
    """Plot one SaturationCase: the single latent truth curve, dashed lines at the +/-
    saturation threshold, and observed points colored by the *true* (hidden) censoring
    indicator -- orange points sit exactly on one of the threshold lines (the classic
    "flat-lined" signature of a saturated sensor), black points are uncensored."""
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.axhline(data.threshold, color="C1", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axhline(-data.threshold, color="C1", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    is_censored = data.is_censored_train
    ax.scatter(data.x_train[~is_censored], data.y_train[~is_censored], color="black", s=8, zorder=3)
    ax.scatter(data.x_train[is_censored], data.y_train[is_censored], color="C1", s=8, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  thr={data.threshold:.2f}",
        fontsize=9,
    )


# Allowed keys a `ds` config (experiments/synthetic/conf/ds/*.yaml) can set on top of
# `source` itself; only these are forwarded to `make_saturation_instance`, so a
# config can override any subset without a code change in synthetic.py.
SATURATION_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "threshold_frac", "ell_range", "alpha_range",
)
