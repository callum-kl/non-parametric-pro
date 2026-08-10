"""Synthetic skewed-noise regression instances.

Observation noise is drawn from a (mean-centred, std-matched) Gamma distribution
instead of Gaussian, applied over the whole domain -- not confined to a region, like
`heavy_tailed.py`'s/`huber.py`'s final forms (skewness isn't spatially correlated with
`x` any more than heavy tails or contamination are). The latent function is a single,
ordinary stationary-GP draw.

This is the fourth "shape of the noise distribution" axis in this package, alongside
`heteroskedastic.py` (variance/2nd moment), `heavy_tailed.py` (kurtosis/4th moment), and
`huber.py` (a discrete mixture): `skewed.py` targets the 3rd moment specifically. A
Gaussian-likelihood model (standard GP or PRO) can't represent asymmetry around the mean
at all, no matter how well tuned -- so, like the other three, this tests whether PRO's
mixture-of-Gaussians predictive (`nlpd_pro`) can approximate an asymmetric shape better
than a single-Gaussian predictive, purely from having enough particles to place
asymmetric mass around the mean.

`noise_skewness` is the swept severity variable: near 0 recovers approximately Gaussian
noise, and larger values are more asymmetric -- consistent with the "larger swept value
= worse" convention used by `amplitude_frac`/`mix_prob`/`roughness_factor`/`gap_frac`/
`contamination_prob` elsewhere in this package.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class SkewedCase(NamedTuple):
    """A synthetic regression instance whose observation noise is asymmetric (skewed)
    everywhere -- see the module docstring."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    noise_std: float
    noise_skewness: float
    skew_sign: float  # +1.0 (right-skewed, long tail above) or -1.0 (left-skewed, below)
    ell: float
    alpha: float


def make_skewed_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    noise_skewness: float = 2.0,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> SkewedCase:
    """Draw a synthetic skewed-noise regression instance.

    A single mean-zero RBF-kernel GP prior, with lengthscale/variance (`ell`, `alpha`)
    randomised per instance, is drawn once for the latent function -- an ordinary,
    symmetric stationary process; only the *noise* is skewed.

    Noise is drawn from a Gamma distribution with shape ``k = 4 / noise_skewness^2`` (a
    Gamma's skewness is exactly ``2/sqrt(k)``, so this inverts that relationship to make
    `noise_skewness` directly settable) and scale chosen so its std matches
    ``noise_std_frac * sqrt(alpha)``, then re-centred to zero mean -- so only the shape
    differs from Gaussian noise at the same scale, never the scale itself. The sign of
    the skew (right- vs left-skewed) is randomised per instance, so different keys give
    genuinely different-looking asymmetry, not always the same direction.

    `noise_std_frac` is a fraction of the draw's own signal std rather than an absolute
    unit, for the same reason `heteroskedastic.py` normalises its noise levels this way.
    """
    x_key, ell_key, alpha_key, latent_key, sign_key, noise_key, split_key = jr.split(key, 7)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    noise_std = noise_std_frac * signal_std

    gamma_shape = 4.0 / (noise_skewness**2)
    gamma_scale = noise_std / jnp.sqrt(gamma_shape)
    # jr.gamma is the standard Gamma(a, scale=1); scale by gamma_scale for Gamma(a,
    # scale=gamma_scale), then re-centre (its mean is gamma_shape * gamma_scale) to
    # zero -- matching noise_std, mean-zero, only the shape differs from Gaussian.
    magnitude = jr.gamma(noise_key, gamma_shape, (n,)) * gamma_scale - gamma_shape * gamma_scale
    sign = jnp.where(jr.bernoulli(sign_key, 0.5), 1.0, -1.0)
    noise = sign * magnitude

    y_obs = y_truth + noise

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return SkewedCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        noise_skewness=float(noise_skewness),
        skew_sign=float(sign),
        ell=float(ell),
        alpha=float(alpha),
    )


def plot_skewed_case(ax, data: SkewedCase) -> None:
    """Plot one SkewedCase: the single latent truth curve and observed points, applied
    over the whole domain (no region -- skewness isn't spatially correlated with x any
    more than heavy tails or contamination are). Points more than 1.5 noise stds out on
    the *long-tail side* (using the instance's known `skew_sign`, not a symmetric
    threshold like `plot_heavy_tailed_case`'s) are highlighted -- the asymmetry should
    show up as most of the highlighted points sitting on one side of the curve, not
    scattered evenly above and below it."""
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    residual = data.y_train[:, 0] - data.y_truth_train[:, 0]
    is_long_tail = (data.skew_sign * residual) > 1.5 * data.noise_std
    ax.scatter(data.x_train[~is_long_tail], data.y_train[~is_long_tail], color="black", s=8, zorder=3)
    ax.scatter(data.x_train[is_long_tail], data.y_train[is_long_tail], color="C1", s=8, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  "
        f"skew={data.skew_sign * data.noise_skewness:+.1f}",
        fontsize=9,
    )


# Allowed keys a `ds` config (experiments/synthetic/conf/ds/*.yaml) can set on top of
# `source` itself; only these are forwarded to `make_skewed_instance`, so a
# config can override any subset without a code change in synthetic.py.
SKEWED_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "noise_skewness", "ell_range", "alpha_range",
)
