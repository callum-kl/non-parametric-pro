"""Synthetic Huber-contamination regression instances.

Observation noise is a two-component Gaussian scale mixture -- the classical Huber
(1964) "gross error" model from robust statistics: with probability
`contamination_prob` ("epsilon" in the classical notation), a point's noise is drawn
from a much wider Gaussian (`outlier_scale` times the ordinary std) instead of the
usual one, applied independently at every point over the whole domain -- not confined
to a region, like `heavy_tailed.py`'s final form (gross errors aren't spatially
correlated with `x` in the classical model). The latent function is a single, ordinary
stationary-GP draw.

Unlike `heavy_tailed.py` (a single smooth Student-t family), this is a genuine discrete
two-component mixture -- and the true data-generating process is *literally* a mixture
of two Gaussians, which is structurally what PRO's own predictive is (`nlpd_pro`'s
J-particle Gaussian mixture). So this is close to the cleanest possible test of whether
that structural match actually buys PRO anything: if there's any dataset where a
mixture-of-Gaussians predictive should have a real edge over a single-Gaussian one,
it's this one.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class HuberCase(NamedTuple):
    """A synthetic regression instance whose observation noise is a two-component
    (ordinary / gross-error) Gaussian scale mixture, applied everywhere -- see the
    module docstring."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    is_outlier_train: jax.Array  # hidden contamination indicator (bool); NOT visible to a fitted model
    noise_std: float
    outlier_scale: float
    contamination_prob: float
    ell: float
    alpha: float


def make_huber_instance(  # noqa: PLR0913
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    contamination_prob: float = 0.1,
    outlier_scale: float = 10.0,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> HuberCase:
    """Draw a synthetic Huber-contamination regression instance.

    A single mean-zero RBF-kernel GP prior, with lengthscale/variance (`ell`, `alpha`)
    randomised per instance, is drawn once for the latent function -- an ordinary,
    unperturbed stationary process.

    Each observation's noise is drawn from ``N(0, noise_std^2)`` with probability
    ``1 - contamination_prob``, or from ``N(0, (outlier_scale * noise_std)^2)`` with
    probability `contamination_prob` -- i.e. a small fraction of points get noise
    `outlier_scale` times wider than the rest, independent of `x`. `contamination_prob`
    is the swept severity variable (larger = more frequent gross errors), consistent
    with the "larger swept value = worse" convention used by `amplitude_frac`/
    `mix_prob`/`roughness_factor`/`gap_frac` elsewhere in this package (unlike
    `heavy_tailed.py`'s inverted `noise_df` convention).

    `noise_std_frac` is a fraction of the draw's own signal std rather than an absolute
    unit, for the same reason `heteroskedastic.py` normalises its noise levels this way.
    """
    x_key, ell_key, alpha_key, latent_key, contam_key, noise_key, split_key = jr.split(key, 7)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    noise_std = noise_std_frac * signal_std
    is_outlier = jr.bernoulli(contam_key, contamination_prob, (n,))
    scale = jnp.where(is_outlier, outlier_scale * noise_std, noise_std)
    noise = scale * jr.normal(noise_key, (n,))
    y_obs = y_truth + noise

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return HuberCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        is_outlier_train=is_outlier[train_idx],
        noise_std=float(noise_std),
        outlier_scale=float(outlier_scale),
        contamination_prob=float(contamination_prob),
        ell=float(ell),
        alpha=float(alpha),
    )
