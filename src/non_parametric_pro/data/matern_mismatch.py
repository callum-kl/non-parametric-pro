"""Synthetic Matern-kernel-mismatch regression instances.

The latent function is drawn from a *blended* kernel -- a convex combination of an RBF
component and a Matern-3/2 component, ``k = (1 - matern_mix) * k_RBF + matern_mix *
k_Matern32`` (both sharing the same lengthscale, so it's smoothness *class* that's
blending, not scale) -- while both `standard_gp` and `pro_gp` always fit a plain RBF
kernel, regardless. `matern_mix=0` recovers the well-specified case (pure RBF, no
mismatch); larger values mean more of the true process comes from Matern-3/2, which
(unlike RBF) is only once-differentiable -- rougher, more jagged sample paths that an
RBF kernel, with its built-in infinite smoothness, structurally cannot represent no
matter how its lengthscale/variance are tuned.

This is a different *class* of misspecification from every other dataset in this
package: `regime_switch.py` mismatches the lengthscale of the *same* kernel family (RBF
vs RBF, just the wrong scale, and only locally) -- an optimisation contest that plain
MLE tends to win, since both methods share the same, fully-flexible-enough model class
there. Here, the true smoothness *class* itself is wrong everywhere, which no amount of
RBF lengthscale/variance tuning can fix (RBF paths are always infinitely
differentiable; a Matern-3/2-blended path structurally isn't) -- the standard framing
of GP kernel misspecification in the spatial-statistics literature (e.g. Stein,
*Interpolation of Spatial Data*, on the consequences of an incorrect smoothness class).

`matern_mix` is the swept severity variable, following the "larger swept value =
worse" convention used by `amplitude_frac`/`mix_prob`/`roughness_factor`/`gap_frac`/
`contamination_prob`/`noise_skewness` elsewhere in this package.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class MaternMismatchCase(NamedTuple):
    """A synthetic regression instance whose latent function is drawn from an
    RBF/Matern-3/2 blended kernel, though any fitted method in this package always
    assumes plain RBF -- see the module docstring."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    noise_std: float
    matern_mix: float
    ell: float
    alpha: float


def make_matern_mismatch_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    matern_mix: float = 0.5,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> MaternMismatchCase:
    """Draw a synthetic Matern-kernel-mismatch regression instance.

    The latent function is one draw from ``k = (1 - matern_mix) * RBF(ell, alpha) +
    matern_mix * Matern32(ell, alpha)`` -- both components share the same
    lengthscale/variance (`ell`, `alpha`, randomised per instance), so only smoothness
    *class* blends with `matern_mix`, never scale (the total variance at any point is
    always exactly `alpha`, regardless of `matern_mix`). Homoscedastic Gaussian noise
    is added as usual.

    `noise_std_frac` is a fraction of the draw's own signal std rather than an absolute
    unit, for the same reason `heteroskedastic.py` normalises its noise levels this way.
    """
    x_key, ell_key, alpha_key, latent_key, noise_key, split_key = jr.split(key, 6)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.SumKernel(
        kernels=[
            gpx.kernels.RBF(lengthscale=ell, variance=(1.0 - matern_mix) * alpha),
            gpx.kernels.Matern32(lengthscale=ell, variance=matern_mix * alpha),
        ]
    )
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    noise_std = noise_std_frac * signal_std
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return MaternMismatchCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        matern_mix=float(matern_mix),
        ell=float(ell),
        alpha=float(alpha),
    )
