"""Synthetic linear-kernel-mismatch regression instances.

The latent function is drawn from a *blended* kernel -- a combination of an RBF
component and a Linear component, ``k = (1 - linear_mix) * k_RBF + linear_mix *
k_Linear`` (``k_Linear(x, y) = variance * x * y``) -- while both `standard_gp` and
`pro_gp` always fit a plain RBF kernel, regardless. `linear_mix=0` recovers the
well-specified case (pure RBF, no mismatch); larger values mean more of the true
process comes from the Linear component.

Unlike `matern_mismatch.py`'s old RBF/Matern-3/2 blend (same *smoothness class*
mismatch, but constant marginal variance everywhere -- see git history), the Linear
component is non-stationary: ``k_Linear(x, x) = variance * x**2`` grows away from the
origin, so the *total* marginal variance grows toward the domain edges and is smallest
near ``x=0``, regardless of `linear_mix`. This is a structurally different kind of
misspecification from every other dataset in this package: an RBF kernel is stationary
by construction (constant marginal variance everywhere it's fit), so no amount of
lengthscale/variance tuning can reproduce this growth -- a common real-world pattern
where a process has a global linear trend/drift superimposed on local, roughly
stationary correlation (e.g. a physical quantity drifting further from a reference
value the further out you look, plus local noise/wiggles).

`linear_mix` is the swept severity variable, following the "larger swept value =
worse" convention used by `amplitude_frac`/`mix_prob`/`roughness_factor`/`gap_frac`/
`contamination_prob`/`noise_skewness` elsewhere in this package. The Linear component's
`variance` is scaled so that, at the domain edge farthest from the origin, its marginal
variance contribution equals ``linear_mix * alpha`` -- mirroring how `matern_mix`
apportioned the `alpha` budget between components, just evaluated at the edge rather
than everywhere (since it's no longer constant).
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class LinearMismatchCase(NamedTuple):
    """A synthetic regression instance whose latent function is drawn from an
    RBF/Linear blended kernel, though any fitted method in this package always
    assumes plain RBF -- see the module docstring."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    noise_std: float
    linear_mix: float
    ell: float
    alpha: float


def make_linear_mismatch_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    linear_mix: float = 0.5,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> LinearMismatchCase:
    """Draw a synthetic linear-kernel-mismatch regression instance.

    The latent function is one draw from ``k = (1 - linear_mix) * RBF(ell, alpha) +
    linear_mix * Linear(alpha / edge**2)``, where ``edge = max(|x_min|, |x_max|)`` --
    so the Linear component's marginal variance equals exactly ``linear_mix * alpha``
    at the domain edge farthest from the origin, and shrinks toward 0 at ``x=0``.
    `ell`/`alpha` are randomised per instance as usual. Homoscedastic Gaussian noise is
    added on top.

    `noise_std_frac` is a fraction of the draw's own signal std (``sqrt(alpha)``)
    rather than an absolute unit, for the same reason `heteroskedastic.py` normalises
    its noise levels this way.
    """
    x_key, ell_key, alpha_key, latent_key, noise_key, split_key = jr.split(key, 6)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    edge = max(abs(x_min), abs(x_max))
    linear_variance = linear_mix * alpha / edge**2

    kernel = gpx.kernels.SumKernel(
        kernels=[
            gpx.kernels.RBF(lengthscale=ell, variance=(1.0 - linear_mix) * alpha),
            gpx.kernels.Linear(variance=linear_variance),
        ]
    )
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    noise_std = noise_std_frac * signal_std
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return LinearMismatchCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        linear_mix=float(linear_mix),
        ell=float(ell),
        alpha=float(alpha),
    )


def plot_linear_mismatch_case(ax, data: LinearMismatchCase) -> None:
    """Plot one LinearMismatchCase: the single latent truth curve (drawn from the
    RBF/Linear blend -- visibly wider swings toward the domain edges as `linear_mix`
    grows, since the Linear component's marginal variance grows away from `x=0`) and
    the noisy observed points. No special coloring -- there's no hidden branch,
    censoring, or outlier label here, just a single well-defined function whose
    *non-stationarity* is what's being tested."""
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  linear={data.linear_mix:.2f}",
        fontsize=9,
    )


# Allowed keys a `ds` config (experiments/synthetic/conf/ds/*.yaml) can set on top of
# `source` itself; only these are forwarded to `make_linear_mismatch_instance`, so a
# config can override any subset without a code change in synthetic.py.
LINEAR_MISMATCH_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "linear_mix", "ell_range", "alpha_range",
)
