"""Synthetic Student-t-noise regression instances.

Observation noise is drawn from a heavy-tailed Student-t distribution, matched in scale
(std) to what a comparably-configured Gaussian-noise instance would have, applied over
the *whole* domain -- not confined to a local region, unlike `heteroskedastic.py`,
`multimodal.py`, and `regime_switch.py`. The latent function is a single, ordinary
stationary-GP draw.

Unlike those three (which all test *local* misspecification -- a region that's
misspecified against an otherwise-correct background), this tests a *global* one: is
the noise family itself (Gaussian vs. heavy-tailed) right, everywhere, all the time?
That's deliberate: both `standard_gp` and `pro_gp` fit a single global RBF kernel and
Gaussian-shaped predictive components, so a misspecification that's just "the wrong
scalar" (a lengthscale, a noise std) is an optimisation contest that plain MLE tends to
win outright (see `regime_switch.py`'s finding). What *should* favour PRO specifically
is that its predictive is a mixture of J Gaussians (`nlpd_pro`), and a mixture of
Gaussians is itself heavier-tailed than any single Gaussian component -- so PRO's
predictive is structurally better suited to occasional large residuals than a standard
GP's single-Gaussian predictive, regardless of how well either method's hyperparameters
are tuned. Applying it everywhere (rather than locally) means every single instance is a
clean, maximal test of that -- no averaging against a well-specified background diluting
the effect.

`noise_df` is the swept severity variable, but -- unlike every other dataset in this
package -- *smaller* is more severe here (fewer degrees of freedom = heavier tails):
this breaks the "larger swept value = worse" convention used elsewhere deliberately,
since reporting results by degrees of freedom is the standard, immediately-recognisable
convention for Student-t experiments and using some inverted "tail heaviness" quantity
instead would only obscure that.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class HeavyTailedCase(NamedTuple):
    """A synthetic regression instance whose observation noise is Student-t (heavy
    tailed) everywhere -- see the module docstring."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    noise_std: float
    noise_df: float
    ell: float
    alpha: float


def make_heavy_tailed_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    noise_df: float = 3.0,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> HeavyTailedCase:
    """Draw a synthetic Student-t-noise regression instance.

    A single mean-zero RBF-kernel GP prior, with lengthscale/variance (`ell`, `alpha`)
    randomised per instance, is drawn once for the latent function `y_truth` -- valid,
    unmodified, over the whole domain.

    Observation noise is Student-t with `noise_df` degrees of freedom (requires
    ``noise_df > 2`` for a finite variance), scaled to the std a comparably-configured
    Gaussian-noise instance would have (``noise_std_frac * sqrt(alpha)``) -- so only the
    tail *shape* is being tested, never the scale.

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
    # Student-t, rescaled from unit-variance-at-df (Var = df/(df-2) for df > 2) down to
    # noise_std -- matches what Gaussian noise at this std would look like in scale, so
    # only the tail shape is what's being tested.
    noise = noise_std * jnp.sqrt((noise_df - 2.0) / noise_df) * jr.t(noise_key, noise_df, (n,))
    y_obs = y_truth + noise

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return HeavyTailedCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        noise_df=float(noise_df),
        ell=float(ell),
        alpha=float(alpha),
    )
