"""Synthetic hyperparameter-ambiguity regression instances.

Sparse, randomly-scattered training data drawn from a genuinely *short*-lengthscale,
*low*-noise RBF GP -- but sparse enough, relative to that lengthscale, that the same
few points are also plausibly explained by a much *longer*-lengthscale, *higher*-noise
GP. This is the classic small-N GP hyperparameter non-identifiability: with few,
spread-out points, "wiggly function, barely any noise" and "smooth function, moderate
noise absorbing the deviations" can have comparable marginal likelihood (see e.g.
Rasmussen & Williams, *Gaussian Processes for Machine Learning*, §5.4.3, on local
optima/ridges in GP hyperparameter learning). Densely-sampled test points -- drawn
independently of the training locations, over the same domain -- then reveal which
hypothesis was actually right: interpolating *between* training points is exactly
where the two explanations diverge most.

In practice, the dominant failure mode this dataset exposes isn't subtle hyperparameter
averaging so much as a much blunter one: `standard_gp`'s noise parameter is
*unconstrained* MLE, and with only `n_train` points there is a genuine (sometimes
global) likelihood optimum at noise->0, lengthscale->whatever threads the GP through
every training point almost exactly. That collapses predictive variance almost
everywhere, so any test point whose true value isn't near a training point -- which is
most of them, since `x_test` is independent and dense -- incurs a catastrophic NLPD
once the near-zero predictive std divides a nonzero residual. `pro_gp` structurally
can't fall into this hole: its adapted `sigma` is `SigmoidBounded(..., low=sigma_min)`,
a hard floor the optimiser cannot cross regardless of how much the sparse data would
otherwise reward collapsing it. That floor -- not deep ensemble/Bayesian machinery --
is why `pro_gp` avoids the catastrophic instances `standard_gp` occasionally produces
here; see `experiments/synthetic/synthetic.py`'s `fit_gp`/`fit_pro` for where each
method's noise parameter is (or isn't) bounded.

`n_train` is the swept severity variable, but *inverted* relative to every other
dataset's `param_name` (like `heavy_tailed.py`'s `noise_df`/`saturation.py`'s
`threshold_frac`): *smaller* `n_train` means *more* severe ambiguity (fewer points to
pin down the hyperparameters), converging toward the ordinary, unambiguous regime as
`n_train` grows.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey


class HyperparameterAmbiguousCase(NamedTuple):
    """A synthetic regression instance whose sparse training data is ambiguous
    between a short-lengthscale/low-noise and a long-lengthscale/high-noise
    explanation -- see the module docstring."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    noise_std: float
    ell: float
    alpha: float


def make_hyperparameter_ambiguous_instance(  # noqa: PLR0913
    key: PRNGKey,
    *,
    n_train: int = 15,
    n_test: int = 200,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.03,
    ell_range: tuple[float, float] = (0.12, 0.25),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> HyperparameterAmbiguousCase:
    """Draw a synthetic hyperparameter-ambiguity regression instance.

    The latent function is one draw from a *short*-lengthscale RBF GP (`ell`
    randomised within `ell_range`, deliberately short -- this is the true, "wiggly"
    generating process) with *low* homoscedastic noise (`noise_std_frac` a small
    fraction of the draw's own signal std). `n_train` points are then sampled sparsely
    at random over the domain -- sparse enough, relative to `ell`, that the resulting
    handful of points is also plausible under a much smoother, noisier explanation.
    `n_test` points are sampled *independently* and much more densely over the same
    domain, so most fall *between* training points, where the two competing
    explanations diverge most.
    """
    train_x_key, test_x_key, ell_key, alpha_key, latent_key, noise_key = jr.split(key, 6)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x_train = jr.uniform(train_x_key, (n_train, 1), minval=x_min, maxval=x_max)
    x_test = jr.uniform(test_x_key, (n_test, 1), minval=x_min, maxval=x_max)
    x_all = jnp.concatenate([x_train, x_test], axis=0)
    y_truth_all = prior.predict(x_all).sample(latent_key)

    noise_std = noise_std_frac * signal_std
    y_obs_all = y_truth_all + noise_std * jr.normal(noise_key, y_truth_all.shape)

    y_truth_train, y_truth_test = y_truth_all[:n_train], y_truth_all[n_train:]
    y_obs_train, y_obs_test = y_obs_all[:n_train], y_obs_all[n_train:]

    return HyperparameterAmbiguousCase(
        x_train=x_train,
        y_train=y_obs_train.reshape(-1, 1),
        y_truth_train=y_truth_train.reshape(-1, 1),
        x_test=x_test,
        y_test=y_obs_test.reshape(-1, 1),
        y_truth_test=y_truth_test.reshape(-1, 1),
        noise_std=float(noise_std),
        ell=float(ell),
        alpha=float(alpha),
    )
