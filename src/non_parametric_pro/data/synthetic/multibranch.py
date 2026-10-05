from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class MultibranchCase(NamedTuple):
    x_train: jax.Array
    y_train: jax.Array
    branch_train: jax.Array
    y_branches_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    branch_test: jax.Array
    y_branches_test: jax.Array
    noise_std: float
    ell: float
    alpha: float
    num_branches: int


def make_multibranch_instance(
    key: PRNGKey,
    *,
    num_branches: int = 2,
    n: int = 200,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.15,
    ell_range: tuple[float, float] = (0.5, 1.0),
    alpha_range: tuple[float, float] = (0.5, 2.0),
    shared_width: float = 0.6,
    transition_width: float = 0.8,
) -> MultibranchCase:
    """
    `make_multimodal_instance` generalised to `num_branches` branches: a shared GP draw
    plus one independent GP divergence draw per branch, blended in away from a random
    center. Each observation comes from a uniformly chosen branch.
    """
    (
        x_key,
        ell_key,
        alpha_key,
        shared_key,
        div_key,
        center_key,
        branch_key,
        split_key,
        noise_key,
    ) = jr.split(key, 9)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha**2)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    latent = prior.predict(x)
    y_shared = latent.sample(shared_key)
    divergences = latent.sample(div_key, (num_branches,))

    center = jr.uniform(center_key, (), minval=x_min, maxval=x_max)
    dist = jnp.abs(x[:, 0] - center)
    branch_weight = jnp.clip((dist - shared_width) / transition_width, 0.0, 1.0)
    y_branches = y_shared[None, :] + branch_weight[None, :] * divergences

    branch = jr.randint(branch_key, (n,), 0, num_branches)
    y_truth = y_branches[branch, jnp.arange(n)]

    noise_std = noise_std_frac * alpha
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return MultibranchCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        branch_train=branch[train_idx],
        y_branches_train=y_branches[:, train_idx],
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        branch_test=branch[test_idx],
        y_branches_test=y_branches[:, test_idx],
        noise_std=float(noise_std),
        ell=float(ell),
        alpha=float(alpha),
        num_branches=num_branches,
    )


MULTIBRANCH_KWARGS = (
    "num_branches",
    "n",
    "test_fraction",
    "x_min",
    "x_max",
    "noise_std_frac",
    "ell_range",
    "alpha_range",
    "shared_width",
    "transition_width",
)
