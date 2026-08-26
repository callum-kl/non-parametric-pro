from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class MultimodalCase(NamedTuple):
    x_train: jax.Array
    y_train: jax.Array
    z_train: jax.Array
    y_truth_a_train: jax.Array
    y_truth_b_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    z_test: jax.Array
    y_truth_a_test: jax.Array
    y_truth_b_test: jax.Array
    noise_std: float
    ell: float
    alpha: float


def make_multimodal_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    mix_prob: float = 0.5,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
    shared_width: float = 0.6,
    transition_width: float = 0.8,
    shared_center_frac_range: tuple[float, float] = (0.35, 0.65),
) -> MultimodalCase:
    """
    Two branches, `y_truth_a`/`y_truth_b`, built from a shared GP draw plus two
    independent GP "divergence" draws, blended in via a weight that is exactly 0
    within `shared_width` of a (randomly placed) center -- so the branches are
    exactly equal there -- and ramps up to 1 over the next `transition_width`, past
    which the branches vary fully independently of each other. Each observation is
    drawn from one branch or the other, chosen independently with probability
    `mix_prob`.
    """
    (
        x_key,
        ell_key,
        alpha_key,
        shared_key,
        a_key,
        b_key,
        center_key,
        mode_key,
        split_key,
        noise_key,
    ) = jr.split(key, 10)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)

    y_shared = prior.predict(x).sample(shared_key)
    y_div_a = prior.predict(x).sample(a_key)
    y_div_b = prior.predict(x).sample(b_key)

    center_frac = jr.uniform(
        center_key, (), minval=shared_center_frac_range[0], maxval=shared_center_frac_range[1]
    )
    center = x_min + center_frac * (x_max - x_min)
    dist = jnp.abs(x[:, 0] - center)
    branch_weight = jnp.clip((dist - shared_width) / transition_width, 0.0, 1.0)

    y_a = y_shared + branch_weight * y_div_a
    y_b = y_shared + branch_weight * y_div_b

    z = jr.bernoulli(mode_key, mix_prob, (n,))
    y_truth = jnp.where(z, y_b, y_a)

    noise_std = noise_std_frac * signal_std
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return MultimodalCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        z_train=z[train_idx],
        y_truth_a_train=y_a[train_idx].reshape(-1, 1),
        y_truth_b_train=y_b[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        z_test=z[test_idx],
        y_truth_a_test=y_a[test_idx].reshape(-1, 1),
        y_truth_b_test=y_b[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        ell=float(ell),
        alpha=float(alpha),
    )


def plot_multimodal_case(
    ax,
    data: MultimodalCase,
    *,
    show_curves: bool = True,
    color_by_branch: bool = True,
    show_train: bool = True,
) -> None:
    if show_curves:
        x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
        y_a_full = jnp.concatenate([data.y_truth_a_train[:, 0], data.y_truth_a_test[:, 0]])
        y_b_full = jnp.concatenate([data.y_truth_b_train[:, 0], data.y_truth_b_test[:, 0]])
        order = jnp.argsort(x_full)
        x_sorted = x_full[order]

        ax.plot(x_sorted, y_a_full[order], color="C0", linewidth=1.5)
        ax.plot(x_sorted, y_b_full[order], color="C1", linewidth=1.5)

    x_points, y_points, z_points = (
        (data.x_train, data.y_train, data.z_train)
        if show_train
        else (data.x_test, data.y_test, data.z_test)
    )
    if color_by_branch:
        ax.scatter(x_points[~z_points], y_points[~z_points], color="C0", s=8, zorder=3)
        ax.scatter(x_points[z_points], y_points[z_points], color="C1", s=8, zorder=3)
    else:
        ax.scatter(x_points, y_points, color="black", s=8, zorder=3)
    ax.set_title(f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}", fontsize=9)


MULTIMODAL_KWARGS = (
    "n",
    "test_fraction",
    "x_min",
    "x_max",
    "noise_std_frac",
    "mix_prob",
    "ell_range",
    "alpha_range",
    "shared_width",
    "transition_width",
    "shared_center_frac_range",
)
