
from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class InterpolationCase(NamedTuple):

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
    return jnp.abs(x - gap_center) <= gap_width / 2


def make_interpolation_instance(
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


def plot_interpolation_case(ax, data: InterpolationCase) -> None:
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.axvspan(
        data.gap_center - data.gap_width / 2, data.gap_center + data.gap_width / 2,
        color="C1", alpha=0.15, linewidth=0,
    )
    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    in_gap = interpolation_gap_mask(data.x_test[:, 0], data.gap_center, data.gap_width)
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.scatter(data.x_test[~in_gap], data.y_test[~in_gap], color="gray", s=8, zorder=3)
    ax.scatter(data.x_test[in_gap], data.y_test[in_gap], color="C1", s=8, zorder=3)
    ax.set_title(f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}", fontsize=9)


INTERPOLATION_GAP_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "gap_frac", "ell_range", "alpha_range",
)
