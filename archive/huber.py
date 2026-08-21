
from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class HuberCase(NamedTuple):

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    is_outlier_train: jax.Array
    noise_std: float
    outlier_scale: float
    contamination_prob: float
    ell: float
    alpha: float


def make_huber_instance(
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


def plot_huber_case(ax, data: HuberCase) -> None:
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    is_outlier = data.is_outlier_train
    ax.scatter(data.x_train[~is_outlier], data.y_train[~is_outlier], color="black", s=8, zorder=3)
    ax.scatter(data.x_train[is_outlier], data.y_train[is_outlier], color="C1", s=8, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  $\\epsilon$={data.contamination_prob:.2f}",
        fontsize=9,
    )


HUBER_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "contamination_prob", "outlier_scale", "ell_range", "alpha_range",
)
