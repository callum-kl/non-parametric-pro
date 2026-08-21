
from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class ExtrapolationCase(NamedTuple):

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_test: jax.Array
    noise_std: float
    ell: float
    alpha: float
    boundary: float
    extrapolate_right: bool


def extrapolation_region_mask(x: jax.Array, boundary: float, extrapolate_right: bool) -> jax.Array:
    return jnp.where(extrapolate_right, x > boundary, x < boundary)


def make_extrapolation_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    extrapolation_frac: float = 0.2,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> ExtrapolationCase:
    x_key, ell_key, alpha_key, latent_key, side_key, noise_key, split_key = jr.split(key, 7)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    signal_std = jnp.sqrt(alpha)

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    noise_std = noise_std_frac * signal_std
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    extrapolate_right = jr.bernoulli(side_key, 0.5)
    domain_width = x_max - x_min
    boundary = jnp.where(
        extrapolate_right,
        x_max - extrapolation_frac * domain_width,
        x_min + extrapolation_frac * domain_width,
    )
    in_extrap = extrapolation_region_mask(x[:, 0], boundary, extrapolate_right)

    extrap_idx = jnp.where(in_extrap)[0]
    non_extrap_idx = jnp.where(~in_extrap)[0]

    split = train_val_split(
        split_key, x[non_extrap_idx], y_truth[non_extrap_idx], val_fraction=test_fraction
    )
    train_idx = non_extrap_idx[split.train_idx]
    background_test_idx = non_extrap_idx[split.val_idx]
    test_idx = jnp.concatenate([extrap_idx, background_test_idx])

    return ExtrapolationCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_test=y_obs[test_idx].reshape(-1, 1),
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        ell=float(ell),
        alpha=float(alpha),
        boundary=float(boundary),
        extrapolate_right=bool(extrapolate_right),
    )


def plot_extrapolation_case(ax, data: ExtrapolationCase) -> None:
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    edge = float(x_sorted.max()) if data.extrapolate_right else float(x_sorted.min())
    ax.axvspan(
        min(data.boundary, edge), max(data.boundary, edge), color="C1", alpha=0.15, linewidth=0
    )
    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    in_extrap = extrapolation_region_mask(data.x_test[:, 0], data.boundary, data.extrapolate_right)
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.scatter(data.x_test[~in_extrap], data.y_test[~in_extrap], color="gray", s=8, zorder=3)
    ax.scatter(data.x_test[in_extrap], data.y_test[in_extrap], color="C1", s=8, zorder=3)
    ax.set_title(f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}", fontsize=9)


EXTRAPOLATION_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "extrapolation_frac", "ell_range", "alpha_range",
)
