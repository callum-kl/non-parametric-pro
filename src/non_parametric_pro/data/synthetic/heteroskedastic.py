from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split


class HeteroskedasticCase(NamedTuple):
    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_truth_test: jax.Array
    y_test: jax.Array
    sigma_train: jax.Array
    sigma_test: jax.Array
    noise_floor: float
    ell: float
    alpha: float
    center: float
    width: float
    variance_frac: float


def heteroskedastic_noise_std(
    x: jax.Array,
    *,
    center: float,
    width: float,
    excess_variance: float,
    noise_floor: float,
) -> jax.Array:
    """Noise SD of a single Gaussian bump of unit height at `center`, so the local
    variance rises from `noise_floor**2` to `noise_floor**2 + excess_variance`."""
    bump = jnp.exp(-0.5 * ((x - center) / width) ** 2)
    return jnp.sqrt(noise_floor**2 + excess_variance * bump**2)


def make_heteroskedastic_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    test_fraction: float = 0.3,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.15,
    variance_frac: float = 0.4,
    min_width: float = 0.3,
    max_width: float = 0.8,
    ell_range: tuple[float, float] = (0.5, 1.0),
    alpha_range: tuple[float, float] = (0.5, 2.0),
) -> HeteroskedasticCase:
    (
        x_key,
        ell_key,
        alpha_key,
        latent_key,
        center_key,
        width_key,
        split_key,
        noise_key,
    ) = jr.split(key, 8)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])

    kernel = gpx.kernels.RBF(lengthscale=ell, variance=alpha**2)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    center = jr.uniform(center_key, (), minval=x_min, maxval=x_max)
    width = jr.uniform(width_key, (), minval=min_width, maxval=max_width)
    noise_floor = noise_std_frac * alpha

    sigma = heteroskedastic_noise_std(
        x[:, 0],
        center=center,
        width=width,
        excess_variance=variance_frac * alpha,
        noise_floor=noise_floor,
    )
    y_obs = y_truth + sigma * jr.normal(noise_key, y_truth.shape)

    split = train_val_split(split_key, x, y_truth, val_fraction=test_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return HeteroskedasticCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        y_test=y_obs[test_idx].reshape(-1, 1),
        sigma_train=sigma[train_idx],
        sigma_test=sigma[test_idx],
        noise_floor=float(noise_floor),
        ell=float(ell),
        alpha=float(alpha),
        center=float(center),
        width=float(width),
        variance_frac=float(variance_frac),
    )


def plot_heteroskedastic_case(
    ax,
    data: HeteroskedasticCase,
    *,
    show_curve: bool = True,
    show_noise_bands: bool = True,
    show_train: bool = True,
) -> None:
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_truth_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_truth_sorted = x_full[order], y_truth_full[order]

    if show_curve:
        ax.plot(x_sorted, y_truth_sorted, color="C0", linewidth=1.5)
    if show_noise_bands:
        sigma_sorted = heteroskedastic_noise_std(
            x_sorted,
            center=data.center,
            width=data.width,
            excess_variance=data.variance_frac * data.alpha,
            noise_floor=data.noise_floor,
        )
        ax.fill_between(
            x_sorted,
            y_truth_sorted - 2 * sigma_sorted,
            y_truth_sorted + 2 * sigma_sorted,
            color="C0",
            alpha=0.15,
            linewidth=0,
        )
        ax.fill_between(
            x_sorted,
            y_truth_sorted - sigma_sorted,
            y_truth_sorted + sigma_sorted,
            color="C0",
            alpha=0.3,
            linewidth=0,
        )
    if show_train:
        ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    else:
        ax.scatter(data.x_test, data.y_test, color="black", s=8, zorder=3)
    ax.set_title(
        f"$\\sigma_0$={data.noise_floor:.2f}  $\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}",
        fontsize=9,
    )


HETEROSKEDASTIC_KWARGS = (
    "n",
    "test_fraction",
    "x_min",
    "x_max",
    "noise_std_frac",
    "variance_frac",
    "min_width",
    "max_width",
    "ell_range",
    "alpha_range",
)
