
from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split

KERNEL_TYPES = ("rbf", "matern12", "matern32", "matern52")

_KERNEL_CONSTRUCTORS = {
    "rbf": gpx.kernels.RBF,
    "matern12": gpx.kernels.Matern12,
    "matern32": gpx.kernels.Matern32,
    "matern52": gpx.kernels.Matern52,
}


def build_kernel(kernel_type: str, *, lengthscale, variance=None) -> gpx.kernels.AbstractKernel:
    try:
        kernel_cls = _KERNEL_CONSTRUCTORS[kernel_type]
    except KeyError:
        msg = f"Unknown kernel_type={kernel_type!r}; expected one of {KERNEL_TYPES}"
        raise ValueError(msg) from None
    if variance is None:
        return kernel_cls(lengthscale=lengthscale)
    return kernel_cls(lengthscale=lengthscale, variance=variance)


class WellSpecifiedCase(NamedTuple):

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_truth_test: jax.Array
    y_test: jax.Array
    noise_std: float
    ell: float
    alpha: float
    kernel_type: str
    n: int


def make_well_specified_instance(
    key: PRNGKey,
    *,
    n: int = 300,
    train_fraction: float = 0.2,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
    kernel_types: tuple[str, ...] = KERNEL_TYPES,
) -> WellSpecifiedCase:
    x_key, ell_key, alpha_key, kernel_key, latent_key, split_key, noise_key = jr.split(key, 7)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    kernel_idx = jr.randint(kernel_key, (), 0, len(kernel_types))
    kernel_type = kernel_types[int(kernel_idx)]

    kernel = build_kernel(kernel_type, lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    noise_std = noise_std_frac * jnp.sqrt(alpha)
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    split = train_val_split(split_key, x, y_truth, val_fraction=1.0 - train_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return WellSpecifiedCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        y_test=y_obs[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        ell=float(ell),
        alpha=float(alpha),
        kernel_type=kernel_type,
        n=n,
    )


def plot_well_specified_case(
    ax, data: WellSpecifiedCase, *, show_curve: bool = True, show_train: bool = True
) -> None:
    if show_train:
        ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    else:
        ax.scatter(data.x_test, data.y_test, color="black", s=8, zorder=3)
    if show_curve:
        x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
        y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
        order = jnp.argsort(x_full)
        x_sorted, y_sorted = x_full[order], y_full[order]
        ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)
    ax.set_title(
        f"{data.kernel_type}  $\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  "
        f"n={data.n} (train={data.x_train.shape[0]})",
        fontsize=9,
    )


WELL_SPECIFIED_KWARGS = (
    "n", "train_fraction", "x_min", "x_max", "noise_std_frac",
    "ell_range", "alpha_range", "kernel_types",
)
