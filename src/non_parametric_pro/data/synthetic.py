"""Synthetic regression datasets for benchmarking (ported from Julia)."""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

_OUTLIER_INDICES = (6, 16, 27, 30, 3, 11, 22, 33)
_OUTLIER_SHIFTS = (2.8, -2.4, 2.5, -2.8, 2.1, -2.2, 2.4, -2.3)


class TrainValSplit(NamedTuple):
    """A random train/validation split of ``(x, y)``."""

    x_train: jax.Array
    y_train: jax.Array
    x_val: jax.Array
    y_val: jax.Array
    train_idx: jax.Array
    val_idx: jax.Array


class CleanRegressionCase(NamedTuple):
    """A noise-only (no contamination) synthetic GP regression instance."""

    x_train: jax.Array
    y_train: jax.Array
    x_test: jax.Array
    y_truth: jax.Array
    y_test: jax.Array
    sigma_model: float
    ell: float
    alpha: float


class ContaminatedCase(NamedTuple):
    """A synthetic regression instance with a fixed pattern of point outliers."""

    x_train: jax.Array
    y_train: jax.Array
    x_test: jax.Array
    y_truth: jax.Array
    y_obs_test: jax.Array
    sigma: float
    outlier_idx: jax.Array
    outlier_shifts: jax.Array


class HeteroskedasticCase(NamedTuple):
    """A synthetic regression instance with input-dependent observation noise."""

    x_train: jax.Array
    y_train: jax.Array
    x_test: jax.Array
    y_truth: jax.Array
    y_obs_test: jax.Array
    sigma_test: jax.Array
    sigma_model: jax.Array
    ell: float
    alpha: float
    tail_regions: list[tuple[float, float]]


class BlockOutlierCase(NamedTuple):
    """A synthetic regression instance with a contiguous block of outliers."""

    x_train: jax.Array
    y_train: jax.Array
    x_test: jax.Array
    y_truth: jax.Array
    y_test: jax.Array
    sigma_model: float
    ell: float
    alpha: float
    outlier_idx: jax.Array
    corruption_region: tuple[float, float]


class HuberSplit(NamedTuple):
    """A single (train or test) split of Huber-contaminated data."""

    x: jax.Array
    f: jax.Array
    y: jax.Array


class HuberData(NamedTuple):
    """Train/test splits with additive, asymmetric Huber-style contamination."""

    train: HuberSplit
    test: HuberSplit


class MixtureData(NamedTuple):
    """A synthetic regression instance from a two-component mixture of GP draws."""

    x_train: jax.Array
    y_train: jax.Array
    x_test: jax.Array
    y_truth: jax.Array
    y_obs_test: jax.Array
    sigma: float


def regression_truth(x: jax.Array) -> jax.Array:
    """Evaluate the latent regression function shared by the contamination scenarios."""
    return jnp.sin(2 * jnp.pi * x) + 0.3 * jnp.cos(4 * jnp.pi * x)


def _rbf_kernel(x1: jax.Array, x2: jax.Array, *, ell: float, alpha: float) -> jax.Array:
    """Evaluate a 1-D RBF gram matrix between ``x1`` and ``x2``."""
    diff = (x1.reshape(-1, 1) - x2.reshape(1, -1)) / ell
    return alpha * jnp.exp(-0.5 * diff**2)


def _col(x: jax.Array) -> jax.Array:
    """Reshape a 1-D array to a 2-D column vector, matching gpjax's ``Dataset``."""
    return x.reshape(-1, 1)


def make_clean_regression_case(
    key: PRNGKey,
    *,
    sigma: float = 0.12,
    ell_true: float = 0.18,
    alpha_true: float = 1.0,
    n: int = 35,
) -> CleanRegressionCase:
    """Draw a clean (uncontaminated) GP regression instance from an RBF prior."""
    latent_key, train_noise_key, test_noise_key = jr.split(key, 3)

    x_train = jnp.linspace(0.0, 1.0, n)
    x_test = jnp.linspace(0.0, 1.0, 300)
    x_all = jnp.concatenate([x_train, x_test])

    latent_cov = _rbf_kernel(
        x_all, x_all, ell=ell_true, alpha=alpha_true
    ) + 1e-8 * jnp.eye(x_all.shape[0])
    latent = jr.multivariate_normal(latent_key, jnp.zeros(x_all.shape[0]), latent_cov)

    y_train = latent[:n] + sigma * jr.normal(train_noise_key, (n,))
    y_truth = latent[n:]

    y_test = y_truth + sigma * jr.normal(test_noise_key, (x_test.shape[0],))

    return CleanRegressionCase(
        x_train=_col(x_train),
        y_train=_col(y_train),
        x_test=_col(x_test),
        y_truth=_col(y_truth),
        y_test=_col(y_test),
        sigma_model=sigma,
        ell=ell_true,
        alpha=alpha_true,
    )


def contaminated_outlier_pattern(n_outliers: int) -> tuple[jax.Array, jax.Array]:
    """Return the first ``n_outliers`` indices/shifts of a fixed outlier pattern."""
    if not 1 <= n_outliers <= len(_OUTLIER_INDICES):
        msg = f"n_outliers must be between 1 and {len(_OUTLIER_INDICES)}"
        raise ValueError(msg)
    idx = jnp.array(_OUTLIER_INDICES[:n_outliers]) - 1  # Julia indices are 1-based
    shifts = jnp.array(_OUTLIER_SHIFTS[:n_outliers])
    return idx, shifts


def make_contaminated_data(key: PRNGKey, *, n_outliers: int = 4) -> ContaminatedCase:
    """Draw a regression instance with a fixed pattern of training-point outliers."""
    train_key, test_key = jr.split(key)
    n, sigma = 35, 0.12

    x_train = jnp.linspace(0.0, 1.0, n)
    y_train = regression_truth(x_train) + sigma * jr.normal(train_key, (n,))

    outlier_idx, outlier_shifts = contaminated_outlier_pattern(n_outliers)
    y_train = y_train.at[outlier_idx].add(outlier_shifts)

    x_test = jnp.linspace(0.0, 1.0, 300)
    y_truth = regression_truth(x_test)
    y_obs_test = y_truth + sigma * jr.normal(test_key, (x_test.shape[0],))

    return ContaminatedCase(
        x_train=_col(x_train),
        y_train=_col(y_train),
        x_test=_col(x_test),
        y_truth=_col(y_truth),
        y_obs_test=_col(y_obs_test),
        sigma=sigma,
        outlier_idx=outlier_idx,
        outlier_shifts=outlier_shifts,
    )


def heteroskedastic_sigma_profile(x: jax.Array) -> jax.Array:
    """Evaluate the input-dependent noise standard deviation profile."""
    left_tail = jnp.exp(-0.5 * ((x - 0.10) / 0.09) ** 2)
    right_tail = jnp.exp(-0.5 * ((x - 0.90) / 0.09) ** 2)
    center_quiet = 1 - jnp.exp(-0.5 * ((x - 0.50) / 0.22) ** 2)
    return 0.025 + 0.55 * jnp.maximum(left_tail, right_tail) + 0.04 * center_quiet


def make_heteroskedastic_instance(key: PRNGKey) -> HeteroskedasticCase:
    """Draw a regression instance with input-dependent observation noise."""
    train_key, test_key = jr.split(key)
    n, ell, alpha = 72, 0.17, 1.0

    x_train = jnp.linspace(0.0, 1.0, n)
    sigma_train = heteroskedastic_sigma_profile(x_train)
    y_train = regression_truth(x_train) + sigma_train * jr.normal(train_key, (n,))

    x_test = jnp.linspace(0.0, 1.0, 420)
    y_truth = regression_truth(x_test)
    sigma_test = heteroskedastic_sigma_profile(x_test)

    y_obs_test = y_truth + sigma_test * jr.normal(test_key, (x_test.shape[0],))

    sigma_model = jnp.sqrt(jnp.mean(sigma_train**2))

    return HeteroskedasticCase(
        x_train=_col(x_train),
        y_train=_col(y_train),
        x_test=_col(x_test),
        y_truth=_col(y_truth),
        y_obs_test=_col(y_obs_test),
        sigma_test=sigma_test,
        sigma_model=sigma_model,
        ell=ell,
        alpha=alpha,
        tail_regions=[(0.0, 0.3), (0.7, 1.0)],
    )


def make_block_outlier_case(
    key: PRNGKey,
    *,
    amplitude: float = 2.4,
    region: tuple[float, float] = (0.28, 0.52),
    corrupt_fraction: float = 0.75,
) -> BlockOutlierCase:
    """Draw a regression instance with a contiguous block of corrupted points."""
    train_key, mask_key, test_key = jr.split(key, 3)
    n, sigma_model, ell, alpha = 45, 0.11, 0.17, 1.0

    x_train = jnp.linspace(0.0, 1.0, n)
    y_train = regression_truth(x_train) + sigma_model * jr.normal(train_key, (n,))

    block_idx = jnp.flatnonzero((x_train >= region[0]) & (x_train <= region[1]))
    keep_mask = jr.bernoulli(mask_key, corrupt_fraction, (block_idx.shape[0],))
    outlier_idx = block_idx[keep_mask]
    if outlier_idx.size == 0 and block_idx.size > 0:
        outlier_idx = block_idx[jnp.array([block_idx.shape[0] // 2])]

    y_train = y_train.at[outlier_idx].add(amplitude)

    x_test = jnp.linspace(0.0, 1.0, 250)
    y_truth = regression_truth(x_test)
    y_test = y_truth + sigma_model * jr.normal(test_key, (x_test.shape[0],))

    return BlockOutlierCase(
        x_train=_col(x_train),
        y_train=_col(y_train),
        x_test=_col(x_test),
        y_truth=_col(y_truth),
        y_test=_col(y_test),
        sigma_model=sigma_model,
        ell=ell,
        alpha=alpha,
        outlier_idx=outlier_idx,
        corruption_region=region,
    )


def make_huber_data(  # noqa: PLR0913
    key: PRNGKey,
    kernel: gpx.kernels.AbstractKernel,
    *,
    sigma: float,
    every_nth: int = 30,
    n_test: int = 1000,
    percent_miss: float = 0.15,
    shift: float = 1.0,
    scale: float = 2.0,
) -> HuberData:
    """Draw train/test GP function values with asymmetric Huber-style contamination."""
    f_key, train_key, test_key = jr.split(key, 3)

    x_test = jnp.linspace(0.0, 1.0, n_test)
    k_test = jnp.asarray(
        kernel.gram(x_test.reshape(-1, 1)).as_matrix()
    ) + 1e-6 * jnp.eye(n_test)
    f_test = jr.multivariate_normal(f_key, jnp.zeros(n_test), k_test)

    idx = jnp.arange(0, n_test, every_nth)
    x_train = x_test[idx]
    f_train = f_test[idx]

    def add_noise_and_outliers(key: PRNGKey, f: jax.Array) -> jax.Array:
        noise_key, perm_key, sign_key, magnitude_key = jr.split(key, 4)
        y = f + jr.normal(noise_key, f.shape) * sigma
        n_miss = int(jnp.floor(percent_miss * f.shape[0]))
        miss_idx = jr.permutation(perm_key, f.shape[0])[:n_miss]
        signs = jnp.where(jr.bernoulli(sign_key, 0.5, (n_miss,)), 1.0, -1.0)
        delta = signs * (jr.uniform(magnitude_key, (n_miss,)) * scale + shift)
        return y.at[miss_idx].add(y[miss_idx] + delta)

    y_train = add_noise_and_outliers(train_key, f_train)
    y_test = add_noise_and_outliers(test_key, f_test)

    return HuberData(
        train=HuberSplit(x=_col(x_train), f=_col(f_train), y=_col(y_train)),
        test=HuberSplit(x=_col(x_test), f=_col(f_test), y=_col(y_test)),
    )


def make_mixture_data(
    key: PRNGKey,
    kernel: gpx.kernels.AbstractKernel | None = None,
    *,
    n: int = 300,
    sigma: float = 0.1,
    keep_every: int = 10,
) -> MixtureData:
    """Draw a regression instance from a 50/50 mixture of two independent GP draws."""
    if kernel is None:
        kernel = gpx.kernels.Matern32(lengthscale=0.5, variance=0.5)

    f1_key, f2_key, omega_key, noise_key = jr.split(key, 4)
    x_test = jnp.linspace(0.0, 1.0, n)
    k = jnp.asarray(kernel.gram(x_test.reshape(-1, 1)).as_matrix()) + 1e-6 * jnp.eye(n)
    f_1 = jr.multivariate_normal(f1_key, jnp.zeros(n), k)
    f_2 = jr.multivariate_normal(f2_key, jnp.zeros(n), k)
    omega = jr.bernoulli(omega_key, 0.5, (n,))
    y_truth = jnp.where(omega, f_1, f_2)
    y_obs_test = y_truth + jr.normal(noise_key, (n,)) * sigma

    obs_i = jnp.arange(0, n, keep_every)
    x_train = x_test[obs_i]
    y_train = y_obs_test[obs_i]

    return MixtureData(
        x_train=_col(x_train),
        y_train=_col(y_train),
        x_test=_col(x_test),
        y_truth=_col(y_truth),
        y_obs_test=_col(y_obs_test),
        sigma=sigma,
    )
