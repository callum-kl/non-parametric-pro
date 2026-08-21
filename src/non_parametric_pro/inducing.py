from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import paramax
from blackjax.types import PRNGKey
from jax.scipy.linalg import solve_triangular


@runtime_checkable
class InducingBasis(Protocol):
    def inducing_cov(self, kernel: gpx.kernels.AbstractKernel) -> jax.Array:
        """K_zz: (M, M) covariance between inducing features."""
        ...

    def cross_cov(self, kernel: gpx.kernels.AbstractKernel, x: jax.Array) -> jax.Array:
        """K_zx: (M, N) cross-covariance between inducing features and x."""
        ...

    def output_dim(self) -> int:
        """Number of columns of the ``(N, output_dim)`` basis this produces."""
        ...


@dataclass
class PointInducingBasis:
    """
    Standard inducing point basis.
    """

    z: jax.Array

    def _z2d(self) -> jax.Array:
        return self.z.reshape(-1, 1) if self.z.ndim == 1 else self.z

    def inducing_cov(self, kernel: gpx.kernels.AbstractKernel) -> jax.Array:
        return kernel.gram(self._z2d()).as_matrix()

    def cross_cov(self, kernel: gpx.kernels.AbstractKernel, x: jax.Array) -> jax.Array:
        return kernel.cross_covariance(self._z2d(), x)

    def output_dim(self) -> int:
        return self.z.shape[0]


def kmeans_inducing_points(
    rng_key: PRNGKey,
    x_train: jax.Array,
    num_inducing: int,
    *,
    num_iters: int = 100,
) -> "PointInducingBasis":
    """
    Select inducing locations via k-means clustering of the training inputs.

    Parameters
    ----------
    rng_key
        JAX random key used to initialise centroids from a random subset of
        training points.
    x_train
        Training inputs, shape ``(N,)`` or ``(N, D)``.
    num_inducing
        Number of inducing points (clusters) ``M``.
    num_iters
        Number of Lloyd's algorithm iterations.  100 is usually sufficient.

    Returns
    -------
    A :class:`PointInducingBasis` whose ``z`` are the ``(M, D)`` centroids.

    """
    x = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train  # (N, D)

    init_idx = jr.choice(rng_key, x.shape[0], (num_inducing,), replace=False)
    centroids = x[init_idx]

    def step(centroids: jax.Array, _: None) -> tuple[jax.Array, None]:
        dists = jnp.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)  # (N, M)
        assignments = jnp.argmin(dists, axis=1)  # (N,)
        one_hot = jax.nn.one_hot(assignments, num_inducing)  # (N, M)
        counts = one_hot.sum(axis=0)  # (M,)
        new_centroids = one_hot.T @ x / jnp.maximum(counts[:, None], 1)
        return jnp.where(counts[:, None] > 0, new_centroids, centroids), None

    centroids, _ = jax.lax.scan(step, centroids, None, length=num_iters)
    return PointInducingBasis(centroids)


def sample_rff_frequencies(
    rng_key: PRNGKey,
    kernel: gpx.kernels.AbstractKernel,
    num_basis_fns: int,
    *,
    num_dims: int | None = None,
) -> jax.Array:
    """
    Sample raw (pre-lengthscale-scaling) Random Fourier Feature frequencies.
    """
    if num_dims is None:
        num_dims = kernel.n_dims
        if num_dims is None:
            msg = (
                "Could not infer the number of input dimensions from the "
                "kernel (its lengthscale is a scalar, so gpjax's n_dims is "
                "None). Pass num_dims explicitly."
            )
            raise ValueError(msg)
    return kernel.spectral_density.sample(
        key=rng_key, sample_shape=(num_basis_fns, num_dims)
    )


@dataclass
class RFFInducingBasis:
    """
    Random Fourier Features (Rahimi & Recht, 2008) inducing basis.
    """

    frequencies: jax.Array  # (M, D)

    def _features(self, kernel: gpx.kernels.AbstractKernel, x: jax.Array) -> jax.Array:
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        lengthscale = paramax.unwrap(kernel.lengthscale)
        z = x @ (self.frequencies / lengthscale).T  # (N, M)
        return jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)  # (N, 2M)

    def inducing_cov(self, kernel: gpx.kernels.AbstractKernel) -> jax.Array:
        variance = paramax.unwrap(kernel.variance)
        num_features = 2 * self.frequencies.shape[0]
        scaling = variance / self.frequencies.shape[0]
        return jnp.eye(num_features) / scaling

    def cross_cov(self, kernel: gpx.kernels.AbstractKernel, x: jax.Array) -> jax.Array:
        return self._features(kernel, x).T  # (2M, N)

    def output_dim(self) -> int:
        return 2 * self.frequencies.shape[0]

    def basis_matrix(
        self,
        kernel: gpx.kernels.AbstractKernel,
        x: jax.Array,
        jitter: float = 1e-6,
    ) -> jax.Array:
        variance = paramax.unwrap(kernel.variance)
        scaling = variance / self.frequencies.shape[0]
        cholesky_diag = jnp.sqrt(1.0 / scaling + jitter)
        return self._features(kernel, x) / cholesky_diag


def compute_inducing_basis(
    inducing_basis: InducingBasis,
    kernel: gpx.kernels.AbstractKernel,
    x: jax.Array,
    jitter: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    if isinstance(inducing_basis, RFFInducingBasis):
        x_2d = x.reshape(-1, 1) if x.ndim == 1 else x
        basis = inducing_basis.basis_matrix(kernel, x, jitter)
        q_diag = jnp.sum(basis**2, axis=1)
        k_diag = jax.vmap(kernel, in_axes=(0, 0))(x_2d, x_2d)
        residual_std = jnp.sqrt(jnp.maximum(k_diag - q_diag, 0.0)).reshape(-1, 1)
        return basis, residual_std

    x_2d = x.reshape(-1, 1) if x.ndim == 1 else x
    k_zz = inducing_basis.inducing_cov(kernel)
    k_zx = inducing_basis.cross_cov(kernel, x)
    l_z = jnp.linalg.cholesky(k_zz + jitter * jnp.eye(k_zz.shape[0]))
    basis = solve_triangular(l_z, k_zx, lower=True).T  # (N, M)
    q_diag = jnp.sum(basis**2, axis=1)
    k_diag = jax.vmap(kernel, in_axes=(0, 0))(x_2d, x_2d)
    residual_std = jnp.sqrt(jnp.maximum(k_diag - q_diag, 0.0)).reshape(-1, 1)
    return basis, residual_std
