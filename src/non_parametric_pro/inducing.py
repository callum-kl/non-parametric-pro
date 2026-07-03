"""Inducing feature bases for sparse GP approximations."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey
from jax.scipy.linalg import solve_triangular


@runtime_checkable
class InducingBasis(Protocol):
    """
    Protocol for inducing-feature bases.

    Implementations provide the two kernel evaluations needed to build
    the basis matrix ``B = K_xz L_zz^{-T}``:

    - ``inducing_cov`` — the ``(M, M)`` covariance between inducing features
    - ``cross_cov`` — the ``(M, N)`` cross-covariance between inducing features
      and the training inputs

    Any class that implements these two methods satisfies the protocol, so
    point inducing inputs, variational Fourier features, random Fourier
    features, etc. all fit naturally without inheriting from a common base.
    """

    def inducing_cov(self, kernel: gpx.kernels.AbstractKernel) -> jax.Array:
        """K_zz: (M, M) covariance between inducing features."""
        ...

    def cross_cov(
        self, kernel: gpx.kernels.AbstractKernel, x: jax.Array
    ) -> jax.Array:
        """K_zx: (M, N) cross-covariance between inducing features and x."""
        ...


@dataclass
class PointInducingBasis:
    """
    Standard inducing point basis (Titsias 2009).

    Inducing features are GP evaluations at ``M`` fixed locations ``z``,
    so ``K_zz = k(z, z)`` and ``K_zx = k(z, x)`` are ordinary kernel matrices.
    """

    z: jax.Array  # (M,) or (M, D) inducing locations

    def _z2d(self) -> jax.Array:
        return self.z.reshape(-1, 1) if self.z.ndim == 1 else self.z

    def inducing_cov(self, kernel: gpx.kernels.AbstractKernel) -> jax.Array:
        return kernel.gram(self._z2d()).as_matrix()

    def cross_cov(
        self, kernel: gpx.kernels.AbstractKernel, x: jax.Array
    ) -> jax.Array:
        return kernel.cross_covariance(self._z2d(), x)


def kmeans_inducing_points(
    rng_key: PRNGKey,
    x_train: jax.Array,
    num_inducing: int,
    *,
    num_iters: int = 100,
) -> "PointInducingBasis":
    """
    Select inducing locations via k-means clustering of the training inputs.

    Runs Lloyd's algorithm for ``num_iters`` steps (no early stopping) with a
    random-subset initialisation.  The returned :class:`PointInducingBasis`
    uses the cluster centroids as inducing points, which tend to give better
    coverage than random subsets, especially for unevenly distributed inputs.

    This function is JIT-compatible; wrap in ``jax.jit`` for repeated calls.

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
    centroids = x[init_idx]  # (M, D)

    def step(centroids: jax.Array, _: None) -> tuple[jax.Array, None]:
        dists = jnp.sum(
            (x[:, None, :] - centroids[None, :, :]) ** 2, axis=-1
        )                                               # (N, M)
        assignments = jnp.argmin(dists, axis=1)        # (N,)
        one_hot = jax.nn.one_hot(assignments, num_inducing)  # (N, M)
        counts = one_hot.sum(axis=0)                   # (M,)
        new_centroids = one_hot.T @ x / jnp.maximum(counts[:, None], 1)
        # Keep old centroid for any empty cluster
        return jnp.where(counts[:, None] > 0, new_centroids, centroids), None

    centroids, _ = jax.lax.scan(step, centroids, None, length=num_iters)
    return PointInducingBasis(centroids)


def compute_inducing_basis(
    inducing_basis: InducingBasis,
    kernel: gpx.kernels.AbstractKernel,
    x: jax.Array,
    jitter: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """
    Build the basis ``B = K_xz L_zz^{-T}`` and residual std for any ``InducingBasis``.

    The residual std is ``sqrt(max(diag(K_xx) - diag(B Bᵀ), 0))``, i.e. the
    per-point GP posterior standard deviation not captured by the inducing
    features. It is shaped ``(N, 1)`` for broadcasting with ``y`` and the
    particle predictions.

    Parameters
    ----------
    inducing_basis
        Any object implementing the ``InducingBasis`` protocol.
    kernel
        A gpjax kernel (already ``paramax.unwrap``-ed if needed by the caller).
    x
        Training inputs, shape ``(N, D)``.
    jitter
        Diagonal jitter for the Cholesky of ``K_zz``.
    """
    k_zz = inducing_basis.inducing_cov(kernel)
    k_zx = inducing_basis.cross_cov(kernel, x)
    l_z = jnp.linalg.cholesky(k_zz + jitter * jnp.eye(k_zz.shape[0]))
    basis = solve_triangular(l_z, k_zx, lower=True).T  # (N, M)
    q_diag = jnp.sum(basis**2, axis=1)
    k_diag = jnp.diag(kernel.gram(x).as_matrix())
    residual_std = jnp.sqrt(jnp.maximum(k_diag - q_diag, 0.0)).reshape(-1, 1)
    return basis, residual_std
