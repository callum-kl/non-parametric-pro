"""Inducing feature bases for sparse GP approximations."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import gpjax as gpx
import jax
import jax.numpy as jnp
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


def compute_inducing_basis(
    inducing_basis: InducingBasis,
    kernel: gpx.kernels.AbstractKernel,
    x: jax.Array,
    jitter: float,
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
