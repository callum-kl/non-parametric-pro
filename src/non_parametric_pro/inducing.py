"""Inducing feature bases for sparse GP approximations."""

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


def sample_rff_frequencies(
    rng_key: PRNGKey,
    kernel: gpx.kernels.AbstractKernel,
    num_basis_fns: int,
    *,
    num_dims: int | None = None,
) -> jax.Array:
    """
    Sample raw (pre-lengthscale-scaling) Random Fourier Feature frequencies.

    Delegates to ``gpjax``'s own ``kernel.spectral_density.sample``, so the
    sampling matches whatever gpjax considers correct for the given
    stationary kernel type — this module doesn't re-derive any spectral
    density itself. The returned frequencies are independent of the
    kernel's *current* lengthscale value; :class:`RFFInducingBasis` rescales
    them by the adapted lengthscale on every call, so the same sample can be
    reused across an optimisation.

    Parameters
    ----------
    num_dims
        Number of input dimensions ``D``. Inferred from ``kernel.n_dims``
        (set automatically by gpjax when the kernel is constructed with an
        array-valued, i.e. ARD, ``lengthscale``) when not given explicitly.
        Must be supplied if the kernel has a scalar (isotropic) lengthscale,
        since gpjax then has no way to infer ``D`` on its own.
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

    Thin adapter around ``gpjax``'s own tested
    ``gpx.kernels.approximations.rff`` feature computation, reshaped to
    satisfy the ``InducingBasis`` protocol (``inducing_cov`` / ``cross_cov``)
    used by :func:`compute_inducing_basis` elsewhere in this module. Unlike
    :class:`PointInducingBasis`, ``kernel`` here is the *plain* base kernel
    (e.g. ``gpx.kernels.RBF``/``Matern32``) whose ``lengthscale``/``variance``
    are being fit — this class does its own feature computation rather than
    wrapping the kernel in ``gpx.kernels.approximations.RFF``, so it works
    with the same kernel object used everywhere else (adaptation, plain
    ``kernel.gram(...)`` calls for the residual/diagonal term, etc).

    Given fixed raw frequencies ``ω`` (see :func:`sample_rff_frequencies`),
    the feature map is ``φ(x) = [cos(x·ω/ℓ), sin(x·ω/ℓ)]`` (shape ``2M``),
    with ``k(x, y) ≈ (variance / M) * φ(x)·φ(y)``. Setting
    ``inducing_cov = (M / variance) * I`` and ``cross_cov(x) = φ(x)ᵀ`` makes
    :func:`compute_inducing_basis`'s generic ``L_zz``-solve reduce to a
    trivial diagonal rescale, and reproduces gpjax's own
    ``RFF(...).gram(x)`` to floating-point precision (verified numerically
    against it) — this is exactly the Rahimi & Recht Monte-Carlo kernel
    approximation, not an independent derivation.

    Parameters
    ----------
    frequencies
        Raw ``(M, D)`` frequencies from :func:`sample_rff_frequencies`,
        fixed for the lifetime of this basis (only the kernel's lengthscale
        rescales them on each call).
    """

    frequencies: jax.Array  # (M, D)

    def _features(
        self, kernel: gpx.kernels.AbstractKernel, x: jax.Array
    ) -> jax.Array:
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        lengthscale = paramax.unwrap(kernel.lengthscale)
        z = x @ (self.frequencies / lengthscale).T  # (N, M)
        return jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)  # (N, 2M)

    def inducing_cov(self, kernel: gpx.kernels.AbstractKernel) -> jax.Array:
        variance = paramax.unwrap(kernel.variance)
        num_features = 2 * self.frequencies.shape[0]
        scaling = variance / self.frequencies.shape[0]
        return jnp.eye(num_features) / scaling

    def cross_cov(
        self, kernel: gpx.kernels.AbstractKernel, x: jax.Array
    ) -> jax.Array:
        return self._features(kernel, x).T  # (2M, N)

    def basis_matrix(
        self,
        kernel: gpx.kernels.AbstractKernel,
        x: jax.Array,
        jitter: float = 1e-6,
    ) -> jax.Array:
        """
        Compute ``K_xz L_zz^{-T}`` without materialising or factorising ``K_zz``.

        For RFFs, ``K_zz = I / scaling`` with
        ``scaling = variance / num_frequencies``. The generic Cholesky solve
        therefore reduces to a scalar rescaling of the feature matrix.
        """
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
    if isinstance(inducing_basis, RFFInducingBasis):
        x_2d = x.reshape(-1, 1) if x.ndim == 1 else x
        basis = inducing_basis.basis_matrix(kernel, x, jitter)
        q_diag = jnp.sum(basis**2, axis=1)
        k_diag = jax.vmap(kernel, in_axes=(0, 0))(x_2d, x_2d)
        residual_std = jnp.sqrt(jnp.maximum(k_diag - q_diag, 0.0)).reshape(-1, 1)
        return basis, residual_std

    k_zz = inducing_basis.inducing_cov(kernel)
    k_zx = inducing_basis.cross_cov(kernel, x)
    l_z = jnp.linalg.cholesky(k_zz + jitter * jnp.eye(k_zz.shape[0]))
    basis = solve_triangular(l_z, k_zx, lower=True).T  # (N, M)
    q_diag = jnp.sum(basis**2, axis=1)
    k_diag = jnp.diag(kernel.gram(x).as_matrix())
    residual_std = jnp.sqrt(jnp.maximum(k_diag - q_diag, 0.0)).reshape(-1, 1)
    return basis, residual_std
