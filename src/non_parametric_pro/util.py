"""Utilities for posterior predictive summaries."""

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import paramax
from blackjax.types import PRNGKey
from jax.scipy.linalg import solve_triangular

from non_parametric_pro.density import ProParameters
from non_parametric_pro.inducing import InducingBasis

PARTICLE_MATRIX_NDIM = 2
SCANNED_PARTICLES_NDIM = 3


def _as_particle_matrix(particles: jax.Array) -> jax.Array:
    """Return particles as a ``basis_dim x num_particles`` matrix."""
    if particles.ndim == PARTICLE_MATRIX_NDIM:
        return particles
    if particles.ndim == SCANNED_PARTICLES_NDIM:
        return jnp.transpose(particles, (1, 0, 2)).reshape(particles.shape[1], -1)
    msg = (
        "particles must have shape (basis_dim, num_particles) or "
        "(draws, basis_dim, particles)"
    )
    raise ValueError(msg)


def project_particles(basis: jax.Array, particles: jax.Array) -> jax.Array:
    """Project latent particles through a prediction basis matrix."""
    return basis @ _as_particle_matrix(particles)


def predictive_moments(
    basis: jax.Array,
    particles: jax.Array,
    *,
    noise_std: jax.Array | float = 0.0,
    residual_std: jax.Array | float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute predictive mean and standard deviation from retained particles.

    ``basis`` should have shape ``(num_points, basis_dim)``. ``particles`` can
    either have shape ``(basis_dim, num_particles)`` or the scanned shape
    ``(num_draws, basis_dim, num_particles)``.
    """
    projected = project_particles(basis, particles)
    mean = jnp.mean(projected, axis=1)
    particle_variance = jnp.mean((projected - mean[:, None]) ** 2, axis=1)
    variance = particle_variance + noise_std**2 + residual_std**2
    return mean, jnp.sqrt(jnp.maximum(variance, 0.0))


def draw_predictive_samples(  # noqa: PLR0913
    rng_key: PRNGKey,
    basis: jax.Array,
    particles: jax.Array,
    *,
    num_samples: int,
    noise_std: jax.Array | float = 0.0,
    residual_std: jax.Array | float = 0.0,
) -> jax.Array:
    """
    Draw posterior predictive samples from retained latent particles.

    The returned array has shape ``(num_points, num_samples)``. Each sample
    chooses one retained particle and adds Gaussian observation/residual noise.
    """
    projected = project_particles(basis, particles)
    num_particles = projected.shape[1]
    index_key, noise_key = jr.split(rng_key)
    particle_indices = jr.randint(index_key, (num_samples,), 0, num_particles)
    selected = projected[:, particle_indices]

    sample_std = jnp.sqrt(noise_std**2 + residual_std**2)
    if jnp.ndim(sample_std) == 1:
        sample_std = sample_std[:, None]
    noise = sample_std * jr.normal(noise_key, selected.shape)
    return selected + noise


def prediction_basis(
    kernel: gpx.kernels.AbstractKernel,
    x_train: jax.Array,
    x_test: jax.Array,
    parameters: ProParameters,
    *,
    inducing_basis: InducingBasis | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute ``(test_basis, test_covariance)`` for ``posterior_function_draws``.

    Both cases share the same formula ``B* = K(·, anchors) L⁻ᵀ``; what differs
    is the choice of anchors and the Cholesky ``L``:

    - **Full GP** (``inducing_basis=None``): anchors are the ``N`` training
      inputs; ``L`` is ``parameters.basis`` (the ``N×N`` training Cholesky).
    - **Inducing** (``inducing_basis`` given): anchors are the ``M`` inducing
      features; ``L_zz`` is recomputed from the kernel and
      ``inducing_basis.inducing_cov``.

    The returned ``test_covariance`` is ``K(x_test, x_test)`` in both cases.
    ``posterior_function_draws`` subtracts ``test_basis @ test_basis.T`` from
    it to get the residual epistemic uncertainty at test points.

    Parameters
    ----------
    kernel
        The fitted/adapted kernel (paramax-wrapped or plain).
    x_train
        Training inputs ``(N, D)``. Only used in the full GP case.
    x_test
        Test inputs ``(N_test, D)``.
    parameters
        Current ``ProParameters``; ``parameters.jitter`` is reused when
        recomputing ``L_zz`` for the inducing case.
    inducing_basis
        An :class:`InducingBasis` instance, or ``None`` for the full GP path.
    """
    k = paramax.unwrap(kernel)
    x_tr = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
    x_te = x_test.reshape(-1, 1) if x_test.ndim == 1 else x_test
    test_covariance = k.gram(x_te).as_matrix()

    if inducing_basis is None:
        L = parameters.basis  # (N, N) — the training Cholesky
        K_test_train = k.cross_covariance(x_te, x_tr)  # (N_test, N)
        test_basis = solve_triangular(L, K_test_train.T, lower=True).T
    else:
        K_zz = inducing_basis.inducing_cov(k)  # (M, M)
        L_zz = jnp.linalg.cholesky(
            K_zz + parameters.jitter * jnp.eye(K_zz.shape[0])
        )
        K_z_test = inducing_basis.cross_cov(k, x_te)  # (M, N_test)
        test_basis = solve_triangular(L_zz, K_z_test, lower=True).T  # (N_test, M)

    return test_basis, test_covariance


def nlpd_gp(
    y_test: jax.Array,
    predictive_mean: jax.Array,
    predictive_std: jax.Array,
) -> jax.Array:
    """
    Mean negative log predictive density for a Gaussian GP predictive.

    Parameters
    ----------
    y_test
        Observed test targets, shape ``(N_test,)`` or ``(N_test, 1)``.
    predictive_mean
        GP predictive mean, same shape as ``y_test``.
    predictive_std
        GP predictive standard deviation (including observation noise),
        same shape as ``y_test``.

    Returns
    -------
    Scalar NLPD averaged over test points.
    """
    y = y_test.squeeze()
    mu = predictive_mean.squeeze()
    std = predictive_std.squeeze()
    log_p = (
        -0.5 * jnp.log(2 * jnp.pi)
        - jnp.log(std)
        - 0.5 * ((y - mu) / std) ** 2
    )
    return -jnp.mean(log_p)


def nlpd_pro(
    y_test: jax.Array,
    test_basis: jax.Array,
    test_covariance: jax.Array,
    particles: jax.Array,
    *,
    sigma: jax.Array | float,
) -> jax.Array:
    """
    Mean negative log predictive density for the PRO GP mixture predictive.

    The predictive is a mixture of Gaussians, one per retained particle:

    .. math::

        p(y^* | \\mathcal{D}) \\approx \\frac{1}{J} \\sum_j
            \\mathcal{N}(y^* \\mid B^* z_j,\\; \\sigma_{\\text{eff}}^2)

    where :math:`\\sigma_{\\text{eff},i}^2 = \\sigma^2 + s_i^2` and
    :math:`s_i = \\sqrt{\\max(k(x^*_i, x^*_i) - \\|B^*_{i,:}\\|^2, 0)}` is the
    sparse-approximation residual at each test point (zero for the full GP).

    Parameters
    ----------
    y_test
        Observed test targets, shape ``(N_test, 1)``.
    test_basis
        Test projection matrix from :func:`prediction_basis`,
        shape ``(N_test, M)`` (inducing) or ``(N_test, N)`` (full GP).
    test_covariance
        Prior kernel matrix at test points ``K(x_*, x_*)``,
        shape ``(N_test, N_test)``, also from :func:`prediction_basis`.
    particles
        Retained latent particles, shape ``(M, J)`` or
        ``(num_steps, M, J)`` (scanned output from
        :func:`blackjax.util.run_inference_algorithm`).
    sigma
        Observation noise standard deviation (plain array or
        ``paramax``-wrapped; unwrapped automatically).

    Returns
    -------
    Scalar NLPD averaged over test points.
    """
    sigma_val = paramax.unwrap(sigma)
    particle_matrix = _as_particle_matrix(particles)       # (M, J_total)
    projected = test_basis @ particle_matrix               # (N_test, J_total)
    j_total = projected.shape[1]

    # Sparse-approximation residual at test points; 0 for full GP.
    q_diag = jnp.sum(test_basis**2, axis=1)               # (N_test,)
    k_diag = jnp.diag(test_covariance)                    # (N_test,)
    test_residual_std = jnp.sqrt(jnp.maximum(k_diag - q_diag, 0.0))
    sigma_eff = jnp.sqrt(sigma_val**2 + test_residual_std**2).reshape(-1, 1)

    y = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test   # (N_test, 1)
    log_normals = (
        -0.5 * jnp.log(2 * jnp.pi)
        - jnp.log(sigma_eff)
        - 0.5 * ((y - projected) / sigma_eff) ** 2
    )                                                      # (N_test, J_total)
    log_p = jax.nn.logsumexp(log_normals, axis=1) - jnp.log(j_total)
    return -jnp.mean(log_p)


def posterior_function_draws(  # noqa: PLR0913
    rng_key: PRNGKey,
    test_basis: jax.Array,
    test_covariance: jax.Array,
    particles: jax.Array,
    *,
    num_draws: int = 120,
    jitter: float = 1e-6,
) -> jax.Array:
    """
    Draw smooth posterior function trajectories from retained particles.

    ``test_basis`` should be the same projection matrix used for
    ``predictive_moments``. In the full-GP case this is ``V.T``, where
    ``V = solve(train_cholesky, K_test_train.T)``.
    """
    particle_matrix = _as_particle_matrix(particles)
    num_test = test_covariance.shape[0]

    index_key, noise_key = jr.split(rng_key)
    particle_indices = jr.randint(index_key, (num_draws,), 0, particle_matrix.shape[1])
    conditional_means = test_basis @ particle_matrix[:, particle_indices]

    conditional_covariance = (
        test_covariance
        - test_basis @ test_basis.T
        + jitter * jnp.eye(num_test, dtype=test_covariance.dtype)
    )
    residual_cholesky = jnp.linalg.cholesky(conditional_covariance)
    residual_noise = jr.normal(noise_key, (num_test, num_draws))
    return conditional_means + residual_cholesky @ residual_noise
