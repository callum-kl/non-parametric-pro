"""Utilities for posterior predictive summaries."""

import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

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
