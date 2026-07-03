"""Tests for posterior predictive utilities."""

import jax
import jax.numpy as jnp
import jax.random as jr

from non_parametric_pro.util import (
    crps_gp,
    crps_pro,
    draw_predictive_samples,
    pit_gp,
    pit_pro,
    posterior_function_draws,
    predictive_moments,
    project_particles,
)


def test_predictive_moments_from_particle_matrix() -> None:
    """Predictive moments match direct projected-particle calculations."""
    basis = jnp.eye(2)
    particles = jnp.array([[1.0, 3.0], [2.0, 6.0]])

    mean, std = predictive_moments(basis, particles, noise_std=0.5)

    assert jnp.allclose(mean, jnp.array([2.0, 4.0]))
    assert jnp.allclose(std, jnp.sqrt(jnp.array([1.25, 4.25])))


def test_project_particles_flattens_scanned_history() -> None:
    """Scanned histories are flattened across draws and particles."""
    basis = jnp.eye(2)
    particles = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    projected = project_particles(basis, particles)

    assert projected.shape == (2, 4)
    expected = jnp.array([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]])
    assert jnp.allclose(projected, expected)


def test_draw_predictive_samples_shape() -> None:
    """Predictive samples have one column per requested draw."""
    basis = jnp.eye(3)
    particles = jnp.arange(12.0).reshape(3, 4)

    samples = draw_predictive_samples(
        jr.key(0),
        basis,
        particles,
        num_samples=5,
        noise_std=jnp.ones(3) * 0.1,
    )

    assert samples.shape == (3, 5)


def test_posterior_function_draws_shape() -> None:
    """Smooth function draws have one column per requested draw."""
    test_basis = jnp.eye(3)
    test_covariance = jnp.eye(3)
    particles = jnp.arange(12.0).reshape(3, 4)

    draws = posterior_function_draws(
        jr.key(0),
        test_basis,
        test_covariance,
        particles,
        num_draws=5,
    )

    assert draws.shape == (3, 5)


def test_crps_pro_single_particle_matches_gaussian_crps() -> None:
    """A one-component PRO mixture reduces to the Gaussian CRPS."""
    y = jnp.array([0.2, -0.4])
    test_basis = jnp.eye(2)
    test_covariance = jnp.eye(2)
    particles = jnp.array([[0.1], [-0.2]])
    sigma = 0.3

    pro_score = crps_pro(y, test_basis, test_covariance, particles, sigma=sigma)
    gp_score = crps_gp(y, particles[:, 0], jnp.full_like(y, sigma))

    assert jnp.allclose(pro_score, gp_score)


def test_crps_pro_matches_brute_force_mixture_formula() -> None:
    """PRO CRPS matches the closed-form mixture CRPS without residual variance."""
    y = jnp.array([0.0, 1.0])
    test_basis = jnp.eye(2)
    test_covariance = jnp.eye(2)
    particles = jnp.array([[0.0, 0.5, 1.0], [1.0, 1.5, 2.0]])
    sigma = 0.7
    sqrt_two = jnp.sqrt(jnp.array(2.0))

    projected = test_basis @ particles
    z = (y[:, None] - projected) / sigma
    first = jnp.mean(
        sigma
        * (
            2 * jnp.exp(-(z**2) / 2) / jnp.sqrt(2 * jnp.pi)
            + z * (2 * 0.5 * (1 + jax.scipy.special.erf(z / sqrt_two)) - 1)
        ),
        axis=1,
    )

    d = (projected[:, :, None] - projected[:, None, :]) / (sigma * sqrt_two)
    pairwise_abs = sigma * sqrt_two * (
        2 * jnp.exp(-(d**2) / 2) / jnp.sqrt(2 * jnp.pi)
        + d * (2 * 0.5 * (1 + jax.scipy.special.erf(d / sqrt_two)) - 1)
    )
    expected = jnp.mean(first - 0.5 * jnp.mean(pairwise_abs, axis=(1, 2)))

    assert jnp.allclose(
        crps_pro(y, test_basis, test_covariance, particles, sigma=sigma),
        expected,
    )


def test_pit_gp_matches_gaussian_cdf() -> None:
    """Gaussian PIT is the predictive CDF evaluated at each observation."""
    y = jnp.array([0.0, 1.0])
    mean = jnp.array([0.0, 0.0])
    std = jnp.array([1.0, 2.0])

    pit = pit_gp(y, mean, std)
    expected = jax.scipy.stats.norm.cdf((y - mean) / std)

    assert pit.shape == y.shape
    assert jnp.allclose(pit, expected)


def test_pit_pro_single_particle_matches_gaussian_pit() -> None:
    """A one-component PRO mixture has the same PIT as its Gaussian component."""
    y = jnp.array([0.2, -0.4])
    test_basis = jnp.eye(2)
    test_covariance = jnp.eye(2)
    particles = jnp.array([[0.1], [-0.2]])
    sigma = 0.3

    pro_pit = pit_pro(y, test_basis, test_covariance, particles, sigma=sigma)
    gp_pit = pit_gp(y, particles[:, 0], jnp.full_like(y, sigma))

    assert jnp.allclose(pro_pit, gp_pit)


def test_pit_pro_averages_component_cdfs() -> None:
    """PRO PIT evaluates the exact mixture CDF at each observation."""
    y = jnp.array([0.0])
    test_basis = jnp.array([[1.0]])
    test_covariance = jnp.array([[1.0]])
    particles = jnp.array([[-1.0, 1.0]])
    sigma = 0.5

    pit = pit_pro(y, test_basis, test_covariance, particles, sigma=sigma)
    expected = jnp.array([0.5])

    assert jnp.allclose(pit, expected)
