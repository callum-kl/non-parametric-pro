"""Tests for posterior predictive utilities."""

import jax.numpy as jnp
import jax.random as jr

from non_parametric_pro.util import (
    draw_predictive_samples,
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
