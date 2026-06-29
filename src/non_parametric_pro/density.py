"""Density and score functions for predictively oriented ULA."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


class ProParameters(NamedTuple):
    """Parameters controlling one ULA iteration."""

    y: jax.Array
    basis: jax.Array
    step_size: float
    nu: jax.Array | float
    alpha: float
    tolerance: float
    jitter: float


def normal_logpdf(y, mean, nu):
    """Evaluate the Gaussian observation density for each particle."""
    return (
        -0.5 * ((y - mean) / nu) ** 2
        - jnp.log(nu)
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )


def pro_logdensity_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Compute the log density of the posterior over latent particles."""
    a = parameters.basis @ z
    log_density = normal_logpdf(parameters.y, a, parameters.nu)
    num_particles = log_density.shape[1]
    log_marginal = logsumexp(log_density, axis=1) - jnp.log(num_particles)
    likelihood = num_particles * parameters.alpha * jnp.sum(log_marginal)
    return likelihood - 0.5 * jnp.sum(z**2)


def pro_logdensity_and_grad_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> tuple[jax.Array, jax.Array]:
    """Compute the scalar PRO log density and its hand-derived gradient."""
    a = parameters.basis @ z

    log_density = normal_logpdf(parameters.y, a, parameters.nu)
    num_particles = log_density.shape[1]

    log_marginal = logsumexp(log_density, axis=1, keepdims=True) - jnp.log(
        num_particles
    )

    weights = jnp.exp(log_density - log_marginal) * (
        parameters.y - a
    ) / parameters.nu**2

    grad = parameters.alpha * (parameters.basis.T @ weights) - z

    likelihood = num_particles * parameters.alpha * jnp.sum(log_marginal[:, 0])
    logdensity = likelihood - 0.5 * jnp.sum(z**2)

    return logdensity, grad


def predictive_score(z: jax.Array, parameters: ProParameters) -> jax.Array:
    """Compute the average negative predictive log score."""
    a = parameters.basis @ z
    log_density = normal_logpdf(parameters.y, a, parameters.nu)
    log_marginal = logsumexp(log_density, axis=1) - jnp.log(log_density.shape[1])
    return -parameters.alpha * jnp.mean(log_marginal)
