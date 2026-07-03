"""Density and score functions for predictively oriented ULA."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import paramax
from jax.scipy.special import logsumexp


class ProParameters(NamedTuple):
    """
    Parameters controlling one ULA iteration.
    """

    y: jax.Array
    basis: jax.Array
    step_size: float
    sigma: jax.Array | float | paramax.AbstractUnwrappable
    alpha: float
    tolerance: float = 1e-30
    jitter: float = 1e-6
    residual_std: jax.Array | None = None  # (N,1) inducing residual; None for full GP


def _effective_sigma(parameters: "ProParameters") -> jax.Array:
    """sigma for full GP, sqrt(sigma² + residual_std²) for inducing."""
    sigma = paramax.unwrap(parameters.sigma)
    if parameters.residual_std is None:
        return sigma
    return jnp.sqrt(sigma**2 + parameters.residual_std**2)


def normal_logpdf(y, mean, sigma):
    """Evaluate the Gaussian observation density for each particle."""
    return (
        -0.5 * ((y - mean) / sigma) ** 2
        - jnp.log(sigma)
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )

def student_t_logpdf(y, mean, sigma, df=4.0):
    """Evaluate the Student-t observation log-density for each particle."""
    z = (y - mean) / sigma
    nu = df
    return (
        jax.scipy.special.gammaln((nu + 1.0) / 2.0)
        - jax.scipy.special.gammaln(nu / 2.0)
        - 0.5 * jnp.log(nu * jnp.pi)
        - jnp.log(sigma)
        - 0.5 * (nu + 1.0) * jnp.log1p((z**2) / nu)
    )


def pro_logdensity_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Compute the log density of the posterior over latent particles."""
    return pro_score_fn(z, parameters) - 0.5 * jnp.sum(z**2)


def pro_score_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Compute the score function."""
    sigma = _effective_sigma(parameters)
    a = parameters.basis @ z
    log_density = normal_logpdf(parameters.y, a, sigma)
    num_particles = log_density.shape[1]
    log_marginal = logsumexp(log_density, axis=1) - jnp.log(num_particles)
    log_marginal = jnp.maximum(log_marginal, jnp.log(parameters.tolerance))
    score = num_particles * parameters.alpha * jnp.sum(log_marginal)
    return score  


def pro_logdensity_and_grad_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> tuple[jax.Array, jax.Array]:
    """Compute the scalar PRO log density and its hand-derived gradient."""
    sigma = _effective_sigma(parameters)
    a = parameters.basis @ z

    log_density = normal_logpdf(parameters.y, a, sigma)
    num_particles = log_density.shape[1]

    log_marginal = logsumexp(log_density, axis=1, keepdims=True) - jnp.log(
        num_particles
    )

    weights = jnp.exp(log_density - log_marginal) * (parameters.y - a) / sigma**2

    grad = parameters.alpha * (parameters.basis.T @ weights) - z

    likelihood = num_particles * parameters.alpha * jnp.sum(log_marginal[:, 0])
    logdensity = likelihood - 0.5 * jnp.sum(z**2)

    return logdensity, grad
