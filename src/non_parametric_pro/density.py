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
    step_size: float
    sigma: jax.Array | float | paramax.AbstractUnwrappable
    alpha: float
    jitter: float = 1e-6
    tolerance: float = 1e-300
    basis: jax.Array | None = None
    residual_std: jax.Array | None = None
    batch_idx: jax.Array | None = None


def _effective_sigma(parameters: "ProParameters") -> jax.Array:
    """Sigma for full GP, sqrt(sigma² + residual_std²) for inducing."""
    sigma = paramax.unwrap(parameters.sigma)
    if parameters.residual_std is None:
        return sigma
    return jnp.sqrt(sigma**2 + parameters.residual_std**2)


def normal_logpdf(y, mean, sigma):
    """Evaluate the Gaussian observation density for each particle."""
    return (
        -0.5 * ((y - mean) / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)
    )


def pro_logdensity_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Compute the log density of the posterior over latent particles."""
    return pro_score_fn(z, parameters) - 0.5 * jnp.sum(z**2)


def _batch_parameters(parameters: ProParameters) -> tuple[ProParameters, jax.Array]:
    if parameters.batch_idx is None:
        return parameters, jnp.asarray(1.0)
    n = parameters.basis.shape[0]
    b = parameters.batch_idx.shape[0]
    basis = jnp.asarray(parameters.basis)
    y = jnp.asarray(parameters.y)
    residual_std = (
        jnp.asarray(parameters.residual_std)[parameters.batch_idx]
        if parameters.residual_std is not None
        else None
    )
    batched = parameters._replace(
        basis=basis[parameters.batch_idx],
        y=y[parameters.batch_idx],
        residual_std=residual_std,
    )
    return batched, n / b


def pro_score_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    parameters, scale = _batch_parameters(parameters)
    sigma = _effective_sigma(parameters)
    a = parameters.basis @ z
    log_density = normal_logpdf(parameters.y, a, sigma)
    num_particles = log_density.shape[1]
    log_marginal = logsumexp(log_density, axis=1) - jnp.log(num_particles)
    log_marginal = jnp.maximum(log_marginal, jnp.log(parameters.tolerance))
    score = num_particles * parameters.alpha * scale * jnp.sum(log_marginal)
    return score


def predictive_score(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Compute the predictive score reported by ULA transition info."""
    return pro_score_fn(z, parameters)


def pro_logdensity_and_grad_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> tuple[jax.Array, jax.Array]:
    """Compute the scalar PRO log density and hand-derived gradient."""
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
