from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import PRNGKey, ArrayTree


class ProParameters(NamedTuple):
    """Parameters controlling one ULA iteration."""

    y: ArrayTree
    basis: ArrayTree
    step_size: float
    nu: ArrayTree | float
    alpha: float
    tolerance: float
    jitter: float

def normal_pdf(y, mean, nu):
    """Evaluate the Gaussian observation density for each particle."""
    return jnp.exp(-0.5 * ((y[:, None] - mean) / nu) ** 2) / (
        jnp.sqrt(2.0 * jnp.pi) * nu
    )

def pro_logdensity_grad_fn(z: ArrayTree, parameters: ProParameters):
    a = parameters.basis @ z

    density = normal_pdf(parameters.y, a, parameters.nu)
    marginal = jnp.maximum(
        jnp.mean(density, axis=1, keepdims=True),
        parameters.tolerance,
    )

    weights = (
        density
        / marginal
        * (parameters.y[:, None] - a)
        / parameters.nu**2
    )

    grad = parameters.alpha * (parameters.basis.T @ weights) - z

    logdensity = (
        parameters.alpha * jnp.sum(jnp.log(marginal[:, 0]))
        - 0.5 * jnp.sum(z**2)
    )

    return logdensity, grad
