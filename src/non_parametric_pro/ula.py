from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from blackjax.mcmc import diffusions
from blackjax.types import PRNGKey, ArrayTree

from non_parametric_pro.density import (
    ProParameters,
    normal_pdf,
    pro_logdensity_grad_fn,
)


class ULAState(NamedTuple):
    """Particle state carried between ULA iterations."""

    position: ArrayTree
    logdensity: ArrayTree | float
    logdensity_grad: ArrayTree


class ULAInfo(NamedTuple):
    """Diagnostics returned by an ULA iteration."""

    score: ArrayTree


def init(position: ArrayTree, parameters: ProParameters) -> ULAState:
    """Initialize a ULA state from particle positions."""
    logdensity, logdensity_grad = pro_logdensity_grad_fn(position, parameters)
    return ULAState(position, logdensity, logdensity_grad)


def score(state: ULAState, parameters: ProParameters) -> ArrayTree:
    """Compute the predictive score."""
    density = normal_pdf(parameters.y, parameters.basis @ state.position, parameters.nu)
    marginal = jnp.maximum(
        jnp.mean(density, axis=1),
        parameters.tolerance,
    )
    return -parameters.alpha * jnp.mean(jnp.log(marginal))


def build_kernel() -> Callable:
    """Build a ULA kernel using BlackJAX's overdamped Langevin diffusion."""
    one_step = diffusions.overdamped_langevin(pro_logdensity_grad_fn)

    def kernel(
        rng_key: PRNGKey,
        state: ULAState,
        logdensity_fn: Callable,
        parameters: ProParameters,
    ) -> tuple[ULAState, ULAInfo]:
        del logdensity_fn

        new_state = one_step(
            rng_key,
            state, # type: ignore  # noqa: PGH003
            parameters.step_size,
            (parameters,),
        )
        new_state = ULAState(*new_state)

        return new_state, ULAInfo(score(new_state, parameters))

    return kernel
