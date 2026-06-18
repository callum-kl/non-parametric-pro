"""BlackJAX-style ULA sampler."""

from collections.abc import Callable
from typing import NamedTuple

import jax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import diffusions
from blackjax.types import PRNGKey

from non_parametric_pro.density import (
    ProParameters,
    predictive_score,
)


class ULAState(NamedTuple):
    """Particle state carried between ULA iterations."""

    position: jax.Array
    logdensity: jax.Array
    logdensity_grad: jax.Array


class ULAInfo(NamedTuple):
    """Diagnostics returned by an ULA iteration."""

    score: jax.Array


def init(
    position: jax.Array,
    parameters: ProParameters,
    logdensity_fn: Callable,
) -> ULAState:
    """Initialize a ULA state from particle positions."""
    grad_fn = build_density_fn(logdensity_fn, parameters)
    logdensity, logdensity_grad = grad_fn(position)
    return ULAState(position, logdensity, logdensity_grad)


def build_kernel() -> Callable:
    """Build a ULA kernel using BlackJAX's overdamped Langevin diffusion."""

    def kernel(
        rng_key: PRNGKey,
        state: ULAState,
        logdensity_fn: Callable,
        parameters: ProParameters,
    ) -> tuple[ULAState, ULAInfo]:

        _logdensity_grad_fn = build_density_fn(logdensity_fn, parameters)
        one_step = diffusions.overdamped_langevin(_logdensity_grad_fn)

        new_state = one_step(
            rng_key,
            state,  # type: ignore  # noqa: PGH003
            parameters.step_size,
        )
        new_state = ULAState(*new_state)  # type: ignore  # noqa: PGH003

        return new_state, ULAInfo(predictive_score(new_state.position, parameters))

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    parameters: ProParameters,
) -> SamplingAlgorithm:
    """Create a BlackJAX-style ULA sampler with ``init`` and ``step`` methods."""
    kernel = build_kernel()

    def init_fn(position: jax.Array, rng_key: PRNGKey | None = None) -> ULAState:
        del rng_key
        return init(position, parameters, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state: ULAState) -> tuple[ULAState, ULAInfo]:
        return kernel(rng_key, state, logdensity_fn, parameters)

    return SamplingAlgorithm(init_fn, step_fn)


def build_density_fn(base: Callable, parameters: ProParameters) -> Callable:
    """Build a density function for the ULA kernel."""

    def density_fn(z: jax.Array) -> tuple[jax.Array, jax.Array]:
        return base(z, parameters)

    return density_fn


def pro_ula(
    logdensity_fn,
    parameters: ProParameters,
) -> SamplingAlgorithm:
    """Build a top-level predictively oriented ULA sampler."""
    return as_top_level_api(logdensity_fn, parameters)
