from collections.abc import Callable
from typing import NamedTuple

import jax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import diffusions
from blackjax.types import PRNGKey

from non_parametric_pro.density import pro_score_fn


class ULAState(NamedTuple):
    position: jax.Array
    logdensity: jax.Array
    logdensity_grad: jax.Array


class ULAInfo(NamedTuple):
    score: jax.Array


def init(
    position: jax.Array,
    parameters: NamedTuple,
    logdensity_fn: Callable,
) -> ULAState:
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(
        position,
        parameters,
    )
    return ULAState(position, logdensity, logdensity_grad)


def refresh(
    state: ULAState,
    parameters: NamedTuple,
    logdensity_fn: Callable,
) -> ULAState:
    return init(state.position, parameters, logdensity_fn)


def build_kernel(logdensity_fn: Callable) -> Callable:
    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
    one_step = diffusions.overdamped_langevin(logdensity_and_grad_fn)

    def kernel(
        rng_key: PRNGKey,
        state: ULAState,
        parameters: NamedTuple,
    ) -> tuple[ULAState, ULAInfo]:
        new_state = one_step(
            rng_key,
            state,
            parameters.step_size,
            (parameters,),
        )
        new_state = ULAState(*new_state)

        score = pro_score_fn(new_state.position, parameters)
        return new_state, ULAInfo(score)

    return kernel


def parametric_ula(
    logdensity_fn: Callable,
    parameters: NamedTuple,
) -> SamplingAlgorithm:
    kernel = build_kernel(logdensity_fn)

    def init_fn(position: jax.Array, rng_key: PRNGKey | None = None) -> ULAState:
        del rng_key
        return init(position, parameters, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state: ULAState) -> tuple[ULAState, ULAInfo]:
        return kernel(rng_key, state, parameters)

    return SamplingAlgorithm(init_fn, step_fn)
