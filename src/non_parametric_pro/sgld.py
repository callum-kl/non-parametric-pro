from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import diffusions
from blackjax.types import PRNGKey

from non_parametric_pro.density import pro_score_fn
from non_parametric_pro.ula import ULAInfo, ULAState, init, refresh

__all__ = [
    "SGLDAlgorithm",
    "build_kernel",
    "init",
    "parametric_sgld",
    "refresh",
    "sgld",
]


def build_kernel(logdensity_fn: Callable, *, batch_size: int) -> Callable:
    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
    one_step = diffusions.overdamped_langevin(logdensity_and_grad_fn)

    def kernel(
        rng_key: PRNGKey,
        state: ULAState,
        parameters: NamedTuple,
    ) -> tuple[ULAState, ULAInfo]:
        """Generate a new sample with the SGLD kernel."""
        batch_key, step_key = jax.random.split(rng_key)
        n = parameters.basis.shape[0]
        batch_idx = jax.random.choice(batch_key, n, (batch_size,), replace=False)
        parameters = parameters._replace(batch_idx=batch_idx)

        new_state = one_step(
            step_key,
            state,
            parameters.step_size,
            (parameters,),
        )
        new_state = ULAState(*new_state)

        score = pro_score_fn(new_state.position, parameters)
        return new_state, ULAInfo(score)

    return kernel


@dataclass(frozen=True)
class SGLDAlgorithm:
    batch_size: int

    def build_kernel(self, logdensity_fn: Callable) -> Callable:
        return build_kernel(logdensity_fn, batch_size=self.batch_size)

    def init(
        self, position: jax.Array, parameters: NamedTuple, logdensity_fn: Callable
    ) -> ULAState:
        return init(position, parameters, logdensity_fn)

    def refresh(
        self, state: ULAState, parameters: NamedTuple, logdensity_fn: Callable
    ) -> ULAState:
        return refresh(state, parameters, logdensity_fn)


def sgld(batch_size: int) -> SGLDAlgorithm:
    return SGLDAlgorithm(batch_size=batch_size)


def parametric_sgld(
    logdensity_fn: Callable,
    parameters: NamedTuple,
    *,
    batch_size: int,
) -> SamplingAlgorithm:
    kernel = build_kernel(logdensity_fn, batch_size=batch_size)

    def init_fn(position: jax.Array, rng_key: PRNGKey | None = None) -> ULAState:
        del rng_key
        return init(position, parameters, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state: ULAState) -> tuple[ULAState, ULAInfo]:
        return kernel(rng_key, state, parameters)

    return SamplingAlgorithm(init_fn, step_fn)
