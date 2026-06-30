"""Adaptation of kernel hyperparameters"""

from collections.abc import Callable
from typing import NamedTuple

from blackjax.base import AdaptationAlgorithm
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.progress_bar import gen_scan_fn
from blackjax.adaptation.base import AdaptationResults

import jax
import jax.numpy as jnp

class ParameterAdaptationState(NamedTuple):
    """State of the parameter adaptation algorithm."""
    
    nu: jax.Array


def base(
    is_mass_matrix_diagonal: bool,
    target_acceptance_rate: float = 0.80,
    initial_inverse_mass_matrix: Array | None = None,
    imm_shrinkage_to_previous: float = 0.0,
) -> tuple[Callable, Callable, Callable]:

    mm_init, mm_update, mm_final = mass_matrix_adaptation(
        is_mass_matrix_diagonal, imm_shrinkage_to_previous
    )
    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    def init(
        position: ArrayLikeTree, initial_step_size: float
    ) -> WindowAdaptationState:
        """Initialze the adaptation state and parameter values.

        Unlike the original Stan window adaptation we do not use the
        `find_reasonable_step_size` algorithm which we found to be unnecessary.
        We may reconsider this choice in the future.

        """
        num_dimensions = pytree_size(position)
        imm_state = mm_init(num_dimensions, initial_inverse_mass_matrix)

        ss_state = da_init(initial_step_size)

        return WindowAdaptationState(
            ss_state,
            imm_state,
            initial_step_size,
            imm_state.inverse_mass_matrix,
        )

    def fast_update(
        position: ArrayLikeTree,
        acceptance_rate: float,
        warmup_state: WindowAdaptationState,
    ) -> WindowAdaptationState:
        del position

        new_ss_state = da_update(warmup_state.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        return WindowAdaptationState(
            new_ss_state,
            warmup_state.imm_state,
            new_step_size,
            warmup_state.inverse_mass_matrix,
        )

    def slow_update(
        position: ArrayLikeTree,
        acceptance_rate: float,
        warmup_state: WindowAdaptationState,
    ) -> WindowAdaptationState:
        new_imm_state = mm_update(warmup_state.imm_state, position)
        new_ss_state = da_update(warmup_state.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        return WindowAdaptationState(
            new_ss_state, new_imm_state, new_step_size, warmup_state.inverse_mass_matrix
        )

    def slow_final(warmup_state: WindowAdaptationState) -> WindowAdaptationState:

        new_imm_state = mm_final(warmup_state.imm_state)
        new_ss_state = da_init(da_final(warmup_state.ss_state))
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        return WindowAdaptationState(
            new_ss_state,
            new_imm_state,
            new_step_size,
            new_imm_state.inverse_mass_matrix,
        )

    def update(
        adaptation_state: WindowAdaptationState,
        adaptation_stage: tuple,
        position: ArrayLikeTree,
        acceptance_rate: float,
    ) -> WindowAdaptationState:
        """Update the adaptation state and parameter values.

        Parameters
        ----------
        adaptation_state
            Current adptation state.
        adaptation_stage
            The current stage of the warmup: whether this is a slow window,
            a fast window and if we are at the last step of a slow window.
        position
            Current value of the model parameters.
        acceptance_rate
            Value of the acceptance rate for the last mcmc step.

        Returns
        -------
        The updated adaptation state.

        """
        stage, is_middle_window_end = adaptation_stage

        warmup_state = jax.lax.switch(
            stage,
            (fast_update, slow_update),
            position,
            acceptance_rate,
            adaptation_state,
        )

        warmup_state = jax.lax.cond(
            is_middle_window_end,
            slow_final,
            lambda x: x,
            warmup_state,
        )

        return warmup_state

    def final(warmup_state: WindowAdaptationState) -> tuple[float, Array]:
        """Return the final values for the step size and mass matrix."""
        step_size = jnp.exp(warmup_state.ss_state.log_step_size_avg)
        inverse_mass_matrix = warmup_state.imm_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, update, final

def merge_parameters(
    base: NamedTuple,
    new: NamedTuple,
    transforms: dict[str, Callable] | None = None,
) -> NamedTuple:
    transforms = transforms or {}
    updates = {
        f: transforms.get(f, lambda v: v)(v)
        for f, v in new._asdict().items()
        if f in base._fields
    }
    return base._replace(**updates)


def parameter_adaptation(
        algorithm,
        logdensity_fn: Callable,
        adaptation_info_fn: Callable,
        base_parameters: NamedTuple,
        progress_bar: bool = False,
):
    """Adapt kernel hyperparameters"""
    
    kernel = algorithm.build_kernel(logdensity_fn)

    adapt_init, adapt_step, adapt_final = base()
    
    def one_step(carry, xs):
        _, rng_key, adaptation_stage = xs
        state, adaptation_state = carry

        base_parameters = merge_parameters(base_parameters, adaptation_state)

        new_state, info = kernel(
            rng_key,
            state,
            adaptation_state
        )
        new_adaptation_state = adapt_step(
            adaptation_state,
            adaptation_stage,
            new_state.position,
            info.acceptance_rate,
        )

        return (
            (new_state, new_adaptation_state),
            adaptation_info_fn(new_state, info, new_adaptation_state),
        )
    
    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):
        init_state = algorithm.init(position, parameters, logdensity_fn)
        init_adaptation_state = adapt_init(position, parameters)

        if progress_bar:
            print("Running window adaptation")
        scan_fn = gen_scan_fn(num_steps, progress_bar=progress_bar)
        start_state = (init_state, init_adaptation_state)
        keys = jax.random.split(rng_key, num_steps)
        schedule = build_schedule(num_steps)
        last_state, info = scan_fn(
            one_step,
            start_state,
            (jnp.arange(num_steps), keys, schedule),
        )

        last_chain_state, last_warmup_state, *_ = last_state

        step_size, inverse_mass_matrix = adapt_final(last_warmup_state)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
            **extra_parameters,
        }

        return (
            AdaptationResults(
                last_chain_state,
                parameters,
            ),
            info,
        )

    return AdaptationAlgorithm(run)

def build_schedule(num_steps: int) -> jax.Array:
    """Build a schedule for the adaptation stages."""
    warmup_steps = num_steps // 2
    adaptation_schedule = jnp.concatenate(
        [
            jnp.zeros(warmup_steps, dtype=jnp.int32),
            jnp.ones(num_steps - warmup_steps, dtype=jnp.int32),
        ]
    )
    return adaptation_schedule