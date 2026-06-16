"""Sampling routines for predictively oriented posteriors."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr


class MeanFieldState(NamedTuple):
    """State carried between ULA iterations."""

    z: jax.Array
    a: jax.Array


class MeanFieldAdaptationState(NamedTuple):
    """State carried between ULA iterations."""

    step_size: float
    nu: float
    basis: jax.Array


class MeanFieldInfo(NamedTuple):
    """Diagnostics returned by an ULA iteration."""

    score: jax.Array


class MeanFieldParameters(NamedTuple):
    """Parameters controlling one ULA iteration."""

    y: jax.Array
    alpha: float
    tolerance: float
    jitter: float


def normal_pdf(y: jax.Array, mean: jax.Array, nu: float) -> jax.Array:
    """Evaluate a normal density at each observation and mean."""
    return jnp.exp(-0.5 * ((y[:, None] - mean) / nu) ** 2) / (
        jnp.sqrt(2.0 * jnp.pi) * nu
    )

def pro_ula_step(
    key: jax.Array,
    state: MeanFieldState,
    adapt: MeanFieldAdaptationState,
    parameters: MeanFieldParameters,
) -> tuple[MeanFieldState, MeanFieldInfo]:
    """Advance the ULA chain by one iteration."""
    z, a = state.z, state.a

    density = normal_pdf(parameters.y, a, adapt.nu)
    marginal = jnp.maximum(
        jnp.mean(density, axis=1, keepdims=True),
        parameters.tolerance,
    )
    weights = (
        density / marginal * (parameters.y[:, None] - a) / adapt.nu**2
    )

    grad = adapt.basis.T @ weights
    noise = jr.normal(key, z.shape)

    new_z = (
        z
        + adapt.step_size * (parameters.alpha * grad - z)
        + jnp.sqrt(2.0 * adapt.step_size) * noise
    )
    new_a = adapt.basis @ new_z

    score = -parameters.alpha * jnp.mean(jnp.log(marginal[:, 0]))

    return MeanFieldState(new_z, new_a), MeanFieldInfo(score)


MEAN_FIELD_TYPE = tuple[tuple[MeanFieldState, MeanFieldAdaptationState], MeanFieldInfo]


def custom_pro_ula_sampling(
        key: jax.Array,
        init_state: MeanFieldState,
        adapt: MeanFieldAdaptationState,
        num_steps: int,
        *,
        parameters: MeanFieldParameters
    ) -> MEAN_FIELD_TYPE:
    """Run the ULA sampler for a given number of steps."""
    keys = jr.split(key, num_steps)

    def one_step(carry: tuple, key: jax.Array) -> MEAN_FIELD_TYPE:
        state, adapt = carry
        new_state, info = pro_ula_step(key, state, adapt, parameters)
        return (new_state, adapt), info
    carry = (init_state, adapt)

    return jax.lax.scan(one_step, carry, keys)

run_sampler_jit = jax.jit(custom_pro_ula_sampling, static_argnames=("num_steps",))
