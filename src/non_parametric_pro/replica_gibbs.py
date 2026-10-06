from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.base import SamplingAlgorithm
from blackjax.types import PRNGKey
from jax.scipy.linalg import solve_triangular

from non_parametric_pro.density import (
    ProParameters,
    _effective_sigma,
    pro_replica_fn,
    pro_score_fn,
)


class ReplicaGibbsState(NamedTuple):
    position: jax.Array


class ReplicaGibbsInfo(NamedTuple):
    score: jax.Array


def init(
    position: jax.Array,
    parameters: ProParameters,
    logdensity_fn: Callable,
) -> ReplicaGibbsState:
    del parameters, logdensity_fn
    return ReplicaGibbsState(position)


def refresh(
    state: ReplicaGibbsState,
    parameters: ProParameters,
    logdensity_fn: Callable,
) -> ReplicaGibbsState:
    return init(state.position, parameters, logdensity_fn)


def validate_r(num_particles: int, alpha: float, *, tol: float = 1e-6) -> int:
    r = num_particles * alpha
    r_int = round(r)
    if r_int < 1 or abs(r - r_int) > tol:
        msg = (
            "replica_gibbs requires r = num_particles * alpha to be a positive "
            f"integer (got num_particles={num_particles}, alpha={alpha}, r={r})."
        )
        raise ValueError(msg)
    return r_int


def _gaussian_block(
    rng_key: PRNGKey,
    basis: jax.Array,
    y: jax.Array,
    sigma: jax.Array,
    counts: jax.Array,
) -> jax.Array:
    basis_dim = basis.shape[1]
    precision_flat = jnp.reshape(jnp.asarray(sigma) ** -2, -1)
    y_flat = y.reshape(-1)
    keys = jr.split(rng_key, counts.shape[1])

    def step(_, xs):
        count_k, key_k = xs
        weight_k = count_k * precision_flat
        weighted_basis = weight_k[:, None] * basis
        precision_matrix = jnp.eye(basis_dim) + basis.T @ weighted_basis
        rhs = basis.T @ (weight_k * y_flat)

        chol = jnp.linalg.cholesky(precision_matrix)
        mean = solve_triangular(
            chol.T, solve_triangular(chol, rhs, lower=True), lower=False
        )
        noise = solve_triangular(
            chol.T, jr.normal(key_k, (basis_dim,)), lower=False
        )
        return None, mean + noise

    _, z_columns = jax.lax.scan(step, None, (counts.T, keys))
    return z_columns.T


def one_step(
    rng_key: PRNGKey,
    state: ReplicaGibbsState,
    parameters: ProParameters,
) -> ReplicaGibbsState:
    alloc_key, gibbs_key = jr.split(rng_key)

    z = state.position
    num_particles = z.shape[1]
    r = jnp.round(num_particles * parameters.alpha)

    basis = parameters.basis
    u = basis @ z
    log_weights = pro_replica_fn(u, parameters)
    counts = jr.multinomial(alloc_key, n=r, p=jnp.exp(log_weights))

    sigma = _effective_sigma(parameters)
    y = jnp.asarray(parameters.y)
    z_new = _gaussian_block(gibbs_key, basis, y, sigma, counts)

    return ReplicaGibbsState(z_new)


def build_kernel(logdensity_fn: Callable) -> Callable:
    del logdensity_fn

    def kernel(
        rng_key: PRNGKey,
        state: ReplicaGibbsState,
        parameters: ProParameters,
    ) -> tuple[ReplicaGibbsState, ReplicaGibbsInfo]:
        new_state = one_step(rng_key, state, parameters)
        score = pro_score_fn(new_state.position, parameters)
        return new_state, ReplicaGibbsInfo(score)

    return kernel


def parametric_replica_gibbs(
    logdensity_fn: Callable,
    parameters: NamedTuple,
) -> SamplingAlgorithm:
    kernel = build_kernel(logdensity_fn)

    def init_fn(
        position: jax.Array, rng_key: PRNGKey | None = None
    ) -> ReplicaGibbsState:
        del rng_key
        return init(position, parameters, logdensity_fn)

    def step_fn(
        rng_key: PRNGKey, state: ReplicaGibbsState
    ) -> tuple[ReplicaGibbsState, ReplicaGibbsInfo]:
        return kernel(rng_key, state, parameters)

    return SamplingAlgorithm(init_fn, step_fn)
