"""Tests for ULA sampler transitions."""

import jax.numpy as jnp
import jax.random as jr

from non_parametric_pro.density import (
    ProParameters,
    predictive_score,
    pro_logdensity_grad_fn,
)
from non_parametric_pro.ula import pro_ula


def test_ula_info_contains_average_predictive_score() -> None:
    """Transition info reports the average negative predictive log score."""
    key = jr.key(0)
    basis = jnp.eye(3)
    y = jnp.array([0.0, 1.0, 2.0])
    position = jr.normal(key, (3, 4))
    parameters = ProParameters(
        y=y,
        basis=basis,
        step_size=1e-3,
        nu=0.2,
        alpha=1.0,
        tolerance=1e-12,
        jitter=1e-5,
    )

    sampler = pro_ula(pro_logdensity_grad_fn, parameters)
    state = sampler.init(position)
    new_state, info = sampler.step(key, state)

    assert jnp.isfinite(info.score)
    assert jnp.allclose(info.score, predictive_score(new_state.position, parameters))
