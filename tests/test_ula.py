"""Tests for ULA sampler transitions."""

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr

from non_parametric_pro.density import (
    ProParameters,
    predictive_score,
    pro_logdensity_and_grad_fn,
    pro_logdensity_fn,
)
from non_parametric_pro.ula import build_kernel, init, parametric_ula, refresh


def test_ula_info_contains_average_predictive_score() -> None:
    """Transition info reports the average negative predictive log score."""
    key = jr.key(0)
    basis = jnp.eye(3)
    y = jnp.array([[0.0], [1.0], [2.0]])
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

    sampler = parametric_ula(pro_logdensity_fn, parameters)
    state = sampler.init(position)
    new_state, info = sampler.step(key, state)

    assert jnp.isfinite(info.score)
    assert jnp.allclose(info.score, predictive_score(new_state.position, parameters))


def test_pro_logdensity_fn_initializes_blackjax_mala() -> None:
    """The scalar PRO logdensity works with BlackJAX MALA."""
    key = jr.key(0)
    basis = jnp.eye(3)
    y = jnp.array([[0.0], [1.0], [2.0]])
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

    mala = blackjax.mala(lambda z: pro_logdensity_fn(z, parameters), 1e-3)
    state = mala.init(position)

    assert state.position.shape == position.shape
    assert state.logdensity.shape == ()
    assert state.logdensity_grad.shape == position.shape


def test_pro_manual_gradient_matches_scalar_logdensity_gradient() -> None:
    """Manual PRO gradient matches autodiff of the scalar PRO logdensity."""
    key = jr.key(1)
    basis = jnp.array([[1.0, 0.2, -0.1], [0.0, 0.5, 1.0]])
    y = jnp.array([[0.3], [-0.7]])
    position = jr.normal(key, (3, 5))
    parameters = ProParameters(
        y=y,
        basis=basis,
        step_size=1e-3,
        nu=0.4,
        alpha=1.2,
        tolerance=1e-12,
        jitter=1e-5,
    )

    manual_logdensity, manual_grad = pro_logdensity_and_grad_fn(position, parameters)
    autodiff_logdensity, autodiff_grad = jax.value_and_grad(pro_logdensity_fn)(
        position,
        parameters,
    )

    assert jnp.allclose(manual_logdensity, autodiff_logdensity)
    assert jnp.allclose(manual_grad, autodiff_grad)


def test_ula_state_can_be_refreshed_after_basis_change() -> None:
    """Changing the basis refreshes cached state without rebuilding the kernel."""
    key = jr.key(2)
    basis = jnp.eye(3)
    y = jnp.array([[0.0], [1.0], [2.0]])
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
    adapted_parameters = parameters._replace(basis=1.1 * basis)

    kernel = build_kernel(pro_logdensity_fn)
    state = init(position, parameters, pro_logdensity_fn)
    refreshed_state = refresh(state, adapted_parameters, pro_logdensity_fn)
    new_state, info = kernel(key, refreshed_state, adapted_parameters)

    assert refreshed_state.position is state.position
    assert jnp.allclose(
        refreshed_state.logdensity,
        pro_logdensity_fn(state.position, adapted_parameters),
    )
    assert new_state.position.shape == position.shape
    assert jnp.allclose(
        info.score,
        predictive_score(new_state.position, adapted_parameters),
    )
