"""Tests for ULA sampler transitions."""

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from gpjax.parameters import NonNegativeReal

from non_parametric_pro.density import (
    ProParameters,
    predictive_score,
    pro_crps_logdensity_fn,
    pro_crps_score_fn,
    pro_energy_score_fn,
    pro_logdensity_and_grad_fn,
    pro_logdensity_fn,
    pro_score_fn,
    regularised_crps_score,
    regularised_energy_score,
    regularised_score,
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
        sigma=NonNegativeReal(0.2),
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
        sigma=NonNegativeReal(0.2),
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
        sigma=NonNegativeReal(0.4),
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


def test_regularised_score_includes_optional_reflected_halfcauchy_sigma_prior() -> None:
    """Optional reflected half-Cauchy prior contributes to regularised score."""
    key = jr.key(3)
    basis = jnp.array([[1.0, 0.2], [0.0, 0.5]])
    y = jnp.array([[0.3], [-0.7]])
    position = jr.normal(key, (2, 4))
    sigma = NonNegativeReal(0.4)
    base_parameters = ProParameters(
        y=y,
        basis=basis,
        step_size=1e-3,
        sigma=sigma,
        alpha=1.2,
        tolerance=1e-12,
        jitter=1e-5,
    )
    prior_parameters = base_parameters._replace(
        sigma_prior_value=0.5,
        sigma_prior_scale=0.2,
    )

    expected_prior = (
        jnp.log(2.0 / jnp.pi)
        - jnp.log(0.2)
        - jnp.log1p(((0.5 - 0.4) / 0.2) ** 2)
    )

    expected_base_score = pro_score_fn(position, base_parameters) / (
        y.shape[0] * position.shape[1]
    )

    assert jnp.allclose(
        regularised_score(position, prior_parameters),
        expected_base_score + expected_prior,
    )
    assert jnp.allclose(
        pro_logdensity_fn(position, prior_parameters),
        pro_logdensity_fn(position, base_parameters),
    )
    assert jnp.allclose(
        pro_logdensity_and_grad_fn(position, prior_parameters)[0],
        pro_logdensity_and_grad_fn(position, base_parameters)[0],
    )

    above_cutoff_parameters = prior_parameters._replace(sigma=NonNegativeReal(0.6))

    assert jnp.isneginf(regularised_score(position, above_cutoff_parameters))


def test_regularised_score_rejects_negative_sigma_prior_scale() -> None:
    """The sigma prior scale must be positive."""
    parameters = ProParameters(
        y=jnp.array([[0.3], [-0.7]]),
        basis=jnp.eye(2),
        step_size=1e-3,
        sigma=NonNegativeReal(0.4),
        alpha=1.0,
        sigma_prior_value=0.5,
        sigma_prior_scale=-0.2,
    )
    position = jnp.ones((2, 4))

    with pytest.raises(ValueError, match="sigma_prior_scale must be positive"):
        regularised_score(position, parameters)


def test_pro_crps_score_matches_brute_force_mixture_formula() -> None:
    """CRPS training objective matches the Gaussian-mixture closed form."""
    y = jnp.array([[0.0], [1.0]])
    basis = jnp.eye(2)
    position = jnp.array([[0.0, 0.5, 1.0], [1.0, 1.5, 2.0]])
    sigma = 0.7
    alpha = 1.3
    parameters = ProParameters(
        y=y,
        basis=basis,
        step_size=1e-3,
        sigma=NonNegativeReal(sigma),
        alpha=alpha,
    )
    sqrt_two = jnp.sqrt(jnp.array(2.0))
    projected = basis @ position

    z = (y - projected) / sigma
    first = jnp.mean(
        sigma
        * (
            2 * jnp.exp(-(z**2) / 2) / jnp.sqrt(2 * jnp.pi)
            + z * (2 * 0.5 * (1 + jax.scipy.special.erf(z / sqrt_two)) - 1)
        ),
        axis=1,
    )

    d = (projected[:, :, None] - projected[:, None, :]) / (sigma * sqrt_two)
    pairwise_abs = sigma * sqrt_two * (
        2 * jnp.exp(-(d**2) / 2) / jnp.sqrt(2 * jnp.pi)
        + d * (2 * 0.5 * (1 + jax.scipy.special.erf(d / sqrt_two)) - 1)
    )
    crps = first - 0.5 * jnp.mean(pairwise_abs, axis=(1, 2))
    expected = -position.shape[1] * alpha * jnp.sum(crps)

    assert jnp.allclose(pro_crps_score_fn(position, parameters), expected)
    assert jnp.allclose(pro_energy_score_fn(position, parameters), expected)


def test_regularised_crps_score_includes_optional_sigma_prior() -> None:
    """Regularised CRPS uses the same optional reflected sigma prior."""
    basis = jnp.eye(2)
    y = jnp.array([[0.3], [-0.7]])
    position = jnp.array([[0.1, 0.2, 0.3], [-0.4, -0.2, 0.0]])
    base_parameters = ProParameters(
        y=y,
        basis=basis,
        step_size=1e-3,
        sigma=NonNegativeReal(0.4),
        alpha=1.2,
    )
    prior_parameters = base_parameters._replace(
        sigma_prior_value=0.5,
        sigma_prior_scale=0.2,
    )

    expected_prior = (
        jnp.log(2.0 / jnp.pi)
        - jnp.log(0.2)
        - jnp.log1p(((0.5 - 0.4) / 0.2) ** 2)
    )
    expected_base_score = pro_crps_score_fn(position, base_parameters) / (
        y.shape[0] * position.shape[1]
    )

    assert jnp.allclose(
        regularised_crps_score(position, prior_parameters),
        expected_base_score + expected_prior,
    )
    assert jnp.allclose(
        regularised_energy_score(position, prior_parameters),
        regularised_crps_score(position, prior_parameters),
    )


def test_pro_crps_logdensity_initializes_ula() -> None:
    """The CRPS-targeted scalar logdensity works with the ULA wrapper."""
    key = jr.key(4)
    basis = jnp.eye(3)
    y = jnp.array([[0.0], [1.0], [2.0]])
    position = jr.normal(key, (3, 4))
    parameters = ProParameters(
        y=y,
        basis=basis,
        step_size=1e-3,
        sigma=NonNegativeReal(0.2),
        alpha=1.0,
    )

    state = init(position, parameters, pro_crps_logdensity_fn)

    assert state.logdensity.shape == ()
    assert state.logdensity_grad.shape == position.shape
    assert jnp.isfinite(state.logdensity)


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
        sigma=NonNegativeReal(0.2),
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
