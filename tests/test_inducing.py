"""Tests for inducing feature basis construction."""

import gpjax as gpx
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from non_parametric_pro.density import ProParameters
from non_parametric_pro.inducing import RFFInducingBasis, compute_inducing_basis
from non_parametric_pro.util import prediction_basis


def _generic_rff_basis(
    basis: RFFInducingBasis,
    kernel: gpx.kernels.AbstractKernel,
    x: jnp.ndarray,
    jitter: float,
) -> jnp.ndarray:
    k_zz = basis.inducing_cov(kernel)
    k_zx = basis.cross_cov(kernel, x)
    l_z = jnp.linalg.cholesky(k_zz + jitter * jnp.eye(k_zz.shape[0]))
    return solve_triangular(l_z, k_zx, lower=True).T


def test_rff_compute_inducing_basis_matches_generic_cholesky_solve() -> None:
    """RFF fast path is algebraically equivalent to the generic Kzz solve."""
    x = jnp.array([[-1.0, 0.2], [0.0, -0.3], [0.7, 1.1]])
    frequencies = jnp.array([[0.2, 1.0], [-0.5, 0.4], [1.2, -0.7]])
    inducing_basis = RFFInducingBasis(frequencies)
    kernel = gpx.kernels.RBF(
        lengthscale=jnp.array([1.4, 0.8]),
        variance=jnp.array(1.7),
    )
    jitter = 1e-4

    fast_basis, fast_residual_std = compute_inducing_basis(
        inducing_basis, kernel, x, jitter
    )
    generic_basis = _generic_rff_basis(inducing_basis, kernel, x, jitter)
    q_diag = jnp.sum(generic_basis**2, axis=1)
    k_diag = jnp.diag(kernel.gram(x).as_matrix())
    generic_residual_std = jnp.sqrt(jnp.maximum(k_diag - q_diag, 0.0)).reshape(-1, 1)

    assert jnp.allclose(fast_basis, generic_basis)
    assert jnp.allclose(fast_residual_std, generic_residual_std)


def test_rff_prediction_basis_matches_generic_cholesky_solve() -> None:
    """RFF prediction basis uses the same scaling as the generic Kzz solve."""
    x_train = jnp.array([[-1.0, 0.2], [0.0, -0.3], [0.7, 1.1]])
    x_test = jnp.array([[0.3, -0.5], [1.2, 0.4]])
    frequencies = jnp.array([[0.2, 1.0], [-0.5, 0.4], [1.2, -0.7]])
    inducing_basis = RFFInducingBasis(frequencies)
    kernel = gpx.kernels.RBF(
        lengthscale=jnp.array([1.4, 0.8]),
        variance=jnp.array(1.7),
    )
    jitter = 1e-4
    train_basis, residual_std = compute_inducing_basis(
        inducing_basis, kernel, x_train, jitter
    )
    parameters = ProParameters(
        y=jnp.zeros((x_train.shape[0], 1)),
        basis=train_basis,
        step_size=1e-3,
        sigma=0.2,
        alpha=1.0,
        jitter=jitter,
        residual_std=residual_std,
    )

    fast_test_basis, test_covariance = prediction_basis(
        kernel,
        x_train,
        x_test,
        parameters,
        inducing_basis=inducing_basis,
    )
    generic_test_basis = _generic_rff_basis(inducing_basis, kernel, x_test, jitter)

    assert jnp.allclose(fast_test_basis, generic_test_basis)
    assert jnp.allclose(test_covariance, kernel.gram(x_test).as_matrix())
