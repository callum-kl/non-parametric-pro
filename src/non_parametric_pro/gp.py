"""
One-shot GP fit used to seed ULA's parameters with a good starting point.

This is not an online adaptation scheme (no ``init``/``update``/``final``
triple): it just fits a GP once with gpjax and returns a :class:`ProParameters`
populated from the optimised kernel hyperparameters (and inducing inputs, if
``num_inducing`` is given). Its return type matches what a later online
adaptation scheme's ``final`` step will also produce, so both can feed
``pro_ula``/``parametric_ula`` the same way.
"""

import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from gpjax.objectives import _val
from jax.scipy.linalg import solve_triangular

from non_parametric_pro.density import ProParameters


def _full_gp_basis(
    posterior: gpx.gps.AbstractPosterior,
    x_train: jnp.ndarray,
    jitter: float,
) -> jnp.ndarray:
    """Cholesky basis of the training Gram matrix under the fitted kernel."""
    kernel = posterior.prior.kernel
    k_train = kernel.gram(x_train).as_matrix()
    return jnp.linalg.cholesky(k_train + jitter * jnp.eye(k_train.shape[0]))


def _sparse_gp_basis(
    variational_family: gpx.variational_families.CollapsedVariationalGaussian,
    x_train: jnp.ndarray,
    jitter: float,
) -> jnp.ndarray:
    """
    Basis mapping inducing-point coefficients to training-input function values.

    Models ``f(x) ~= K_xz K_zz^-1 u`` with ``u = L_z @ z``, ``L_z L_z^T = K_zz``,
    so the basis satisfying ``f(x) = basis @ z`` is ``K_xz L_z^-T``.
    """
    kernel = variational_family.posterior.prior.kernel
    z = _val(variational_family.inducing_inputs)
    k_zz = kernel.gram(z).as_matrix()
    k_zx = kernel.cross_covariance(z, x_train)
    l_z = jnp.linalg.cholesky(k_zz + jitter * jnp.eye(k_zz.shape[0]))
    l_z_inv_k_zx = solve_triangular(l_z, k_zx, lower=True)
    return l_z_inv_k_zx.T


def base_gp_adaptation(  # noqa: PLR0913
    rng_key: jr.PRNGKey,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    kernel: gpx.kernels.AbstractKernel,
    *,
    step_size: float,
    alpha: float,
    tolerance: float,
    jitter: float,
    num_inducing: int | None = None,
    optim: ox.GradientTransformation | None = None,
    num_iters: int = 500,
) -> ProParameters:
    """
    Fit a GP to ``(x_train, y_train)`` and seed ``ProParameters`` from it.

    With ``num_inducing=None`` a full GP is fit by maximising the conjugate
    marginal log-likelihood, and ``basis`` is the Cholesky factor of the
    training Gram matrix. With ``num_inducing`` set, a sparse GP (Titsias,
    2009) is fit by maximising the collapsed ELBO over the kernel
    hyperparameters and inducing input locations, and ``basis`` maps the
    (lower-dimensional) inducing-point coefficients to training-input function
    values.
    """
    if optim is None:
        optim = ox.adam(0.01)

    data = gpx.Dataset(X=x_train, y=y_train)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n)
    posterior = prior * likelihood

    if num_inducing is None:
        opt_posterior, _ = gpx.fit(
            model=posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=data,
            optim=optim,
            num_iters=num_iters,
            verbose=False,
        )
        basis = _full_gp_basis(opt_posterior, x_train, jitter)
        nu = _val(opt_posterior.likelihood.obs_stddev).squeeze()
    else:
        inducing_idx = jr.choice(
            rng_key, x_train.shape[0], (num_inducing,), replace=False
        )
        variational_family = gpx.variational_families.CollapsedVariationalGaussian(
            posterior=posterior,
            inducing_inputs=x_train[inducing_idx],
            jitter=jitter,
        )
        opt_variational_family, _ = gpx.fit(
            model=variational_family,
            objective=lambda p, d: -gpx.objectives.collapsed_elbo(p, d),
            train_data=data,
            optim=optim,
            num_iters=num_iters,
            verbose=False,
        )
        basis = _sparse_gp_basis(opt_variational_family, x_train, jitter)
        nu = _val(opt_variational_family.posterior.likelihood.obs_stddev).squeeze()

    return ProParameters(
        y=y_train,
        basis=basis,
        step_size=step_size,
        nu=nu,
        alpha=alpha,
        tolerance=tolerance,
        jitter=jitter,
    )