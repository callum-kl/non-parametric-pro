"""
Bayesian overlapping mixture of GPs (OMGP) with a noise scale shared across components,
sampled by Gibbs in the Cholesky or inducing basis of a fixed GP kernel.
"""

import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.base import SamplingAlgorithm
from jax.scipy.special import logsumexp

from non_parametric_pro.density import ProParameters, normal_logpdf
from non_parametric_pro.inducing import InducingBasis, compute_inducing_basis
from non_parametric_pro.replica_gibbs import _gaussian_block
from non_parametric_pro.util import (
    cholesky_basis,
    prediction_basis,
    run_inference_algorithm_with_burn_in,
)

TEST_CHUNK = 1024
TRAIN_CHUNK = 4096


class MixtureState(NamedTuple):
    w: jax.Array
    log_pi: jax.Array
    # One shared scale, stored per component.
    sigma: jax.Array


class MixtureParameters(NamedTuple):
    y: jax.Array
    basis: jax.Array
    gamma: float
    sigma_prior_shape: float
    sigma_prior_scale: float
    # Per-datum K(x,x) - Q(x,x) for an inducing basis, treated as extra noise as in PRO.
    residual_var: jax.Array | None = None


def _noise_var(sigma: jax.Array, parameters: MixtureParameters) -> jax.Array:
    if parameters.residual_var is None:
        return sigma**2
    return sigma[None, :] ** 2 + parameters.residual_var.reshape(-1, 1)


def _sample_sigma_mh(key, sigma, onehot, residual, parameters: MixtureParameters):
    """Random-walk MH on log sigma^2 under the IG prior; the inducing residual
    variance breaks conjugacy."""
    a0, b0 = parameters.sigma_prior_shape, parameters.sigma_prior_scale
    residual_var = parameters.residual_var.reshape(-1, 1)

    def log_target(log_s2):
        var = jnp.exp(log_s2)[None, :] + residual_var
        log_lik = jnp.sum(onehot * (-0.5 * jnp.log(var) - 0.5 * residual**2 / var), axis=0)
        return jnp.sum(log_lik, keepdims=True) - a0 * log_s2 - b0 * jnp.exp(-log_s2)

    log_s2 = jnp.log(sigma[:1] ** 2)
    propose_key, accept_key = jr.split(key)
    proposal = log_s2 + 0.2 * jr.normal(propose_key, log_s2.shape)
    log_ratio = log_target(proposal) - log_target(log_s2)
    accept = jnp.log(jr.uniform(accept_key, log_s2.shape)) < log_ratio
    return jnp.sqrt(jnp.exp(jnp.where(accept, proposal, log_s2)))


def one_step(rng_key, state: MixtureState, parameters: MixtureParameters) -> MixtureState:
    assign_key, w_key, pi_key, sigma_key = jr.split(rng_key, 4)
    y = parameters.y.reshape(-1, 1)
    num_components = state.w.shape[1]

    noise_std = jnp.sqrt(_noise_var(state.sigma, parameters))
    logits = state.log_pi + normal_logpdf(y, parameters.basis @ state.w, noise_std)
    onehot = jax.nn.one_hot(jr.categorical(assign_key, logits, axis=1), num_components)
    counts = onehot.sum(axis=0)

    # With sigma=1, _gaussian_block's per-datum weight count * sigma**-2 becomes 1[c_i=k] / sigma**2.
    w = _gaussian_block(
        w_key, parameters.basis, y, 1.0, onehot / _noise_var(state.sigma, parameters)
    )

    log_gammas = jr.loggamma(pi_key, parameters.gamma + counts)
    log_pi = log_gammas - logsumexp(log_gammas)

    residual = y - parameters.basis @ w
    if parameters.residual_var is not None:
        sigma = _sample_sigma_mh(sigma_key, state.sigma, onehot, residual, parameters)
    else:
        sse = jnp.sum(jnp.sum(onehot * residual**2, axis=0), keepdims=True)
        shape = parameters.sigma_prior_shape + 0.5 * jnp.sum(counts, keepdims=True)
        rate = parameters.sigma_prior_scale + 0.5 * sse
        sigma = jnp.sqrt(rate / jr.gamma(sigma_key, shape))
    return MixtureState(w=w, log_pi=log_pi, sigma=jnp.broadcast_to(sigma, counts.shape))


def mixture_gibbs(parameters: MixtureParameters) -> SamplingAlgorithm:
    def init_fn(position: MixtureState, rng_key=None) -> MixtureState:
        del rng_key
        return position

    def step_fn(rng_key, state: MixtureState):
        return one_step(rng_key, state, parameters), None

    return SamplingAlgorithm(init_fn, step_fn)


@partial(jax.jit, static_argnames=("num_components", "num_steps", "burn_fraction", "thin"))
def sample_mixture(
    rng_key,
    basis: jax.Array,
    y: jax.Array,
    *,
    num_components: int,
    gamma: float,
    sigma_init: float,
    sigma_prior_shape: float,
    num_steps: int,
    burn_fraction: float,
    thin: int,
    residual_var: jax.Array | None = None,
) -> MixtureState:
    init_key, sample_key = jr.split(rng_key)
    parameters = MixtureParameters(
        y=y,
        basis=basis,
        gamma=gamma,
        sigma_prior_shape=sigma_prior_shape,
        # Prior mean of sigma**2 is sigma_init**2.
        sigma_prior_scale=(sigma_prior_shape - 1.0) * sigma_init**2,
        residual_var=residual_var,
    )
    initial_state = MixtureState(
        w=jr.normal(init_key, (basis.shape[1], num_components)),
        log_pi=jnp.full((num_components,), -jnp.log(num_components)),
        sigma=jnp.full((num_components,), sigma_init),
    )
    _, (states, _) = run_inference_algorithm_with_burn_in(
        rng_key=sample_key,
        inference_algorithm=mixture_gibbs(parameters),
        num_steps=num_steps,
        burn_ratio=burn_fraction,
        initial_position=initial_state,
    )
    return jax.tree.map(lambda x: x[::thin], states)


def nlpd_mixture(
    y_test: jax.Array, test_basis: jax.Array, test_prior_var: jax.Array, samples: MixtureState
) -> jax.Array:
    """Per-point NLPD; `test_prior_var` is the diagonal of the prior covariance K(x*, x*)."""
    projected = jnp.einsum("td,sdk->stk", test_basis, samples.w)
    residual_var = jnp.maximum(test_prior_var - jnp.sum(test_basis**2, axis=1), 0.0)
    sigma_eff = jnp.sqrt(samples.sigma[:, None, :] ** 2 + residual_var[None, :, None])
    log_components = samples.log_pi[:, None, :] + normal_logpdf(
        y_test.reshape(1, -1, 1), projected, sigma_eff
    )
    return -(logsumexp(log_components, axis=(0, 2)) - jnp.log(samples.w.shape[0]))


def _chunked_inducing_basis(inducing_basis, kernel, x_train):
    """Rows of the inducing basis depend only on their own input, so build it in chunks
    rather than materialising the full train-inducing kernel intermediates at once."""
    bases, residual_stds = [], []
    for start in range(0, x_train.shape[0], TRAIN_CHUNK):
        basis, residual_std = compute_inducing_basis(
            inducing_basis, kernel, x_train[start : start + TRAIN_CHUNK]
        )
        bases.append(basis)
        residual_stds.append(residual_std)
    return jnp.concatenate(bases), jnp.concatenate(residual_stds)


def _test_basis_and_prior_var(kernel, x_train, x_test, parameters, inducing_basis):
    """Chunked over test points so the dense test-test Gram matrix, of which only the
    diagonal is needed, is never built in full."""
    bases, prior_vars = [], []
    for start in range(0, x_test.shape[0], TEST_CHUNK):
        basis, covariance = prediction_basis(
            kernel, x_train, x_test[start : start + TEST_CHUNK], parameters,
            inducing_basis=inducing_basis,
        )
        bases.append(basis)
        prior_vars.append(jnp.diag(covariance))
    return jnp.concatenate(bases), jnp.concatenate(prior_vars)


def fit_mixture_gp(
    rng_key,
    kernel,
    x_train: jax.Array,
    y_train: jax.Array,
    x_test: jax.Array,
    y_test: jax.Array,
    *,
    inducing_basis: InducingBasis | None = None,
    **sample_kwargs,
) -> tuple[jax.Array, float]:
    """Fit one OMGP and return (per-point test NLPD, runtime in seconds)."""
    start = time.perf_counter()
    if inducing_basis is None:
        basis, residual_var = cholesky_basis(kernel, x_train), None
    else:
        basis, residual_std = _chunked_inducing_basis(inducing_basis, kernel, x_train)
        residual_var = residual_std**2
    samples = sample_mixture(rng_key, basis, y_train, residual_var=residual_var, **sample_kwargs)
    test_basis, test_prior_var = _test_basis_and_prior_var(
        kernel,
        x_train,
        x_test,
        ProParameters(y=y_train, step_size=None, sigma=None, alpha=None, basis=basis),
        inducing_basis,
    )
    nlpd = jax.block_until_ready(nlpd_mixture(y_test, test_basis, test_prior_var, samples))
    return nlpd, time.perf_counter() - start


def fit_omgp(
    rng_key,
    kernel,
    x_train: jax.Array,
    y_train: jax.Array,
    x_test: jax.Array,
    y_test: jax.Array,
    *,
    ks,
    sigma_init: float,
    gamma: float,
    sigma_prior_shape: float,
    num_steps: int,
    burn_fraction: float,
    thin: int,
    inducing_basis: InducingBasis | None = None,
) -> dict[int, tuple[jax.Array, float]]:
    """OMGP with each number of components in `ks`: {K: (per-point NLPD, runtime)}."""
    # The first of four splits, so the stored results are reproduced exactly.
    full_key = jr.split(rng_key, 4)[0]
    return {
        int(k): fit_mixture_gp(
            key, kernel, x_train, y_train, x_test, y_test, inducing_basis=inducing_basis,
            num_components=int(k), gamma=gamma, sigma_init=sigma_init,
            sigma_prior_shape=sigma_prior_shape, num_steps=num_steps,
            burn_fraction=burn_fraction, thin=thin,
        )
        for k, key in zip(ks, jr.split(full_key, len(ks)), strict=True)
    }
