"""Bayesian overlapping mixture of GPs (OMGP), sampled by Gibbs in the Cholesky or inducing basis."""

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
    train_val_split,
)


class MixtureState(NamedTuple):
    w: jax.Array
    log_pi: jax.Array
    sigma: jax.Array
    counts: jax.Array


class MixtureParameters(NamedTuple):
    y: jax.Array
    basis: jax.Array
    gamma: float
    sigma_prior_shape: float
    sigma_prior_scale: float
    shared_sigma: bool = False
    # Per-datum K(x,x) - Q(x,x) for an inducing basis, treated as extra noise as in PRO.
    residual_var: jax.Array | None = None


class MixtureInfo(NamedTuple):
    log_likelihood: jax.Array


def _noise_var(sigma: jax.Array, parameters: MixtureParameters) -> jax.Array:
    if parameters.residual_var is None:
        return sigma**2
    return sigma[None, :] ** 2 + parameters.residual_var.reshape(-1, 1)


def _log_component_densities(state: MixtureState, parameters: MixtureParameters):
    y = parameters.y.reshape(-1, 1)
    noise_std = jnp.sqrt(_noise_var(state.sigma, parameters))
    return state.log_pi + normal_logpdf(y, parameters.basis @ state.w, noise_std)


def _sample_sigma_mh(key, sigma, onehot, residual, parameters: MixtureParameters):
    """Random-walk MH on log sigma^2 under the IG prior; the inducing residual
    variance breaks conjugacy."""
    a0, b0 = parameters.sigma_prior_shape, parameters.sigma_prior_scale
    residual_var = parameters.residual_var.reshape(-1, 1)

    def log_target(log_s2):
        var = jnp.exp(log_s2)[None, :] + residual_var
        log_lik = jnp.sum(onehot * (-0.5 * jnp.log(var) - 0.5 * residual**2 / var), axis=0)
        if parameters.shared_sigma:
            log_lik = jnp.sum(log_lik, keepdims=True)
        return log_lik - a0 * log_s2 - b0 * jnp.exp(-log_s2)

    log_s2 = jnp.log(sigma**2)
    if parameters.shared_sigma:
        log_s2 = log_s2[:1]
    propose_key, accept_key = jr.split(key)
    proposal = log_s2 + 0.2 * jr.normal(propose_key, log_s2.shape)
    log_ratio = log_target(proposal) - log_target(log_s2)
    accept = jnp.log(jr.uniform(accept_key, log_s2.shape)) < log_ratio
    return jnp.sqrt(jnp.exp(jnp.where(accept, proposal, log_s2)))


def one_step(
    rng_key, state: MixtureState, parameters: MixtureParameters
) -> MixtureState:
    assign_key, w_key, pi_key, sigma_key = jr.split(rng_key, 4)
    y = parameters.y.reshape(-1, 1)
    num_components = state.w.shape[1]

    logits = _log_component_densities(state, parameters)
    onehot = jax.nn.one_hot(jr.categorical(assign_key, logits, axis=1), num_components)
    counts = onehot.sum(axis=0)

    # With sigma=1, _gaussian_block's per-datum weight count * sigma**-2 becomes 1[c_i=k] / sigma_k**2.
    w = _gaussian_block(
        w_key, parameters.basis, y, 1.0, onehot / _noise_var(state.sigma, parameters)
    )

    log_gammas = jr.loggamma(pi_key, parameters.gamma + counts)
    log_pi = log_gammas - logsumexp(log_gammas)

    residual = y - parameters.basis @ w
    if parameters.residual_var is not None:
        sigma = _sample_sigma_mh(sigma_key, state.sigma, onehot, residual, parameters)
        sigma = jnp.broadcast_to(sigma, counts.shape)
        return MixtureState(w=w, log_pi=log_pi, sigma=sigma, counts=counts)

    sigma_counts = counts
    sse = jnp.sum(onehot * residual**2, axis=0)
    if parameters.shared_sigma:
        sigma_counts, sse = jnp.sum(counts, keepdims=True), jnp.sum(sse, keepdims=True)
    shape = parameters.sigma_prior_shape + 0.5 * sigma_counts
    rate = parameters.sigma_prior_scale + 0.5 * sse
    sigma = jnp.broadcast_to(jnp.sqrt(rate / jr.gamma(sigma_key, shape)), counts.shape)

    return MixtureState(w=w, log_pi=log_pi, sigma=sigma, counts=counts)


def mixture_gibbs(parameters: MixtureParameters) -> SamplingAlgorithm:
    def init_fn(position: MixtureState, rng_key=None) -> MixtureState:
        del rng_key
        return position

    def step_fn(rng_key, state: MixtureState):
        new_state = one_step(rng_key, state, parameters)
        log_likelihood = jnp.sum(
            logsumexp(_log_component_densities(new_state, parameters), axis=1)
        )
        return new_state, MixtureInfo(log_likelihood)

    return SamplingAlgorithm(init_fn, step_fn)


@partial(
    jax.jit,
    static_argnames=(
        "num_components", "num_steps", "burn_fraction", "thin", "shared_sigma"
    ),
)
def sample_mixture(
    rng_key,
    basis: jax.Array,
    y: jax.Array,
    *,
    num_components: int,
    gamma: float,
    sigma_init: float,
    sigma_prior_shape: float = 2.0,
    num_steps: int = 2000,
    burn_fraction: float = 0.5,
    thin: int = 10,
    shared_sigma: bool = False,
    residual_var: jax.Array | None = None,
) -> tuple[MixtureState, MixtureInfo]:
    init_key, sample_key = jr.split(rng_key)
    parameters = MixtureParameters(
        y=y,
        basis=basis,
        gamma=gamma,
        sigma_prior_shape=sigma_prior_shape,
        # Prior mean of sigma_k**2 is sigma_init**2.
        sigma_prior_scale=(sigma_prior_shape - 1.0) * sigma_init**2,
        shared_sigma=shared_sigma,
        residual_var=residual_var,
    )
    initial_state = MixtureState(
        w=jr.normal(init_key, (basis.shape[1], num_components)),
        log_pi=jnp.full((num_components,), -jnp.log(num_components)),
        sigma=jnp.full((num_components,), sigma_init),
        counts=jnp.zeros((num_components,)),
    )
    _, (states, infos) = run_inference_algorithm_with_burn_in(
        rng_key=sample_key,
        inference_algorithm=mixture_gibbs(parameters),
        num_steps=num_steps,
        burn_ratio=burn_fraction,
        initial_position=initial_state,
    )
    return jax.tree.map(lambda x: x[::thin], states), infos


def nlpd_mixture(
    y_test: jax.Array,
    test_basis: jax.Array,
    test_covariance: jax.Array,
    samples: MixtureState,
    *,
    return_per_point: bool = False,
) -> jax.Array:
    projected = jnp.einsum("td,sdk->stk", test_basis, samples.w)
    residual_var = jnp.maximum(
        jnp.diag(test_covariance) - jnp.sum(test_basis**2, axis=1), 0.0
    )
    sigma_eff = jnp.sqrt(samples.sigma[:, None, :] ** 2 + residual_var[None, :, None])
    log_components = samples.log_pi[:, None, :] + normal_logpdf(
        y_test.reshape(1, -1, 1), projected, sigma_eff
    )
    log_p = logsumexp(log_components, axis=(0, 2)) - jnp.log(samples.w.shape[0])
    if return_per_point:
        return -log_p
    return -jnp.mean(log_p)


def occupied_components(samples: MixtureState) -> float:
    return float(jnp.mean(jnp.sum(samples.counts > 0, axis=1)))


class MixtureFit(NamedTuple):
    nlpd_per_point: jax.Array
    samples: MixtureState
    infos: MixtureInfo
    test_basis: jax.Array
    test_covariance: jax.Array
    runtime: float


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
) -> MixtureFit:
    start = time.perf_counter()
    if inducing_basis is None:
        basis, residual_var = cholesky_basis(kernel, x_train), None
    else:
        basis, residual_std = compute_inducing_basis(inducing_basis, kernel, x_train)
        residual_var = residual_std**2
    samples, infos = sample_mixture(
        rng_key, basis, y_train, residual_var=residual_var, **sample_kwargs
    )
    test_basis, test_covariance = prediction_basis(
        kernel,
        x_train,
        x_test,
        ProParameters(y=y_train, step_size=None, sigma=None, alpha=None, basis=basis),
        inducing_basis=inducing_basis,
    )
    nlpd_per_point = jax.block_until_ready(
        nlpd_mixture(y_test, test_basis, test_covariance, samples, return_per_point=True)
    )
    return MixtureFit(
        nlpd_per_point=nlpd_per_point,
        samples=samples,
        infos=infos,
        test_basis=test_basis,
        test_covariance=test_covariance,
        runtime=time.perf_counter() - start,
    )


class OmgpResult(NamedTuple):
    nlpd_per_point: jax.Array
    runtime: float
    occupied: float
    selected_k: int | None = None


def fit_omgp_variants(
    rng_key,
    kernel,
    x_train: jax.Array,
    y_train: jax.Array,
    x_test: jax.Array,
    y_test: jax.Array,
    *,
    sigma_init: float,
    ks,
    gamma: float = 1.0,
    sparse_k: int | None = 10,
    sparse_gamma: float = 0.01,
    select_k: bool = True,
    val_fraction: float = 0.3,
    sigma_prior_shape: float = 2.0,
    num_steps: int = 2000,
    burn_fraction: float = 0.5,
    thin: int = 10,
    shared_sigma: bool = False,
    inducing_basis: InducingBasis | None = None,
) -> dict[str, OmgpResult]:
    """`omgp_k{K}` for each K in `ks`, `omgp_sparse` (K=sparse_k, Dir(sparse_gamma)),
    and `omgp_valk`, the full-data fit of the K with the best validation NLPD. With
    `shared_sigma` the components share one noise scale and names start `omgp_shared_`."""
    prefix = "omgp_shared" if shared_sigma else "omgp"
    sample_kwargs = {
        "sigma_init": sigma_init,
        "sigma_prior_shape": sigma_prior_shape,
        "num_steps": num_steps,
        "burn_fraction": burn_fraction,
        "thin": thin,
        "shared_sigma": shared_sigma,
    }
    full_key, sparse_key, split_key, val_key = jr.split(rng_key, 4)

    def fit(key, k, g, x_tr, y_tr, x_te, y_te):
        return fit_mixture_gp(
            key, kernel, x_tr, y_tr, x_te, y_te, inducing_basis=inducing_basis,
            num_components=int(k), gamma=g, **sample_kwargs,
        )

    results = {}
    for k, key in zip(ks, jr.split(full_key, len(ks)), strict=True):
        result = fit(key, k, gamma, x_train, y_train, x_test, y_test)
        results[f"{prefix}_k{k}"] = OmgpResult(
            result.nlpd_per_point, result.runtime, occupied_components(result.samples)
        )

    if sparse_k is not None:
        result = fit(sparse_key, sparse_k, sparse_gamma, x_train, y_train, x_test, y_test)
        results[f"{prefix}_sparse"] = OmgpResult(
            result.nlpd_per_point, result.runtime, occupied_components(result.samples)
        )

    if select_k:
        split = train_val_split(split_key, x_train, y_train, val_fraction=val_fraction)
        val_nlpds, val_runtime = [], 0.0
        for k, key in zip(ks, jr.split(val_key, len(ks)), strict=True):
            result = fit(key, k, gamma, split.x_train, split.y_train, split.x_val, split.y_val)
            val_nlpds.append(float(jnp.mean(result.nlpd_per_point)))
            val_runtime += result.runtime
        best_k = ks[int(jnp.argmin(jnp.array(val_nlpds)))]
        chosen = results[f"{prefix}_k{best_k}"]
        results[f"{prefix}_valk"] = OmgpResult(
            chosen.nlpd_per_point,
            val_runtime + chosen.runtime,
            chosen.occupied,
            selected_k=int(best_k),
        )

    return results
