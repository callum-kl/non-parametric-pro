from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import paramax
from blackjax.types import PRNGKey
from blackjax.progress_bar import gen_scan_fn
from blackjax.util import run_inference_algorithm
from jax.scipy.linalg import solve_triangular

from non_parametric_pro.density import ProParameters
from non_parametric_pro.inducing import InducingBasis, RFFInducingBasis

PARTICLE_MATRIX_NDIM = 2
SCANNED_PARTICLES_NDIM = 3

class TrainValSplit(NamedTuple):
    x_train: jax.Array
    y_train: jax.Array
    x_val: jax.Array
    y_val: jax.Array
    train_idx: jax.Array
    val_idx: jax.Array

def train_val_split(
    key: PRNGKey,
    x: jax.Array,
    y: jax.Array,
    *,
    val_fraction: float = 0.2,
) -> TrainValSplit:
    n = y.shape[0]
    idx = jr.permutation(key, n)

    n_val = round(val_fraction * n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return TrainValSplit(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_val=x[val_idx],
        y_val=y[val_idx],
        train_idx=train_idx,
        val_idx=val_idx,
    )


def _as_particle_matrix(particles: jax.Array) -> jax.Array:
    if particles.ndim == PARTICLE_MATRIX_NDIM:
        return particles
    if particles.ndim == SCANNED_PARTICLES_NDIM:
        return jnp.transpose(particles, (1, 0, 2)).reshape(particles.shape[1], -1)
    msg = (
        "particles must have shape (basis_dim, num_particles) or "
        "(draws, basis_dim, particles)"
    )
    raise ValueError(msg)


def project_particles(basis: jax.Array, particles: jax.Array) -> jax.Array:
    return basis @ _as_particle_matrix(particles)


def predictive_moments(
    basis: jax.Array,
    particles: jax.Array,
    *,
    noise_std: jax.Array | float = 0.0,
    residual_std: jax.Array | float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    projected = project_particles(basis, particles)
    mean = jnp.mean(projected, axis=1)
    particle_variance = jnp.mean((projected - mean[:, None]) ** 2, axis=1)
    variance = particle_variance + noise_std**2 + residual_std**2
    return mean, jnp.sqrt(jnp.maximum(variance, 0.0))


def draw_predictive_samples(  # noqa: PLR0913
    rng_key: PRNGKey,
    basis: jax.Array,
    particles: jax.Array,
    *,
    num_samples: int,
    noise_std: jax.Array | float = 0.0,
    residual_std: jax.Array | float = 0.0,
) -> jax.Array:
    projected = project_particles(basis, particles)
    num_particles = projected.shape[1]
    index_key, noise_key = jr.split(rng_key)
    particle_indices = jr.randint(index_key, (num_samples,), 0, num_particles)
    selected = projected[:, particle_indices]

    sample_std = jnp.sqrt(noise_std**2 + residual_std**2)
    if jnp.ndim(sample_std) == 1:
        sample_std = sample_std[:, None]
    noise = sample_std * jr.normal(noise_key, selected.shape)
    return selected + noise

def cholesky_basis(kernel: gpx.kernels.AbstractKernel, x: jax.Array, jitter: float = 1e-6) -> jax.Array:
    k = kernel.gram(x).as_matrix()
    return jnp.linalg.cholesky(k + jitter * jnp.eye(k.shape[0]))

def prediction_basis(
    kernel: gpx.kernels.AbstractKernel,
    x_train: jax.Array,
    x_test: jax.Array,
    parameters: ProParameters,
    *,
    inducing_basis: InducingBasis | None = None,
) -> tuple[jax.Array, jax.Array]:
    k = paramax.unwrap(kernel)
    x_tr = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
    x_te = x_test.reshape(-1, 1) if x_test.ndim == 1 else x_test
    test_covariance = k.gram(x_te).as_matrix()

    if inducing_basis is None:
        L = parameters.basis  # (N, N) — the training Cholesky
        K_test_train = k.cross_covariance(x_te, x_tr)  # (N_test, N)
        test_basis = solve_triangular(L, K_test_train.T, lower=True).T
    elif isinstance(inducing_basis, RFFInducingBasis):
        test_basis = inducing_basis.basis_matrix(k, x_te, parameters.jitter)
    else:
        K_zz = inducing_basis.inducing_cov(k)  # (M, M)
        L_zz = jnp.linalg.cholesky(
            K_zz + parameters.jitter * jnp.eye(K_zz.shape[0])
        )
        K_z_test = inducing_basis.cross_cov(k, x_te)  # (M, N_test)
        test_basis = solve_triangular(L_zz, K_z_test, lower=True).T  # (N_test, M)

    return test_basis, test_covariance

def nlpd_gp(
    y_test: jax.Array,
    predictive_mean: jax.Array,
    predictive_std: jax.Array,
    *,
    return_per_point: bool = False,
) -> jax.Array:
    y = y_test.squeeze()
    mu = predictive_mean.squeeze()
    std = predictive_std.squeeze()
    log_p = (
        -0.5 * jnp.log(2 * jnp.pi)
        - jnp.log(std)
        - 0.5 * ((y - mu) / std) ** 2
    )
    if return_per_point:
        return -log_p

    return -jnp.mean(log_p)


def nlpd_pro(
    y_test: jax.Array,
    test_basis: jax.Array,
    test_covariance: jax.Array,
    particles: jax.Array,
    *,
    parameters: ProParameters,
    return_per_point: bool = False,
) -> jax.Array:
    sigma_val = paramax.unwrap(parameters.sigma)
    particle_matrix = _as_particle_matrix(particles)
    projected = test_basis @ particle_matrix
    j_total = projected.shape[1]

    q_diag = jnp.sum(test_basis**2, axis=1)
    k_diag = jnp.diag(test_covariance)
    test_residual_std = jnp.sqrt(jnp.maximum(k_diag - q_diag, 0.0))
    sigma_eff = jnp.sqrt(sigma_val**2 + test_residual_std**2).reshape(-1, 1)

    y = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
    log_normals = (
        -0.5 * jnp.log(2 * jnp.pi)
        - jnp.log(sigma_eff)
        - 0.5 * ((y - projected) / sigma_eff) ** 2
    )
    log_p = jax.nn.logsumexp(log_normals, axis=1) - jnp.log(j_total)

    if return_per_point:
        return -log_p

    return -jnp.mean(log_p)


def posterior_function_draws(
    rng_key: PRNGKey,
    test_basis: jax.Array,
    test_covariance: jax.Array,
    particles: jax.Array,
    *,
    num_draws: int = 120,
    jitter: float = 1e-6,
) -> jax.Array:
    particle_matrix = _as_particle_matrix(particles)
    num_test = test_covariance.shape[0]

    index_key, noise_key = jr.split(rng_key)
    particle_indices = jr.randint(index_key, (num_draws,), 0, particle_matrix.shape[1])
    conditional_means = test_basis @ particle_matrix[:, particle_indices]

    conditional_covariance = test_covariance - test_basis @ test_basis.T
    conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
    eigvals, eigvecs = jnp.linalg.eigh(conditional_covariance)
    eigvals = jnp.maximum(eigvals, 0.0) + jitter
    residual_cholesky = eigvecs * jnp.sqrt(eigvals)[None, :]
    residual_noise = jr.normal(noise_key, (num_test, num_draws))
    return conditional_means + residual_cholesky @ residual_noise



def run_inference_algorithm_with_burn_in(rng_key, inference_algorithm, num_steps, burn_ratio, initial_position, progress_bar=False):
    burn_key, sample_key = jr.split(rng_key)
    num_burn_steps = int(num_steps * burn_ratio)
    num_sample_steps = num_steps - num_burn_steps

    state = inference_algorithm.init(initial_position)
    burn_keys = jr.split(burn_key, num_burn_steps)
    burn_scan_fn = gen_scan_fn(num_burn_steps, progress_bar=progress_bar)

    def burn_step(state, xs):
        _, key = xs
        state, _ = inference_algorithm.step(key, state)
        return state, None

    state, _ = burn_scan_fn(burn_step, state, (jnp.arange(num_burn_steps), burn_keys))
    final_state, (states, infos) = run_inference_algorithm(
        rng_key=sample_key,
        inference_algorithm=inference_algorithm,
        num_steps=num_sample_steps,
        initial_state=state,
        progress_bar=progress_bar,
    )
    return final_state, (states, infos)