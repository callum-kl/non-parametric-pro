from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import paramax
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import PRNGKey
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


def draw_predictive_samples(
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


def cholesky_basis(
    kernel: gpx.kernels.AbstractKernel, x: jax.Array, jitter: float = 1e-6
) -> jax.Array:
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
        L_zz = jnp.linalg.cholesky(K_zz + parameters.jitter * jnp.eye(K_zz.shape[0]))
        K_z_test = inducing_basis.cross_cov(k, x_te)  # (M, N_test)
        test_basis = solve_triangular(L_zz, K_z_test, lower=True).T  # (N_test, M)

    return test_basis, test_covariance


def rmse(
    y_true: jax.Array,
    y_pred: jax.Array,
    *,
    return_per_point: bool = False,
) -> jax.Array:
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    squared_error = (y_true - y_pred) ** 2
    if return_per_point:
        return jnp.sqrt(squared_error)
    return jnp.sqrt(jnp.mean(squared_error))


def energy_score(
    draws: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Energy score (Gneiting & Raftery, 2007) of joint predictive draws against y.

    ES = E||X - y|| - 0.5*E||X - X'||, estimated from `draws` (shape (n, M): M iid
    draws of the n-dimensional joint predictive distribution, e.g. from
    posterior_function_draws or a Gaussian's Cholesky factor). A strictly proper
    scoring rule for the *entire* joint predictive distribution -- computed directly
    from samples, so it needs no Gaussian/parametric assumption and no single draw
    has to exactly explain y (unlike a mixture log-score). Lower is better.
    Generalizes univariate CRPS to joint/multivariate predictive distributions.
    """
    diff_to_y = draws - y[:, None]
    term1 = jnp.mean(jnp.linalg.norm(diff_to_y, axis=0))
    diff_pairs = draws[:, :, None] - draws[:, None, :]
    term2 = jnp.mean(jnp.linalg.norm(diff_pairs, axis=0))
    return term1 - 0.5 * term2


def crps(
    draws: jax.Array,
    y: jax.Array,
    *,
    return_per_point: bool = False,
) -> jax.Array:
    """CRPS (Continuous Ranked Probability Score) at each test point, from samples.

    CRPS_i = E|X_i - y_i| - 0.5*E|X_i - X_i'|, the univariate special case of
    energy_score applied independently per point (no cross-point/joint structure,
    unlike energy_score). A proper scoring rule with no 1/sigma^2 term, so it's more
    robust to tail misspecification than a log-density score (nlpd_gp/nlpd_pro).
    `draws` has shape (n, M): the same per-point marginals used for energy_score
    work directly here. Lower is better.
    """
    term1 = jnp.mean(jnp.abs(draws - y[:, None]), axis=1)
    diff_pairs = draws[:, :, None] - draws[:, None, :]
    term2 = jnp.mean(jnp.abs(diff_pairs), axis=(1, 2))
    per_point = term1 - 0.5 * term2
    if return_per_point:
        return per_point
    return jnp.mean(per_point)


def variogram_score(
    draws: jax.Array,
    y: jax.Array,
    *,
    p: float = 1.0,
    standardize: bool = True,
) -> jax.Array:
    """Variogram score (Scheuerer & Hamill, 2015), order `p`, uniform pair weights.

    VS = sum_{i,j} (|y_i - y_j|^p - E_F|X_i - X_j|^p)^2, estimated from `draws`
    (shape (n, M)). Scores *pairwise relative* differences rather than absolute
    levels -- suited to spatial/graph fields, where getting the relative pattern
    right (which points are higher/lower than which) matters independently of a
    uniform bias in overall level. Lower is better.

    The raw sum scales with the number of pairs (~n^2), so it isn't comparable
    across test sets of different sizes -- e.g. a handful of in-region points vs.
    dozens out-of-region. `standardize=True` (default) divides by the number of
    off-diagonal pairs n*(n-1), giving the *mean* squared discrepancy per pair
    instead, which is directly comparable across n. Returns 0 (no off-diagonal
    pairs to average) rather than dividing by zero when n<=1.
    """
    n = y.shape[0]
    y_diff = jnp.abs(y[:, None] - y[None, :]) ** p
    draw_diff = jnp.abs(draws[:, None, :] - draws[None, :, :]) ** p
    expected_diff = jnp.mean(draw_diff, axis=2)
    total = jnp.sum((y_diff - expected_diff) ** 2)
    if not standardize:
        return total
    return total / (n * (n - 1)) if n > 1 else jnp.asarray(0.0)


def pit_values(draws: jax.Array, y: jax.Array) -> jax.Array:
    """Empirical PIT (probability integral transform) at each test point.

    PIT_i = (1/M) * sum_m 1[draws[i, m] <= y_i] -- the empirical predictive CDF
    evaluated at the observation. Should be ~Uniform(0,1) across test points if the
    (marginal, per-point) predictive distribution is calibrated: values clustering
    near 0/1 indicate under-dispersion (overconfident), a hump in the middle
    indicates over-dispersion. `draws` has shape (n, M). Meant to be pooled across
    many test points/splits and viewed as a histogram, not read as a single number.
    """
    return jnp.mean(draws <= y[:, None], axis=1)


def gaussian_nll(
    y: jax.Array,
    mean: jax.Array,
    cov: jax.Array,
    *,
    jitter: float = 1e-6,
) -> jax.Array:
    """-log N(y; mean, cov) -- a single (non-mixture) joint Gaussian NLL.

    Shared by exact-GP scripts (full predictive covariance) and PrO's mixture
    machinery (per-particle / moment-matched covariance) so both use the exact same
    formula -- including for region-subset evaluation, where `cov` is just a
    row/column slice of an already-computed full covariance matrix.
    """
    n = cov.shape[0]
    chol = jnp.linalg.cholesky(cov + jitter * jnp.eye(n))
    alpha = solve_triangular(chol, y - mean, lower=True)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    return 0.5 * (n * jnp.log(2.0 * jnp.pi) + log_det + jnp.sum(alpha**2))


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
    log_p = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(std) - 0.5 * ((y - mu) / std) ** 2
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


def run_inference_algorithm_with_burn_in(
    rng_key,
    inference_algorithm,
    num_steps,
    burn_ratio,
    initial_position,
    progress_bar=False,
):
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
