"""Utilities for posterior predictive summaries."""
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
    """A random train/validation split of ``(x, y)``."""

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
    """Split ``(x, y)`` into a random train/validation partition."""
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
    """Return particles as a ``basis_dim x num_particles`` matrix."""
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
    """Project latent particles through a prediction basis matrix."""
    return basis @ _as_particle_matrix(particles)


def predictive_moments(
    basis: jax.Array,
    particles: jax.Array,
    *,
    noise_std: jax.Array | float = 0.0,
    residual_std: jax.Array | float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute predictive mean and standard deviation from retained particles.

    ``basis`` should have shape ``(num_points, basis_dim)``. ``particles`` can
    either have shape ``(basis_dim, num_particles)`` or the scanned shape
    ``(num_draws, basis_dim, num_particles)``.
    """
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
    """
    Draw posterior predictive samples from retained latent particles.

    The returned array has shape ``(num_points, num_samples)``. Each sample
    chooses one retained particle and adds Gaussian observation/residual noise.
    """
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
    """
    Compute ``(test_basis, test_covariance)`` for ``posterior_function_draws``.

    Both cases share the same formula ``B* = K(·, anchors) L⁻ᵀ``; what differs
    is the choice of anchors and the Cholesky ``L``:

    - **Full GP** (``inducing_basis=None``): anchors are the ``N`` training
      inputs; ``L`` is ``parameters.basis`` (the ``N×N`` training Cholesky).
    - **Inducing** (``inducing_basis`` given): anchors are the ``M`` inducing
      features; ``L_zz`` is recomputed from the kernel and
      ``inducing_basis.inducing_cov``.

    The returned ``test_covariance`` is ``K(x_test, x_test)`` in both cases.
    ``posterior_function_draws`` subtracts ``test_basis @ test_basis.T`` from
    it to get the residual epistemic uncertainty at test points.

    Parameters
    ----------
    kernel
        The fitted/adapted kernel (paramax-wrapped or plain).
    x_train
        Training inputs ``(N, D)``. Only used in the full GP case.
    x_test
        Test inputs ``(N_test, D)``.
    parameters
        Current ``ProParameters``; ``parameters.jitter`` is reused when
        recomputing ``L_zz`` for the inducing case.
    inducing_basis
        An :class:`InducingBasis` instance, or ``None`` for the full GP path.
    """
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


def nll_gp(
    y_test: jax.Array,
    predictive_mean: jax.Array,
    sigma: jax.Array | float,
) -> jax.Array:
    """
    Mean negative log likelihood for a GP under the observation noise only.

    Unlike :func:`nlpd_gp`, this uses ``sigma`` (aleatoric noise) rather than
    the full predictive standard deviation, treating the predictive mean as a
    point estimate of the latent function.  Useful for isolating noise
    estimation quality from epistemic uncertainty.

    Parameters
    ----------
    y_test
        Observed test targets, shape ``(N_test,)`` or ``(N_test, 1)``.
    predictive_mean
        GP posterior predictive mean, same shape as ``y_test``.
    sigma
        Observation noise standard deviation (plain array or
        ``paramax``-wrapped; unwrapped automatically).

    Returns
    -------
    Scalar NLL averaged over test points.
    """
    sigma_val = paramax.unwrap(sigma)
    y = y_test.squeeze()
    mu = predictive_mean.squeeze()
    log_p = (
        -0.5 * jnp.log(2 * jnp.pi)
        - jnp.log(sigma_val)
        - 0.5 * ((y - mu) / sigma_val) ** 2
    )
    return -jnp.mean(log_p)


def nll_pro(
    y_test: jax.Array,
    test_basis: jax.Array,
    particles: jax.Array,
    *,
    sigma: jax.Array | float,
) -> jax.Array:
    """
    Mean negative log likelihood for the PRO GP mixture under observation noise only.

    Like :func:`nlpd_pro` but without the sparse-approximation residual
    inflation — uses ``sigma`` alone as the component standard deviation.
    Isolates noise estimation quality from the inducing approximation error.

    Parameters
    ----------
    y_test
        Observed test targets, shape ``(N_test, 1)``.
    test_basis
        Test projection matrix from :func:`prediction_basis`,
        shape ``(N_test, M)`` (inducing) or ``(N_test, N)`` (full GP).
    particles
        Retained latent particles, shape ``(M, J)`` or
        ``(num_steps, M, J)``.
    sigma
        Observation noise standard deviation (plain array or
        ``paramax``-wrapped; unwrapped automatically).

    Returns
    -------
    Scalar NLL averaged over test points.
    """
    sigma_val = paramax.unwrap(sigma)
    particle_matrix = _as_particle_matrix(particles)        # (M, J_total)
    projected = test_basis @ particle_matrix                # (N_test, J_total)
    j_total = projected.shape[1]

    y = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
    log_normals = (
        -0.5 * jnp.log(2 * jnp.pi)
        - jnp.log(sigma_val)
        - 0.5 * ((y - projected) / sigma_val) ** 2
    )                                                       # (N_test, J_total)
    log_p = jax.nn.logsumexp(log_normals, axis=1) - jnp.log(j_total)
    return -jnp.mean(log_p)


def _g(z: jax.Array) -> jax.Array:
    """E[|X|] for X ~ N(z, 1); kernel of the closed-form CRPS/absolute-difference."""
    return 2 * jax.scipy.stats.norm.pdf(z) + z * (2 * jax.scipy.stats.norm.cdf(z) - 1)


def _gaussian_crps(y: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    """Closed-form CRPS for a single Gaussian predictive."""
    z = (y - mean) / std
    return std * (_g(z) - 1 / jnp.sqrt(jnp.pi))


def crps_gp(
    y_test: jax.Array,
    predictive_mean: jax.Array,
    predictive_std: jax.Array,
) -> jax.Array:
    """
    Mean CRPS for a Gaussian GP predictive (lower is better).

    Parameters
    ----------
    y_test
        Observed test targets, shape ``(N_test,)`` or ``(N_test, 1)``.
    predictive_mean
        GP predictive mean, same shape as ``y_test``.
    predictive_std
        GP predictive standard deviation (including observation noise).

    Returns
    -------
    Scalar CRPS averaged over test points.
    """
    y = y_test.squeeze()
    mu = predictive_mean.squeeze()
    std = predictive_std.squeeze()
    return jnp.mean(_gaussian_crps(y, mu, std))


def crps_pro(
    y_test: jax.Array,
    test_basis: jax.Array,
    test_covariance: jax.Array,
    particles: jax.Array,
    *,
    parameters: "ProParameters",
) -> jax.Array:
    """
    Mean CRPS for the PRO GP mixture predictive (lower is better).

    Evaluates the exact closed-form CRPS of the Gaussian mixture
    ``(1/J) Σ_j N(B* z_j, σ_eff²)`` using the identity:

    .. math::

        \\text{CRPS}(F_{\\text{mix}}, y)
            = \\frac{\\sigma_{\\text{eff}}}{J} \\sum_j g(z_j)
            - \\frac{\\sigma_{\\text{eff}} \\sqrt{2}}{2J^2}
              \\sum_j \\sum_k g\\!\\left(\\frac{f_j - f_k}{\\sigma_{\\text{eff}}\\sqrt{2}}\\right)

    where :math:`g(z) = 2\\phi(z) + z(2\\Phi(z)-1)` and
    :math:`z_j = (y - f_j)/\\sigma_{\\text{eff}}`.  The cross term is
    :math:`O(J^2)` per test point but deterministic.

    Parameters
    ----------
    y_test
        Observed test targets, shape ``(N_test, 1)``.
    test_basis
        Test projection matrix from :func:`prediction_basis`,
        shape ``(N_test, M)`` (inducing) or ``(N_test, N)`` (full GP).
    test_covariance
        Prior kernel matrix at test points ``K(x_*, x_*)``,
        shape ``(N_test, N_test)``, also from :func:`prediction_basis`.
    particles
        Retained latent particles, shape ``(M, J)`` or
        ``(num_steps, M, J)``.
    parameters
        Current :class:`ProParameters`; ``sigma`` and ``residual_std`` are
        used to compute ``sigma_eff``.

    Returns
    -------
    Scalar CRPS averaged over test points.
    """
    sigma_val = paramax.unwrap(parameters.sigma)
    particle_matrix = _as_particle_matrix(particles)        # (M, J)
    projected = test_basis @ particle_matrix                # (N_test, J)
    q_diag = jnp.sum(test_basis**2, axis=1)                # (N_test,)
    k_diag = jnp.diag(test_covariance)                     # (N_test,)
    test_residual_std = jnp.sqrt(jnp.maximum(k_diag - q_diag, 0.0))
    sigma_eff = jnp.sqrt(sigma_val**2 + test_residual_std**2)  # (N_test,)

    y = y_test.squeeze()                                    # (N_test,)
    s = sigma_eff.reshape(-1, 1)                            # (N_test, 1)

    # First term: (σ_eff / J) Σ_j g(z_j), z_j = (y - f_j) / σ_eff
    z = (y.reshape(-1, 1) - projected) / s                 # (N_test, J)
    first = jnp.mean(_g(z), axis=1) * sigma_eff            # (N_test,)

    # Cross term: (σ_eff√2 / 2J²) Σ_j Σ_k g(d_jk / (σ_eff√2)).
    # Map over test points to avoid materialising an (N_test, J, J) array.
    sqrt_two = jnp.sqrt(jnp.asarray(2.0, dtype=projected.dtype))

    def cross_one(args: tuple[jax.Array, jax.Array]) -> jax.Array:
        f_i, s_i = args
        d_norm = (f_i[:, None] - f_i[None, :]) / (s_i * sqrt_two)
        return jnp.mean(_g(d_norm)) * s_i * sqrt_two / 2

    cross = jax.lax.map(cross_one, (projected, sigma_eff))  # (N_test,)

    return jnp.mean(first - cross)


def pit_gp(
    y_test: jax.Array,
    predictive_mean: jax.Array,
    predictive_std: jax.Array,
) -> jax.Array:
    """
    Probability integral transform values for a Gaussian GP predictive.

    Returns one PIT value per test point:
    ``F_i(y_i)`` where ``F_i`` is the Gaussian predictive CDF at test point
    ``i``. For calibrated scalar predictive distributions, these values should
    be approximately uniform on ``[0, 1]``.

    Parameters
    ----------
    y_test
        Observed test targets, shape ``(N_test,)`` or ``(N_test, 1)``.
    predictive_mean
        GP predictive mean, same shape as ``y_test``.
    predictive_std
        GP predictive standard deviation (including observation noise),
        same shape as ``y_test``.

    Returns
    -------
    PIT values, shape ``(N_test,)``.
    """
    y = y_test.squeeze()
    mu = predictive_mean.squeeze()
    std = predictive_std.squeeze()
    return jax.scipy.stats.norm.cdf((y - mu) / std)


def pit_pro(
    y_test: jax.Array,
    test_basis: jax.Array,
    test_covariance: jax.Array,
    particles: jax.Array,
    *,
    sigma: jax.Array | float,
) -> jax.Array:
    """
    Probability integral transform values for the PRO GP mixture predictive.

    Evaluates the exact pointwise mixture CDF
    ``(1/J) sum_j Phi((y_i - f_ij) / sigma_eff_i)`` where
    ``f_ij`` is the projected value of retained particle ``j`` at test point
    ``i``. For calibrated scalar predictive distributions, these values should
    be approximately uniform on ``[0, 1]``.

    Parameters
    ----------
    y_test
        Observed test targets, shape ``(N_test,)`` or ``(N_test, 1)``.
    test_basis
        Test projection matrix from :func:`prediction_basis`,
        shape ``(N_test, M)`` (inducing) or ``(N_test, N)`` (full GP).
    test_covariance
        Prior kernel matrix at test points ``K(x_*, x_*)``,
        shape ``(N_test, N_test)``, also from :func:`prediction_basis`.
    particles
        Retained latent particles, shape ``(M, J)`` or
        ``(num_steps, M, J)``.
    sigma
        Observation noise standard deviation (plain array or
        ``paramax``-wrapped; unwrapped automatically).

    Returns
    -------
    PIT values, shape ``(N_test,)``.
    """
    sigma_val = paramax.unwrap(sigma)
    particle_matrix = _as_particle_matrix(particles)
    projected = test_basis @ particle_matrix

    q_diag = jnp.sum(test_basis**2, axis=1)
    k_diag = jnp.diag(test_covariance)
    test_residual_std = jnp.sqrt(jnp.maximum(k_diag - q_diag, 0.0))
    sigma_eff = jnp.sqrt(sigma_val**2 + test_residual_std**2).reshape(-1, 1)

    y = y_test.squeeze().reshape(-1, 1)
    return jnp.mean(jax.scipy.stats.norm.cdf((y - projected) / sigma_eff), axis=1)


def nlpd_gp(
    y_test: jax.Array,
    predictive_mean: jax.Array,
    predictive_std: jax.Array,
    *,
    return_per_point: bool = False,
) -> jax.Array:
    """
    Mean negative log predictive density for a Gaussian GP predictive.

    Parameters
    ----------
    y_test
        Observed test targets, shape ``(N_test,)`` or ``(N_test, 1)``.
    predictive_mean
        GP predictive mean, same shape as ``y_test``.
    predictive_std
        GP predictive standard deviation (including observation noise),
        same shape as ``y_test``.

    Returns
    -------
    Scalar NLPD averaged over test points.
    """
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
    """
    Mean negative log predictive density for the PRO GP mixture predictive.

    The predictive is a mixture of Gaussians, one per retained particle:

    .. math::

        p(y^* | \\mathcal{D}) \\approx \\frac{1}{J} \\sum_j
            \\mathcal{N}(y^* \\mid B^* z_j,\\; \\sigma_{\\text{eff}}^2)

    where :math:`\\sigma_{\\text{eff},i}^2 = \\sigma^2 + s_i^2` and
    :math:`s_i = \\sqrt{\\max(k(x^*_i, x^*_i) - \\|B^*_{i,:}\\|^2, 0)}` is the
    sparse-approximation residual at each test point (zero for the full GP).

    Parameters
    ----------
    y_test
        Observed test targets, shape ``(N_test, 1)``.
    test_basis
        Test projection matrix from :func:`prediction_basis`,
        shape ``(N_test, M)`` (inducing) or ``(N_test, N)`` (full GP).
    test_covariance
        Prior kernel matrix at test points ``K(x_*, x_*)``,
        shape ``(N_test, N_test)``, also from :func:`prediction_basis`.
    particles
        Retained latent particles, shape ``(M, J)`` or
        ``(num_steps, M, J)`` (scanned output from
        :func:`blackjax.util.run_inference_algorithm`).
    sigma
        Observation noise standard deviation (plain array or
        ``paramax``-wrapped; unwrapped automatically).

    Returns
    -------
    Scalar NLPD averaged over test points.
    """
    sigma_val = paramax.unwrap(parameters.sigma)
    particle_matrix = _as_particle_matrix(particles)       # (M, J_total)
    projected = test_basis @ particle_matrix               # (N_test, J_total)
    j_total = projected.shape[1]

    # Sparse-approximation residual at test points; 0 for full GP.
    q_diag = jnp.sum(test_basis**2, axis=1)               # (N_test,)
    k_diag = jnp.diag(test_covariance)                    # (N_test,)
    test_residual_std = jnp.sqrt(jnp.maximum(k_diag - q_diag, 0.0))
    sigma_eff = jnp.sqrt(sigma_val**2 + test_residual_std**2).reshape(-1, 1)

    y = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test   # (N_test, 1)
    log_normals = (
        -0.5 * jnp.log(2 * jnp.pi)
        - jnp.log(sigma_eff)
        - 0.5 * ((y - projected) / sigma_eff) ** 2
    )                                                      # (N_test, J_total)
    log_p = jax.nn.logsumexp(log_normals, axis=1) - jnp.log(j_total)

    if return_per_point:
        return -log_p

    return -jnp.mean(log_p)


def posterior_function_draws(  # noqa: PLR0913
    rng_key: PRNGKey,
    test_basis: jax.Array,
    test_covariance: jax.Array,
    particles: jax.Array,
    *,
    num_draws: int = 120,
    jitter: float = 1e-6,
) -> jax.Array:
    """
    Draw smooth posterior function trajectories from retained particles.

    ``test_basis`` should be the same projection matrix used for
    ``predictive_moments``. In the full-GP case this is ``V.T``, where
    ``V = solve(train_cholesky, K_test_train.T)``.

    For a genuine Nystrom/inducing-point basis, ``test_covariance -
    test_basis @ test_basis.T`` is guaranteed PSD (it's a Schur complement).
    That's not true for approximations like Random Fourier Features, where
    ``test_basis @ test_basis.T`` is an independent Monte-Carlo estimate of
    the kernel rather than a projection onto a subspace of it — the
    difference can have negative eigenvalues no jitter reasonably fixes. To
    stay correct for any basis, negative eigenvalues are clipped to zero
    (the nearest PSD matrix in this eigenbasis) before the Cholesky.
    """
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

    # Burn-in: same progress-bar-aware scan blackjax uses internally (gen_scan_fn),
    # but the per-step output is None, so nothing is stacked/stored for these steps.
    state = inference_algorithm.init(initial_position)
    burn_keys = jr.split(burn_key, num_burn_steps)
    burn_scan_fn = gen_scan_fn(num_burn_steps, progress_bar=progress_bar)

    def burn_step(state, xs):
        _, key = xs
        state, _info = inference_algorithm.step(key, state)
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