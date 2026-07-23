"""Density and score functions for predictively oriented ULA."""

from numbers import Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
import paramax
from jax.scipy.special import logsumexp


class ProParameters(NamedTuple):
    """
    Parameters controlling one ULA iteration.
    """

    y: jax.Array
    step_size: float
    sigma: jax.Array | float | paramax.AbstractUnwrappable
    alpha: float
    jitter: float = 1e-6
    tolerance: float = 1e-300
    basis: jax.Array | None = None
    residual_std: jax.Array | None = None  # (N,1) inducing residual; None for full GP
    sigma_prior_value: jax.Array | float | None = None
    sigma_prior_scale: jax.Array | float | None = None
    batch_idx: jax.Array | None = None  # (b,) minibatch row indices; None = full batch


def _effective_sigma(parameters: "ProParameters") -> jax.Array:
    """sigma for full GP, sqrt(sigma² + residual_std²) for inducing."""
    sigma = paramax.unwrap(parameters.sigma)
    if parameters.residual_std is None:
        return sigma
    return jnp.sqrt(sigma**2 + parameters.residual_std**2)


def _reflected_sigma_halfcauchy_logpdf(parameters: "ProParameters") -> jax.Array:
    """Evaluate the optional reflected half-Cauchy prior on sigma."""
    value = parameters.sigma_prior_value
    scale = parameters.sigma_prior_scale
    if value is None and scale is None:
        return jnp.asarray(0.0)
    if value is None or scale is None:
        raise ValueError(
            "sigma_prior_value and sigma_prior_scale must both be set"
        )
    if isinstance(scale, Real) and scale <= 0.0:
        raise ValueError("sigma_prior_scale must be positive")

    sigma = paramax.unwrap(parameters.sigma)
    distance = value - sigma
    safe_scale = jnp.where(jnp.asarray(scale) > 0.0, scale, 1.0)
    logpdf = (
        jnp.log(2.0 / jnp.pi)
        - jnp.log(safe_scale)
        - jnp.log1p((distance / safe_scale) ** 2)
    )
    return jnp.where((safe_scale > 0.0) & (distance >= 0.0), logpdf, -jnp.inf)


def normal_logpdf(y, mean, sigma):
    """Evaluate the Gaussian observation density for each particle."""
    return (
        -0.5 * ((y - mean) / sigma) ** 2
        - jnp.log(sigma)
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )


def student_t_logpdf(y, mean, sigma, df=4.0):
    """Evaluate the Student-t observation log-density for each particle."""
    z = (y - mean) / sigma
    nu = df
    return (
        jax.scipy.special.gammaln((nu + 1.0) / 2.0)
        - jax.scipy.special.gammaln(nu / 2.0)
        - 0.5 * jnp.log(nu * jnp.pi)
        - jnp.log(sigma)
        - 0.5 * (nu + 1.0) * jnp.log1p((z**2) / nu)
    )


def pro_logdensity_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Compute the log density of the posterior over latent particles."""
    return pro_score_fn(z, parameters) - 0.5 * jnp.sum(z**2)


def pro_crps_logdensity_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Compute the CRPS-targeted log density over latent particles."""
    return pro_crps_score_fn(z, parameters) - 0.5 * jnp.sum(z**2)


def _batch_parameters(parameters: ProParameters) -> tuple[ProParameters, jax.Array]:
    """
    Gather `basis`/`y`/`residual_std` down to `parameters.batch_idx`'s rows.

    Returns the gathered parameters and the rescale factor `n / b` that makes
    a sum over the batch an unbiased estimate of the sum over all `n` rows
    (standard minibatch/SGLD estimator; the PrO objective's score term is
    already an explicit average/sum over data points, so this is a direct
    Monte Carlo subsample of it). A no-op (scale 1.0) if `batch_idx` is None.
    """
    if parameters.batch_idx is None:
        return parameters, jnp.asarray(1.0)
    n = parameters.basis.shape[0]
    b = parameters.batch_idx.shape[0]
    # `basis`/`y`/`residual_std` may still be plain numpy arrays (e.g. straight out of
    # sklearn's StandardScaler) -- ordinary arithmetic elsewhere auto-promotes those to
    # jax arrays transparently, but fancy-indexing with a *traced* `batch_idx` does not,
    # so this must coerce explicitly before indexing.
    basis = jnp.asarray(parameters.basis)
    y = jnp.asarray(parameters.y)
    residual_std = (
        jnp.asarray(parameters.residual_std)[parameters.batch_idx]
        if parameters.residual_std is not None
        else None
    )
    batched = parameters._replace(
        basis=basis[parameters.batch_idx],
        y=y[parameters.batch_idx],
        residual_std=residual_std,
    )
    return batched, n / b


def pro_score_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """
    Compute the score function.

    If `parameters.batch_idx` is set, this is a minibatch estimate of the
    full-data score -- rescaled by `n / b` so it stays unbiased -- rather than
    the exact value; see `_batch_parameters`.
    """
    parameters, scale = _batch_parameters(parameters)
    sigma = _effective_sigma(parameters)
    a = parameters.basis @ z
    log_density = normal_logpdf(parameters.y, a, sigma)
    num_particles = log_density.shape[1]
    log_marginal = logsumexp(log_density, axis=1) - jnp.log(num_particles)
    log_marginal = jnp.maximum(log_marginal, jnp.log(parameters.tolerance))
    score = num_particles * parameters.alpha * scale * jnp.sum(log_marginal)
    return score


def _crps_g(x: jax.Array) -> jax.Array:
    """Helper for the closed-form CRPS of a Gaussian."""
    phi = jnp.exp(-0.5 * x**2) / jnp.sqrt(2.0 * jnp.pi)
    Phi = 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))
    return 2.0 * phi + x * (2.0 * Phi - 1.0)


def pro_crps_score_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """
    Compute the negative CRPS objective for scalar PRO predictions.

    The return value follows :func:`pro_score_fn`: larger is better and the
    score is scaled by ``num_particles * alpha``. For scalar regression, CRPS
    is the one-dimensional energy score.
    """
    sigma = _effective_sigma(parameters)
    means = parameters.basis @ z
    y = parameters.y.reshape(-1, 1)
    num_particles = means.shape[1]

    sigma = jnp.asarray(sigma)
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1, 1)
    sigma = jnp.broadcast_to(sigma, y.shape)

    first = sigma[:, 0] * jnp.mean(_crps_g((y - means) / sigma), axis=1)

    sqrt_two = jnp.sqrt(jnp.asarray(2.0, dtype=means.dtype))

    def cross_one(args: tuple[jax.Array, jax.Array]) -> jax.Array:
        mean_i, sigma_i = args
        d_norm = (mean_i[:, None] - mean_i[None, :]) / (sigma_i * sqrt_two)
        return 0.5 * sigma_i * sqrt_two * jnp.mean(_crps_g(d_norm))

    cross = jax.lax.map(cross_one, (means, sigma[:, 0]))
    crps = first - cross
    return -num_particles * parameters.alpha * jnp.sum(crps)


def pro_energy_score_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Alias for :func:`pro_crps_score_fn` in scalar-output regression."""
    return pro_crps_score_fn(z, parameters)


def regularised_score(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Compute the sigma-adaptation objective with an optional sigma prior."""
    num_data = parameters.y.shape[0]
    num_particles = z.shape[1]
    average_log_score = pro_score_fn(z, parameters) / (num_data * num_particles)
    return average_log_score + _reflected_sigma_halfcauchy_logpdf(parameters)


def regularised_crps_score(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Compute the average negative CRPS with an optional sigma prior."""
    num_data = parameters.y.shape[0]
    num_particles = z.shape[1]
    average_crps_score = pro_crps_score_fn(z, parameters) / (
        num_data * num_particles
    )
    return average_crps_score + _reflected_sigma_halfcauchy_logpdf(parameters)


def regularised_energy_score(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Alias for :func:`regularised_crps_score` in scalar-output regression."""
    return regularised_crps_score(z, parameters)


def predictive_score(
    z: jax.Array,
    parameters: ProParameters,
) -> jax.Array:
    """Compute the predictive score reported by ULA transition info."""
    return pro_score_fn(z, parameters)


def pro_logdensity_and_grad_fn(
    z: jax.Array,
    parameters: ProParameters,
) -> tuple[jax.Array, jax.Array]:
    """Compute the scalar PRO log density and its hand-derived gradient."""
    sigma = _effective_sigma(parameters)
    a = parameters.basis @ z

    log_density = normal_logpdf(parameters.y, a, sigma)
    num_particles = log_density.shape[1]

    log_marginal = logsumexp(log_density, axis=1, keepdims=True) - jnp.log(
        num_particles
    )

    weights = jnp.exp(log_density - log_marginal) * (parameters.y - a) / sigma**2

    grad = parameters.alpha * (parameters.basis.T @ weights) - z

    likelihood = num_particles * parameters.alpha * jnp.sum(log_marginal[:, 0])
    logdensity = likelihood - 0.5 * jnp.sum(z**2)

    return logdensity, grad
