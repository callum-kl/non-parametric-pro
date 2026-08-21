from typing import NamedTuple

import equinox as eqx
import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import optax
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import PRNGKey
from gpjax.dataset import Dataset
from gpjax.objectives import _val
from jax import vmap
from jax.scipy.linalg import solve_triangular


def predictive_log_likelihood(
    variational_family,
    data: Dataset,
    *,
    beta: float = 1.0,
) -> jnp.ndarray:
    r"""
    Predictive log likelihood objective for sparse variational GPs.

    Proposed in Jankowiak et al. (2020) "Parametric Gaussian Process Regressors"
    (https://arxiv.org/abs/1910.07123) as an alternative to the ELBO that
    produces better-calibrated predictive variances.

    The objective is

    .. math::

        \mathcal{L}_\text{PLL} =
            \sum_{i=1}^N \log \mathcal{N}(y_i;\, \mu_q(x_i),\, \sigma^2 + v_q(x_i))
            - \beta\, \text{KL}[q(u) \| p(u)]

    where :math:`\mu_q` and :math:`v_q` are the mean and marginal variance of
    the variational predictive :math:`q(f)`, and :math:`\sigma^2` is the
    observation noise.  Unlike the ELBO, the :math:`\log` sits *outside* the
    expectation over :math:`q(u)`, so Jensen's inequality makes this an upper
    bound on the ELBO (PLL :math:`\geq` ELBO).

    Parameters
    ----------
    variational_family
        A :class:`gpjax.variational_families.VariationalGaussian` instance.
    data
        Training dataset (may be a mini-batch; scaled automatically).
    beta
        Weight on the KL term.  ``beta=1`` (default) recovers the objective
        from the paper.  Values < 1 reduce regularisation similarly to
        :math:`\beta`-VAEs.

    Returns
    -------
    jnp.ndarray
        Scalar PLL value (higher is better; negate for minimisation).

    """
    x, y, n = data.X, data.y, data.n

    # ---- unpack variational family -----------------------------------------
    variational_mean = _val(variational_family.variational_mean)
    variational_sqrt = _val(variational_family.variational_root_covariance)
    inducing_inputs = _val(variational_family.inducing_inputs)

    mean_fn = variational_family.posterior.prior.mean_function
    kernel = variational_family.posterior.prior.kernel
    noise = _val(variational_family.posterior.likelihood.obs_stddev) ** 2
    jitter = variational_family.jitter

    # ---- kernel matrices ----------------------------------------------------
    Kzz = kernel.gram(inducing_inputs).as_matrix()
    Kzz = Kzz + jitter * jnp.eye(Kzz.shape[0])
    Lz = jnp.linalg.cholesky(Kzz)

    Kzx = kernel.cross_covariance(inducing_inputs, x)
    Kxx_diag = vmap(kernel, in_axes=(0, 0))(x, x)

    muz = mean_fn(inducing_inputs)
    mux = mean_fn(x)

    # ---- predictive mean: μ(x) = μx + Kxz Kzz⁻¹ (mz − μz) ----------------
    Lz_inv_Kzx = solve_triangular(Lz, Kzx, lower=True)
    Kzz_inv_Kzx = solve_triangular(Lz.T, Lz_inv_Kzx, lower=False)

    pred_mean = (mux + Kzz_inv_Kzx.T @ (variational_mean - muz)).squeeze()

    # ---- predictive variance (diagonal only) --------------------------------

    A = variational_sqrt.T @ Kzz_inv_Kzx
    pred_var = Kxx_diag - jnp.sum(Lz_inv_Kzx**2, axis=0) + jnp.sum(A**2, axis=0)

    # ---- per-point log N(y_i; μ_i, σ² + v_i) -------------------------------
    total_var = noise + pred_var
    log_lik = (
        -0.5 * jnp.log(2.0 * jnp.pi * total_var)
        - 0.5 * (y.squeeze() - pred_mean) ** 2 / total_var
    )

    # ---- KL(q(u) || p(u)) -----
    S = variational_sqrt @ variational_sqrt.T
    Ls = jnp.linalg.cholesky(S + jitter * jnp.eye(S.shape[0]))

    log_det_Kzz = 2.0 * jnp.sum(jnp.log(jnp.diag(Lz)))
    log_det_S = 2.0 * jnp.sum(jnp.log(jnp.diag(Ls)))

    Lz_inv_Ls = solve_triangular(Lz, Ls, lower=True)
    trace_term = jnp.sum(Lz_inv_Ls**2)

    diff = variational_mean - muz
    Lz_inv_diff = solve_triangular(Lz, diff, lower=True)
    mahalanobis = jnp.sum(Lz_inv_diff**2)

    m = inducing_inputs.shape[0]
    kl = 0.5 * (log_det_Kzz - log_det_S - m + trace_term + mahalanobis)

    N_total = variational_family.posterior.likelihood.num_datapoints

    return jnp.sum(log_lik) * N_total / n - beta * kl


class NaturalGradientState(NamedTuple):
    """Carry for :func:`natural_gradient_svgp_fit`'s optimisation loop."""

    posterior: gpx.gps.AbstractPosterior
    inducing_inputs: jnp.ndarray
    natural_vector: jnp.ndarray  # (M, 1) -- theta_1 = S^{-1} mu
    natural_matrix: jnp.ndarray  # (M, M) -- theta_2 = -S^{-1} / 2
    hyper_opt_state: optax.OptState


def _natural_to_moments(
    natural_vector: jnp.ndarray,
    natural_matrix: jnp.ndarray,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Recover ``(mu, S)`` from natural parameters ``(theta_1, theta_2)``.

    ``S^{-1} = -2 theta_2``, ``mu = S theta_1``. Uses the same reversed-Cholesky
    trick as ``gpjax.variational_families.NaturalVariationalGaussian`` itself
    (``prior_kl``/``predict``), rather than a direct ``jnp.linalg.inv``, to stay
    numerically consistent with how gpjax computes the same quantity internally.
    """
    m = natural_matrix.shape[-1]
    s_inv = -2 * natural_matrix + jitter * jnp.eye(m)
    sqrt_inv = jnp.swapaxes(
        jnp.linalg.cholesky(s_inv[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
    )
    sqrt = jsp.linalg.solve_triangular(sqrt_inv, jnp.eye(m), lower=True)
    s = sqrt @ sqrt.T
    mu = s @ natural_vector
    return mu, s


def _eta_loss(
    eta: tuple[jnp.ndarray, jnp.ndarray],
    data: Dataset,
    posterior: gpx.gps.AbstractPosterior,
    inducing_inputs: jnp.ndarray,
    jitter: float,
) -> jnp.ndarray:
    """ELBO as a function of q(u)'s expectation parameters only (ascent target)."""
    eta1, eta2 = eta
    exp_vf = gpx.variational_families.ExpectationVariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        expectation_vector=eta1,
        expectation_matrix=eta2,
        jitter=jitter,
    )
    return gpx.objectives.elbo(exp_vf, data)


def _hyper_loss(
    hyper: tuple[gpx.gps.AbstractPosterior, jnp.ndarray],
    data: Dataset,
    eta1: jnp.ndarray,
    eta2: jnp.ndarray,
    jitter: float,
) -> jnp.ndarray:
    """Negative ELBO as a function of (posterior, inducing_inputs) only."""
    posterior, inducing_inputs = hyper
    exp_vf = gpx.variational_families.ExpectationVariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        expectation_vector=eta1,
        expectation_matrix=eta2,
        jitter=jitter,
    )
    return -gpx.objectives.elbo(exp_vf, data)


def natural_gradient_svgp_fit(
    posterior: gpx.gps.AbstractPosterior,
    inducing_inputs: jnp.ndarray,
    train_data: Dataset,
    *,
    natural_lr: float = 0.1,
    hyper_optimizer: optax.GradientTransformation | None = None,
    num_iters: int = 2000,
    batch_size: int = -1,
    key: PRNGKey | None = None,
    jitter: float = 1e-6,
    progress_bar: bool = False,
) -> tuple[gpx.variational_families.NaturalVariationalGaussian, jnp.ndarray]:
    r"""
    Fit an uncollapsed SVGP with natural gradients on ``q(u)``, Adam on the rest.

    Alternates, every step, between:

    - A **natural-gradient ascent** step on ``q(u)``'s natural parameters
      ``(theta_1, theta_2) = (S^{-1} mu, -S^{-1}/2)``. The natural gradient of the
      ELBO w.r.t. ``theta`` equals the ordinary (Euclidean) gradient of the ELBO
      w.r.t. the corresponding *expectation* parameters ``eta = (mu, S + mu mu^T)``
      (Hensman et al., 2012; Salimbeni et al., 2018) -- so this evaluates the ELBO
      via :class:`gpjax.variational_families.ExpectationVariationalGaussian`,
      differentiates w.r.t. its ``expectation_vector``/``expectation_matrix``, and
      adds that gradient directly to ``theta`` (plain ascent, no Adam -- an
      adaptive per-parameter step would break the natural-gradient identity).
    - An ordinary **Adam** step on the kernel/likelihood hyperparameters and
      ``inducing_inputs``, holding ``q(u)`` fixed at its current value.

    This is the standard fix for how slowly plain Euclidean Adam optimises the
    uncollapsed ELBO's ``q(u)`` covariance in practice; gpjax ships the
    ``NaturalVariationalGaussian``/``ExpectationVariationalGaussian``
    parameterisations this relies on, but no ready-made optimiser gluing them
    together, hence this loop.

    Parameters
    ----------
    posterior
        Prior * likelihood, as passed to any gpjax variational family.
    inducing_inputs
        Initial inducing locations ``(M, D)`` (e.g. from
        :func:`non_parametric_pro.inducing.kmeans_inducing_points`).
    train_data
        Full training :class:`gpjax.Dataset`. ``posterior.likelihood.num_datapoints``
        must equal ``train_data.n`` (the *full* size, not any minibatch) --
        ``gpx.objectives.elbo`` rescales minibatches using that ratio.
    natural_lr
        Step size for the natural-gradient ascent on ``q(u)``. Unlike Adam's
        learning rate, this isn't preconditioned per-parameter, so it usually
        wants to be an order of magnitude or so, not the small values (``1e-2``
        - ``1e-3``) typical of Adam.
    hyper_optimizer
        Optax optimiser for ``(posterior, inducing_inputs)``. Defaults to
        ``optax.adam(1e-2)``.
    num_iters
        Number of alternating steps.
    batch_size
        If ``> 0`` and less than ``train_data.n``, resample a fresh random
        minibatch of this size every step instead of using the full dataset.
        ``-1`` (default) disables minibatching.
    key
        Required if ``batch_size`` enables minibatching; unused otherwise.
    jitter
        Diagonal jitter for the ``S^{-1}`` Cholesky and for gpjax's own ``K_zz``
        factorisations inside the variational family.
    progress_bar
        Show a progress bar during the scan (same style used elsewhere in this
        package, e.g. :func:`non_parametric_pro.adaptation.parameter_adaptation`).

    Returns
    -------
    A ``(NaturalVariationalGaussian, elbo_history)`` pair: the fitted variational
    family (natural parameterisation) and the per-step ELBO trace, shape
    ``(num_iters,)``.

    """
    if hyper_optimizer is None:
        hyper_optimizer = optax.adam(1e-2)

    minibatch = 0 < batch_size < train_data.n
    if minibatch and key is None:
        msg = "`key` is required when `batch_size` enables minibatching."
        raise ValueError(msg)

    num_inducing = inducing_inputs.shape[0]
    init_state = NaturalGradientState(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        natural_vector=jnp.zeros((num_inducing, 1)),
        natural_matrix=-0.5 * jnp.eye(num_inducing),
        hyper_opt_state=hyper_optimizer.init(
            eqx.filter((posterior, inducing_inputs), eqx.is_array)
        ),
    )

    def step(
        state: NaturalGradientState, xs: tuple[jnp.ndarray, PRNGKey]
    ) -> tuple[NaturalGradientState, jnp.ndarray]:
        _, step_key = xs
        data = train_data
        if minibatch:
            batch_idx = jr.choice(step_key, train_data.n, (batch_size,), replace=False)
            data = Dataset(X=train_data.X[batch_idx], y=train_data.y[batch_idx])

        eta1, eta2 = _natural_to_moments(
            state.natural_vector, state.natural_matrix, jitter
        )
        eta2_full = eta2 + eta1 @ eta1.T

        elbo_val, (grad_eta1, grad_eta2) = eqx.filter_value_and_grad(_eta_loss)(
            (eta1, eta2_full), data, state.posterior, state.inducing_inputs, jitter
        )
        new_natural_vector = state.natural_vector + natural_lr * grad_eta1
        new_natural_matrix = state.natural_matrix + natural_lr * grad_eta2

        hyper = (state.posterior, state.inducing_inputs)
        grad_hyper = eqx.filter_grad(_hyper_loss)(hyper, data, eta1, eta2_full, jitter)
        updates, new_hyper_opt_state = hyper_optimizer.update(
            grad_hyper, state.hyper_opt_state, eqx.filter(hyper, eqx.is_array)
        )
        new_posterior, new_inducing_inputs = eqx.apply_updates(hyper, updates)

        new_state = NaturalGradientState(
            posterior=new_posterior,
            inducing_inputs=new_inducing_inputs,
            natural_vector=new_natural_vector,
            natural_matrix=new_natural_matrix,
            hyper_opt_state=new_hyper_opt_state,
        )
        return new_state, elbo_val

    keys = (
        jr.split(key, num_iters)
        if minibatch
        else jnp.zeros((num_iters, 2), dtype=jnp.uint32)
    )
    scan_fn = gen_scan_fn(num_iters, progress_bar=progress_bar)
    final_state, elbo_history = scan_fn(step, init_state, (jnp.arange(num_iters), keys))

    final_vf = gpx.variational_families.NaturalVariationalGaussian(
        posterior=final_state.posterior,
        inducing_inputs=final_state.inducing_inputs,
        natural_vector=final_state.natural_vector,
        natural_matrix=final_state.natural_matrix,
        jitter=jitter,
    )
    return final_vf, elbo_history
