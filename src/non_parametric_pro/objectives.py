# """Additional GPJax-compatible training objectives."""

# import jax.numpy as jnp
# from gpjax.dataset import Dataset
# from gpjax.objectives import _val
# from jax import vmap
# from jax.scipy.linalg import solve_triangular


# def predictive_log_likelihood(
#     variational_family,
#     data: Dataset,
#     *,
#     beta: float = 1.0,
# ) -> jnp.ndarray:
#     r"""Predictive log likelihood objective for sparse variational GPs.

#     Proposed in Jankowiak et al. (2020) "Parametric Gaussian Process Regressors"
#     (https://arxiv.org/abs/1910.07123) as an alternative to the ELBO that
#     produces better-calibrated predictive variances.

#     The objective is

#     .. math::

#         \mathcal{L}_\text{PLL} =
#             \sum_{i=1}^N \log \mathcal{N}(y_i;\, \mu_q(x_i),\, \sigma^2 + v_q(x_i))
#             - \beta\, \text{KL}[q(u) \| p(u)]

#     where :math:`\mu_q` and :math:`v_q` are the mean and marginal variance of
#     the variational predictive :math:`q(f)`, and :math:`\sigma^2` is the
#     observation noise.  Unlike the ELBO, the :math:`\log` sits *outside* the
#     expectation over :math:`q(u)`, so Jensen's inequality makes this an upper
#     bound on the ELBO (PLL :math:`\geq` ELBO).

#     This function is designed for :class:`gpjax.variational_families.VariationalGaussian`
#     (non-collapsed).  For the collapsed family the inducing distribution is
#     integrated out analytically, making the per-point marginal variances of the
#     collapsed posterior the natural plug-in.

#     Parameters
#     ----------
#     variational_family
#         A :class:`gpjax.variational_families.VariationalGaussian` instance.
#     data
#         Training dataset (may be a mini-batch; scaled automatically).
#     beta
#         Weight on the KL term.  ``beta=1`` (default) recovers the objective
#         from the paper.  Values < 1 reduce regularisation similarly to
#         :math:`\beta`-VAEs.

#     Returns
#     -------
#     jnp.ndarray
#         Scalar PLL value (higher is better; negate for minimisation).
#     """
#     x, y, n = data.X, data.y, data.n

#     # ---- unpack variational family -----------------------------------------
#     variational_mean = _val(variational_family.variational_mean)        # (M, 1)
#     variational_sqrt = _val(variational_family.variational_root_covariance)  # (M, M)
#     inducing_inputs  = _val(variational_family.inducing_inputs)         # (M, D)

#     mean_fn = variational_family.posterior.prior.mean_function
#     kernel  = variational_family.posterior.prior.kernel
#     noise   = _val(variational_family.posterior.likelihood.obs_stddev) ** 2
#     jitter  = variational_family.jitter

#     # ---- kernel matrices ----------------------------------------------------
#     Kzz = kernel.gram(inducing_inputs).as_matrix()
#     Kzz = Kzz + jitter * jnp.eye(Kzz.shape[0])
#     Lz  = jnp.linalg.cholesky(Kzz)                   # (M, M)

#     Kzx      = kernel.cross_covariance(inducing_inputs, x)  # (M, N)
#     Kxx_diag = vmap(kernel, in_axes=(0, 0))(x, x)          # (N,)

#     muz  = mean_fn(inducing_inputs)   # (M, 1)
#     mux  = mean_fn(x)                 # (N, 1)

#     # ---- predictive mean: μ(x) = μx + Kxz Kzz⁻¹ (mz − μz) ----------------
#     Lz_inv_Kzx  = solve_triangular(Lz, Kzx, lower=True)               # (M, N)
#     Kzz_inv_Kzx = solve_triangular(Lz.T, Lz_inv_Kzx, lower=False)     # (M, N)

#     pred_mean = (mux + Kzz_inv_Kzx.T @ (variational_mean - muz)).squeeze()  # (N,)

#     # ---- predictive variance (diagonal only) --------------------------------
#     # var(f_i) = K_ii
#     #          − ||Lz⁻¹ Kz_i||²          (prior uncertainty removed by inducing)
#     #          + ||sqrt^T Kzz⁻¹ Kz_i||²  (variance added back from q(u))
#     A        = variational_sqrt.T @ Kzz_inv_Kzx              # (M, N)
#     pred_var = (
#         Kxx_diag
#         - jnp.sum(Lz_inv_Kzx ** 2, axis=0)
#         + jnp.sum(A ** 2, axis=0)
#     )                                                         # (N,)

#     # ---- per-point log N(y_i; μ_i, σ² + v_i) -------------------------------
#     total_var = noise + pred_var                              # (N,)
#     log_lik = (
#         -0.5 * jnp.log(2.0 * jnp.pi * total_var)
#         - 0.5 * (y.squeeze() - pred_mean) ** 2 / total_var
#     )                                                         # (N,)

#     # Scale for mini-batching: multiply by N / batch_size
#     N_total = variational_family.posterior.likelihood.num_datapoints
#     kl = variational_family.prior_kl()

#     return jnp.sum(log_lik) * N_total / n - beta * kl
