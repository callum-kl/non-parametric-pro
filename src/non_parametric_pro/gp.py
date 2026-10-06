import jax.numpy as jnp
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
    """
    Predictive log likelihood objective for sparse variational GPs.
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
