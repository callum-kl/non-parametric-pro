"""
Corrupted Hartmann6 / Friedman1 regression benchmark.

Replicates a scoped version of Ament et al. 2024, "Robust Gaussian Processes via Relevance
Pursuit" (NeurIPS 2024), Section 6.1 / Fig. 4: compares a standard (non-robust) exact GP
against this repo's PRO-GP sampler on data corrupted per the paper's sparse-corruption model
(Section 2.2), reporting predictive log-likelihood (= -NLPD) on a clean held-out test set.

This does NOT reproduce the paper's other baselines (RRP itself, Student-t likelihood,
Trimmed MLL, Adaptive Winsorization, Power Transform) or their exact experimental constants
(corruption magnitude, N, noise level) -- see
non_parametric_pro.data.corruption_benchmarks for what is and isn't pinned down by the paper.

Example:
    python experiments/corruption_benchmark.py --function hartmann6 \
        --corruption constant --outlier-fraction 0.05 --num-reps 32
"""

import argparse
import json
import os
from dataclasses import dataclass

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
from blackjax.util import run_inference_algorithm

from non_parametric_pro import ula
from non_parametric_pro.adaptation.parameter_adaptation import parameter_adaptation
from non_parametric_pro.data.corruption_benchmarks import (
    CorruptionType,
    friedman1,
    hartmann6,
    make_corrupted_regression_case,
)
from non_parametric_pro.density import ProParameters, pro_logdensity_fn, pro_score_fn
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import (
    crps_gp,
    crps_pro,
    nlpd_gp,
    nlpd_pro,
    prediction_basis,
    train_val_split,
)

FUNCTIONS = {
    "hartmann6": (hartmann6, 6),
    "friedman5": (friedman1, 5),
    "friedman10": (friedman1, 10),
}


@dataclass
class Metrics:
    gp_nlpd: float
    gp_crps: float
    pro_nlpd: float
    pro_crps: float


def _cholesky_basis(kernel: gpx.kernels.AbstractKernel, x: jax.Array, jitter: float = 1e-6) -> jax.Array:
    k = kernel.gram(x).as_matrix()
    return jnp.linalg.cholesky(k + jitter * jnp.eye(k.shape[0]))


def _fit_exact_gp(
    x_train: jax.Array, y_train: jax.Array, dim: int, *, num_iters: int, kernel_lr: float
) -> tuple[gpx.kernels.AbstractKernel, jax.Array, gpx.Dataset]:
    data = gpx.Dataset(X=x_train, y=y_train)
    kernel = gpx.kernels.Matern32(lengthscale=jnp.ones((dim,)))
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n)
    posterior = prior * likelihood

    opt_posterior, _ = gpx.fit(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=data,
        optim=ox.adam(kernel_lr),
        num_iters=num_iters,
        verbose=False,
    )
    return opt_posterior.prior.kernel, opt_posterior.likelihood.obs_stddev, data


def _run_one(  # noqa: PLR0913
    key: jax.Array,
    function_name: str,
    corruption_type: CorruptionType,
    outlier_fraction: float,
    *,
    n_train: int,
    n_test: int,
    sigma: float,
    gp_num_iters: int,
    kernel_lr: float,
    num_adapt_steps: int,
    num_sample_steps: int,
    num_particles: int,
) -> Metrics:
    fn, dim = FUNCTIONS[function_name]
    data_key, gp_key, pro_key = jr.split(key, 3)

    case = make_corrupted_regression_case(
        data_key,
        fn,
        dim=dim,
        n_train=n_train,
        n_test=n_test,
        sigma=sigma,
        outlier_fraction=outlier_fraction,
        corruption_type=corruption_type,
    )

    # --- Standard (non-robust) exact GP -------------------------------------
    gp_kernel, gp_sigma, data = _fit_exact_gp(
        case.x_train, case.y_train, dim, num_iters=gp_num_iters, kernel_lr=kernel_lr
    )
    posterior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=gp_kernel) * (
        gpx.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=gp_sigma)
    )
    latent = posterior.predict(case.x_test, train_data=data)
    predictive = posterior.likelihood(latent)
    gp_mean, gp_std = predictive.mean, jnp.sqrt(predictive.variance)

    gp_nlpd = float(nlpd_gp(case.y_test, gp_mean, gp_std))
    gp_crps = float(crps_gp(case.y_test, gp_mean, gp_std))

    # --- PRO GP --------------------------------------------------------------
    tv = train_val_split(pro_key, case.x_train, case.y_train, val_fraction=0.2)
    kernel0 = gpx.kernels.Matern32(lengthscale=jnp.ones((dim,)))
    basis0 = _cholesky_basis(kernel0, tv.x_train)

    sigma0 = gpx.parameters.SigmoidBounded(sigma, low=1e-3, high=100.0)
    pro_params = ProParameters(
        y=tv.y_train, basis=basis0, step_size=2e-4, sigma=sigma0, alpha=1.0, residual_std=None
    )

    key, pos_key = jr.split(pro_key)
    pro_position = jr.normal(pos_key, (tv.x_train.shape[0], num_particles))

    adaptation = parameter_adaptation(
        ula,
        pro_logdensity_fn,
        pro_params,
        x_train=tv.x_train,
        initial_kernel=kernel0,
        warmup_steps=num_adapt_steps,
        sigma_adapt_every=1,
        kernel_adapt_every=10,
        objective_fn=pro_score_fn,
        x_val=tv.x_val,
        y_val=tv.y_val,
        sigma_optimizer=ox.adam(1e-2),
        kernel_optimizer=ox.adam(kernel_lr),
        progress_bar=False,
    )
    key, adapt_key = jr.split(key)
    adaptation_results, adaptation_info = adaptation.run(
        adapt_key, pro_position, num_steps=num_adapt_steps
    )
    adapted_kernel = jax.tree.map(lambda a: a[-1], adaptation_info.kernel)
    pro_params = adaptation_results.parameters

    basis_full = _cholesky_basis(adapted_kernel, case.x_train)
    key, pos_key = jr.split(key)
    pro_position = jr.normal(pos_key, (case.x_train.shape[0], num_particles))
    pro_params = pro_params._replace(y=case.y_train, basis=basis_full, residual_std=None)

    algorithm = parametric_ula(pro_logdensity_fn, pro_params)
    key, sample_key = jr.split(key)
    _, (states, _) = run_inference_algorithm(
        rng_key=sample_key,
        inference_algorithm=algorithm,
        num_steps=num_sample_steps,
        initial_position=pro_position,
        progress_bar=False,
    )
    burn = num_sample_steps // 2
    particles = states.position[burn::4]

    test_basis, test_cov = prediction_basis(
        adapted_kernel, case.x_train, case.x_test, pro_params, inducing_basis=None
    )
    pro_nlpd = float(nlpd_pro(case.y_test, test_basis, test_cov, particles, parameters=pro_params))
    pro_crps = float(crps_pro(case.y_test, test_basis, test_cov, particles, parameters=pro_params))

    return Metrics(gp_nlpd=gp_nlpd, gp_crps=gp_crps, pro_nlpd=pro_nlpd, pro_crps=pro_crps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", choices=sorted(FUNCTIONS), default="hartmann6")
    parser.add_argument("--corruption", choices=["constant", "uniform", "student_t"], default="constant")
    parser.add_argument("--outlier-fraction", type=float, default=0.05)
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--num-reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gp-num-iters", type=int, default=1000)
    parser.add_argument("--kernel-lr", type=float, default=0.01)
    parser.add_argument("--num-adapt-steps", type=int, default=1000)
    parser.add_argument("--num-sample-steps", type=int, default=2000)
    parser.add_argument("--num-particles", type=int, default=20)
    parser.add_argument("--out", type=str, default=None, help="Optional path to write JSON results.")
    args = parser.parse_args()

    key = jr.PRNGKey(args.seed)
    results: list[Metrics] = []
    for rep in range(args.num_reps):
        rep_key = jr.fold_in(key, rep)
        metrics = _run_one(
            rep_key,
            args.function,
            args.corruption,
            args.outlier_fraction,
            n_train=args.n_train,
            n_test=args.n_test,
            sigma=args.sigma,
            gp_num_iters=args.gp_num_iters,
            kernel_lr=args.kernel_lr,
            num_adapt_steps=args.num_adapt_steps,
            num_sample_steps=args.num_sample_steps,
            num_particles=args.num_particles,
        )
        results.append(metrics)
        print(
            f"[rep {rep}] GP  NLPD={metrics.gp_nlpd:.4f} CRPS={metrics.gp_crps:.4f}  |  "
            f"PRO NLPD={metrics.pro_nlpd:.4f} CRPS={metrics.pro_crps:.4f}"
        )

    gp_nlpd = np.array([m.gp_nlpd for m in results])
    pro_nlpd = np.array([m.pro_nlpd for m in results])
    print(
        f"\nGP  NLPD: mean={gp_nlpd.mean():.4f} std={gp_nlpd.std():.4f}\n"
        f"PRO NLPD: mean={pro_nlpd.mean():.4f} std={pro_nlpd.std():.4f}"
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump([m.__dict__ for m in results], f, indent=2)


if __name__ == "__main__":
    main()
