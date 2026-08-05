import argparse
import os
from pathlib import Path

# Must be set before `import jax` (and before any transitive jax import, e.g. via
# gpjax) -- setting it later doesn't reliably take effect before jax's backend
# initializes.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

# This script only ever calls savefig, never show(); force a non-interactive backend
# so it doesn't depend on a GUI toolkit being usable (e.g. Qt's xcb plugin, which
# aborts the whole process if there's no X server -- as under plain WSL).
matplotlib.use("Agg")

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

import gpjax as gpx
import optax as ox
import paramax as px

from non_parametric_pro import ula
from non_parametric_pro.density import ProParameters, pro_logdensity_fn, regularised_score
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import prediction_basis, nlpd_gp, nlpd_pro
from non_parametric_pro.parameter_adaptation import cross_validated_parameter_adaptation
from non_parametric_pro.util import run_inference_algorithm_with_burn_in, cholesky_basis

from non_parametric_pro.data.heteroskedastic import (
    heteroskedastic_noise_std,
    make_heteroskedastic_instance,
)

from fastprogress.fastprogress import progress_bar

DEFAULT_SEED = 2421
NUM_INSTANCES = 20
NUM_PARTICLES = 32
GRID_SHAPE = (5, 4)  # rows, cols -- rows * cols must equal NUM_INSTANCES
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def _plot_case(ax, data):
    """Plot one HeteroskedasticCase: true curve, +-1sigma/+-2sigma noise bands (both
    evaluated on the full, sorted train+test pool for a smooth curve), and the
    noisy observed training points."""
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_truth_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_truth_sorted = x_full[order], y_truth_full[order]
    sigma_sorted = heteroskedastic_noise_std(x_sorted, data.regions, noise_floor=data.noise_floor)

    ax.plot(x_sorted, y_truth_sorted, color="C0", linewidth=1.5)
    ax.fill_between(
        x_sorted, y_truth_sorted - 2 * sigma_sorted, y_truth_sorted + 2 * sigma_sorted,
        color="C0", alpha=0.15, linewidth=0,
    )
    ax.fill_between(
        x_sorted, y_truth_sorted - sigma_sorted, y_truth_sorted + sigma_sorted,
        color="C0", alpha=0.3, linewidth=0,
    )
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.set_title(
        f"$\\sigma_0$={data.noise_floor:.2f}  $\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}",
        fontsize=9,
    )


def make_heteroscedastic_panel(key):
    keys = jr.split(key, NUM_INSTANCES)
    nrows, ncols = GRID_SHAPE
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for ax, instance_key in zip(axes.flat, keys, strict=True):
        data = make_heteroskedastic_instance(instance_key)
        _plot_case(ax, data)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heteroskedastic_panel.png", dpi=150)


def run_heteroscedastic(key):
    keys = jr.split(key, NUM_INSTANCES)

    fig, ax = plt.subplots()

    all_data = []
    for instance_key in keys:
        all_data.append(make_heteroskedastic_instance(instance_key))

    _plot_case(ax, all_data[12])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heteroskedastic_instance.png", dpi=150)

def fit_gp(data, key=None):

    x_train, y_train = data.x_train, data.y_train
    x_test, y_test = data.x_test, data.y_test
    gp_data = gpx.Dataset(X=x_train, y=y_train)

    kernel = gpx.kernels.RBF(lengthscale=0.3)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=x_train.shape[0])
    posterior = likelihood * prior

    opt_posterior, _ = gpx.fit_scipy(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=gp_data,
        verbose=False
    )

    opt_kernel = opt_posterior.prior.kernel
    opt_sigma = opt_posterior.likelihood.obs_stddev

    latent = opt_posterior.predict(x_test, train_data=gp_data)
    predictive = opt_posterior.likelihood(latent)
    mean = predictive.mean
    std = jnp.sqrt(predictive.variance)

    return nlpd_gp(y_test, mean, std)

def fit_pro(data, key):

    x_train, y_train = data.x_train, data.y_train
    x_test, y_test = data.x_test, data.y_test

    sigma = gpx.parameters.SigmoidBounded(0.4, low=0.1, high=1.0)
    pro_params = ProParameters(
        y=y_train,
        basis=None,
        step_size=0.0001,
        sigma=sigma,
        alpha=1.0,
        residual_std=None,
    )
    kernel = gpx.kernels.RBF(lengthscale=0.3, variance=px.NonTrainable(jnp.array(1.0)))
    key, cv_key = jr.split(key)
    cv_result = cross_validated_parameter_adaptation(
        ula,
        pro_logdensity_fn,
        pro_params,
        x_full=x_train,
        y_full=y_train,
        initial_kernel=kernel,
        num_folds=1,
        val_fraction=0.25,
        num_particles=NUM_PARTICLES,
        num_steps=10000,
        warmup_steps=1000,
        sigma_adapt_steps=200,
        kernel_adapt_steps=100,
        objective_fn=regularised_score,
        rng_key=cv_key,
        sigma_optimizer=ox.adam(0.1),
        kernel_optimizer=ox.adam(0.01),
        progress_bar=False,
    )
    adapted_kernel = cv_result.kernel
    adapted_sigma_val = float(np.array(cv_result.sigma).reshape(()))

    # recompute basis on full training set, using adapted kernel
    key, pos_key = jr.split(key)
    basis = cholesky_basis(adapted_kernel, x_train)
    basis_dim = x_train.shape[0]
    pro_position = jr.normal(pos_key, (basis_dim, NUM_PARTICLES))
    pro_params = pro_params._replace(
        basis=basis,
        sigma=adapted_sigma_val,
    )

    # rerun sampling on full training set
    key, sample_key = jr.split(key)
    algorithm = parametric_ula(pro_logdensity_fn, pro_params)
    _, (states, _) = run_inference_algorithm_with_burn_in(
        rng_key=sample_key,
        inference_algorithm=algorithm,
        num_steps=10000,
        burn_ratio=0.9,
        initial_position=pro_position,
        progress_bar=False,
    )

    # evaluate on test data
    particles = states.position[::10]
    test_basis, test_cov = prediction_basis(
        adapted_kernel, x_train, x_test, pro_params
    )
    nlpd = float(nlpd_pro(y_test, test_basis, test_cov, particles, parameters=pro_params))

    return nlpd

def evaluate(get_instance, fit_function, key, num_instances):

    keys = jr.split(key, num_instances)

    nlpds = []

    for instance_key in progress_bar(keys):
        data = get_instance(instance_key)
        fit_key, instance_key = jr.split(instance_key)
        nlpd = fit_function(data, fit_key)
        nlpds.append(nlpd)

    mean = np.mean(nlpds)
    std = np.std(nlpds)

    print('\n')
    print("--------------------------------------")
    print(f"NLPD: {mean:.4f}±{std:.4f}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source", type=str, default="heteroskedastic")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--algorithm", type=str, default="standard_gp")
    args = parser.parse_args()

    if args.source == "heteroskedastic":
        get_instance = make_heteroskedastic_instance

    if args.algorithm == "standard_gp":
        fit_algorithm = fit_gp
    elif args.algorithm == "pro_gp":
        fit_algorithm = fit_pro
    else:
        raise ValueError(f"Algorithm {args.algorithm} not suppored")

    key = jr.PRNGKey(args.seed)
    print(f"Evaluating {args.algorithm}")
    evaluate(make_heteroskedastic_instance, fit_algorithm, key, args.n)

