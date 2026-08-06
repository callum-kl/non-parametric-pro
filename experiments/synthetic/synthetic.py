import json
import logging
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

import gpjax as gpx
import hydra
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax as ox
import paramax as px
from fastprogress.fastprogress import progress_bar
from omegaconf import DictConfig, OmegaConf

from non_parametric_pro import ula
from non_parametric_pro.data.heteroskedastic import (
    heteroskedastic_noise_std,
    make_heteroskedastic_instance,
)
from non_parametric_pro.density import (
    ProParameters,
    pro_logdensity_fn,
    regularised_score,
)
from non_parametric_pro.parameter_adaptation import cross_validated_parameter_adaptation
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import (
    cholesky_basis,
    nlpd_gp,
    nlpd_pro,
    prediction_basis,
    run_inference_algorithm_with_burn_in,
)

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parent), replace=True
)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def _plot_case(ax, data):
    """
    Plot one HeteroskedasticCase: true curve, +-1sigma/+-2sigma noise bands (both
    evaluated on the full, sorted train+test pool for a smooth curve), and the
    noisy observed training points.
    """
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


def make_heteroskedastic_panel(
    key, *, get_instance=make_heteroskedastic_instance, num_instances=20, grid_shape=(5, 4)
):
    keys = jr.split(key, num_instances)
    nrows, ncols = grid_shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for ax, instance_key in zip(axes.flat, keys, strict=True):
        data = get_instance(instance_key)
        _plot_case(ax, data)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heteroskedastic_panel.png", dpi=150)


def run_heteroskedastic(key, *, get_instance=make_heteroskedastic_instance, num_instances=20):
    keys = jr.split(key, num_instances)

    fig, ax = plt.subplots()

    all_data = []
    for instance_key in keys:
        all_data.append(get_instance(instance_key))

    _plot_case(ax, all_data[12])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heteroskedastic_instance.png", dpi=150)

def fit_gp(data, key=None, *, kernel_lengthscale=0.3):

    x_train, y_train = data.x_train, data.y_train
    x_test, y_test = data.x_test, data.y_test
    gp_data = gpx.Dataset(X=x_train, y=y_train)

    kernel = gpx.kernels.RBF(lengthscale=kernel_lengthscale)
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

def fit_pro(  # noqa: PLR0913
    data,
    key,
    *,
    kernel_lengthscale=0.3,
    step_size=0.0001,
    alpha=1.0,
    sigma_init=0.4,
    sigma_min=0.1,
    sigma_max=1.0,
    num_particles=32,
    num_folds=1,
    val_fraction=0.25,
    num_adapt_steps=10000,
    warmup_steps=1000,
    sigma_adapt_steps=200,
    kernel_adapt_steps=100,
    sigma_lr=0.1,
    kernel_lr=0.01,
    num_sample_steps=20000,
    burn_fraction=0.9,
    thin=10,
):
    x_train, y_train = data.x_train, data.y_train
    x_test, y_test = data.x_test, data.y_test

    sigma = gpx.parameters.SigmoidBounded(sigma_init, low=sigma_min, high=sigma_max)
    pro_params = ProParameters(
        y=y_train,
        basis=None,
        step_size=step_size,
        sigma=sigma,
        alpha=alpha,
        residual_std=None,
    )
    kernel = gpx.kernels.RBF(
        lengthscale=kernel_lengthscale, variance=px.NonTrainable(jnp.array(1.0))
    )
    key, cv_key = jr.split(key)
    cv_result = cross_validated_parameter_adaptation(
        ula,
        pro_logdensity_fn,
        pro_params,
        x_full=x_train,
        y_full=y_train,
        initial_kernel=kernel,
        num_folds=num_folds,
        val_fraction=val_fraction,
        num_particles=num_particles,
        num_steps=num_adapt_steps,
        warmup_steps=warmup_steps,
        sigma_adapt_steps=sigma_adapt_steps,
        kernel_adapt_steps=kernel_adapt_steps,
        objective_fn=regularised_score,
        rng_key=cv_key,
        sigma_optimizer=ox.adam(sigma_lr),
        kernel_optimizer=ox.adam(kernel_lr),
        progress_bar=False,
    )
    adapted_kernel = cv_result.kernel
    adapted_sigma_val = float(np.array(cv_result.sigma).reshape(()))

    # recompute basis on full training set, using adapted kernel
    key, pos_key = jr.split(key)
    basis = cholesky_basis(adapted_kernel, x_train)
    basis_dim = x_train.shape[0]
    pro_position = jr.normal(pos_key, (basis_dim, num_particles))
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
        num_steps=num_sample_steps,
        burn_ratio=burn_fraction,
        initial_position=pro_position,
        progress_bar=False,
    )

    # evaluate on test data
    particles = states.position[::thin]
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

    mean = float(np.mean(nlpds))
    std = float(np.std(nlpds))

    log.info("NLPD: %.4f±%.4f", mean, std)
    return mean, std, [float(v) for v in nlpds]


# `source` -> kwarg names each dataset's yaml (conf/ds/*.yaml) is allowed to set on
# top of `source` itself; only these are forwarded to the underlying instance
# generator, so a ds config can override any subset without a code change here.
_HETEROSKEDASTIC_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_floor_frac_range",
    "amplitude_frac", "min_width", "max_width",
    "ell_range", "alpha_range", "num_regions",
)


def _heteroskedastic_instance_fn(cfg: DictConfig):
    kwargs = {k: cfg[k] for k in _HETEROSKEDASTIC_KWARGS if k in cfg}
    return lambda key: make_heteroskedastic_instance(key, **kwargs)


_DATASET_SOURCES = {
    "heteroskedastic": _heteroskedastic_instance_fn,
}

def _fit_gp_fn(cfg: DictConfig):
    return lambda data, key=None: fit_gp(data, key=key, kernel_lengthscale=cfg.kernel.lengthscale)


def _fit_pro_fn(cfg: DictConfig):
    return lambda data, key: fit_pro(
        data,
        key,
        kernel_lengthscale=cfg.kernel.lengthscale,
        step_size=cfg.pro.step_size,
        alpha=cfg.pro.alpha,
        sigma_init=cfg.pro.sigma_init,
        sigma_min=cfg.pro.sigma_min,
        sigma_max=cfg.pro.sigma_max,
        num_particles=cfg.pro.num_particles,
        num_folds=cfg.pro.num_folds,
        val_fraction=cfg.pro.val_fraction,
        num_adapt_steps=cfg.pro.num_adapt_steps,
        warmup_steps=cfg.pro.warmup_steps,
        sigma_adapt_steps=cfg.pro.sigma_adapt_steps,
        kernel_adapt_steps=cfg.pro.kernel_adapt_steps,
        sigma_lr=cfg.pro.sigma_lr,
        kernel_lr=cfg.pro.kernel_lr,
        num_sample_steps=cfg.pro.num_sample_steps,
        burn_fraction=cfg.pro.burn_fraction,
        thin=cfg.pro.thin,
    )


_FIT_ALGORITHMS = {
    "standard_gp": _fit_gp_fn,
    "pro_gp": _fit_pro_fn,
}


def _get_instance_fn(cfg: DictConfig):
    try:
        build = _DATASET_SOURCES[cfg.source]
    except KeyError:
        msg = f"Dataset {cfg.source} not supported"
        raise ValueError(msg) from None
    return build(cfg)


def _get_fit_algorithm(cfg: DictConfig):
    try:
        build = _FIT_ALGORITHMS[cfg.algorithm]
    except KeyError:
        msg = f"Algorithm {cfg.algorithm} not supported"
        raise ValueError(msg) from None
    return build(cfg)


def out_dir(cfg: DictConfig, param_value) -> Path:
    """Results land under results_root/<source>/<param_name>_<param_value>/<algorithm>,
    keyed by each dataset's own declared `param_name` (e.g. `num_regions` for
    heteroskedastic) rather than a name hardcoded here, since it differs per dataset."""
    return (
        Path(cfg.results_root)
        / cfg.source
        / f"{cfg.param_name}_{param_value}"
        / cfg.algorithm
    )


@hydra.main(version_base=None, config_path="conf", config_name="synthetic")
def main(cfg: DictConfig) -> None:
    get_instance = _get_instance_fn(cfg)
    key = jr.PRNGKey(cfg.seed)

    if cfg.mode == "panel":
        make_heteroskedastic_panel(
            key,
            get_instance=get_instance,
            num_instances=cfg.panel.num_instances,
            grid_shape=tuple(cfg.panel.grid_shape),
        )
        log.info("Saved panel to %s", FIGURES_DIR / "heteroskedastic_panel.png")
        return

    fit_algorithm = _get_fit_algorithm(cfg)
    param_value = cfg[cfg.param_name]

    log.info(
        "Evaluating %s on %s (%s=%s)", cfg.algorithm, cfg.source, cfg.param_name, param_value
    )
    mean, std, nlpds = evaluate(get_instance, fit_algorithm, key, cfg.num_instances)

    metrics = {
        "source": cfg.source,
        "algorithm": cfg.algorithm,
        "param_name": cfg.param_name,
        "param_value": param_value,
        "num_instances": cfg.num_instances,
        "seed": cfg.seed,
        "nlpd_mean": mean,
        "nlpd_std": std,
        "nlpds": nlpds,
    }

    results_dir = out_dir(cfg, param_value)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(results_dir / "config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", results_dir)


if __name__ == "__main__":
    main()
