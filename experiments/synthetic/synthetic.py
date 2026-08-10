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
from non_parametric_pro.data.extrapolation import (
    EXTRAPOLATION_KWARGS,
    extrapolation_region_mask,
    make_extrapolation_instance,
    plot_extrapolation_case,
)
from non_parametric_pro.data.heavy_tailed import (
    HEAVY_TAILED_KWARGS,
    make_heavy_tailed_instance,
    plot_heavy_tailed_case,
)
from non_parametric_pro.data.heteroskedastic import (
    HETEROSKEDASTIC_KWARGS,
    heteroskedastic_region_mask,
    make_heteroskedastic_instance,
    plot_heteroskedastic_case,
)
from non_parametric_pro.data.huber import HUBER_KWARGS, make_huber_instance, plot_huber_case
from non_parametric_pro.data.interpolation_gap import (
    INTERPOLATION_GAP_KWARGS,
    interpolation_gap_mask,
    make_interpolation_instance,
    plot_interpolation_case,
)
from non_parametric_pro.data.linear_mismatch import (
    LINEAR_MISMATCH_KWARGS,
    make_linear_mismatch_instance,
    plot_linear_mismatch_case,
)
from non_parametric_pro.data.multimodal import (
    MULTIMODAL_KWARGS,
    make_multimodal_instance,
    multimodal_region_mask,
    plot_multimodal_case,
)
from non_parametric_pro.data.regime_switch import (
    REGIME_SWITCH_KWARGS,
    make_regime_switch_instance,
    plot_regime_switch_case,
    regime_switch_region_mask,
)
from non_parametric_pro.data.saturation import (
    SATURATION_KWARGS,
    make_saturation_instance,
    plot_saturation_case,
)
from non_parametric_pro.data.skewed import SKEWED_KWARGS, make_skewed_instance, plot_skewed_case
from non_parametric_pro.data.well_specified import (
    WELL_SPECIFIED_KWARGS,
    build_kernel,
    make_well_specified_instance,
    plot_well_specified_case,
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


# `source` -> `(ax, data) -> None` plotting function, used by `make_dataset_panel`/
# `run_dataset_instance`. Each plot function lives next to its dataset's generator (in
# `non_parametric_pro.data.<source>`), since it's tied to that dataset's specific case
# fields (noise regions vs. two candidate functions + hidden mode, etc.) -- there's no
# shared generic plot the way there's a generic `_instance_fn` below.
_PLOT_CASE_FNS = {
    "heteroskedastic": plot_heteroskedastic_case,
    "multimodal": plot_multimodal_case,
    "regime_switch": plot_regime_switch_case,
    "heavy_tailed": plot_heavy_tailed_case,
    "interpolation_gap": plot_interpolation_case,
    "huber": plot_huber_case,
    "skewed": plot_skewed_case,
    "saturation": plot_saturation_case,
    "linear_mismatch": plot_linear_mismatch_case,
    "extrapolation": plot_extrapolation_case,
    "well_specified": plot_well_specified_case,
}


def _get_plot_case_fn(cfg: DictConfig):
    try:
        return _PLOT_CASE_FNS[cfg.source]
    except KeyError:
        msg = f"No plot function registered for dataset {cfg.source!r}"
        raise ValueError(msg) from None


def make_dataset_panel(
    key,
    *,
    get_instance=make_heteroskedastic_instance,
    plot_case_fn=plot_heteroskedastic_case,
    num_instances=20,
    grid_shape=(5, 4),
    filename="heteroskedastic_panel.png",
):
    keys = jr.split(key, num_instances)
    nrows, ncols = grid_shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for ax, instance_key in zip(axes.flat, keys, strict=True):
        data = get_instance(instance_key)
        plot_case_fn(ax, data)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)


def run_dataset_instance(
    key,
    *,
    get_instance=make_heteroskedastic_instance,
    plot_case_fn=plot_heteroskedastic_case,
    num_instances=20,
    filename="heteroskedastic_instance.png",
):
    keys = jr.split(key, num_instances)

    fig, ax = plt.subplots()

    all_data = []
    for instance_key in keys:
        all_data.append(get_instance(instance_key))

    plot_case_fn(ax, all_data[12])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)

def fit_gp(data, key=None, *, kernel_lengthscale=0.3, kernel_type="rbf"):

    x_train, y_train = data.x_train, data.y_train
    x_test, y_test = data.x_test, data.y_test
    gp_data = gpx.Dataset(X=x_train, y=y_train)

    # `kernel_type` defaults to "rbf" (matching every dataset except well_specified,
    # which is the one place fitting deliberately matches the data's own generative
    # kernel family -- see `_fit_gp_fn`, which pulls it from `data.kernel_type`).
    kernel = build_kernel(kernel_type, lengthscale=kernel_lengthscale)
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

    return nlpd_gp(y_test, mean, std, return_per_point=True)

def fit_pro(  # noqa: PLR0913
    data,
    key,
    *,
    kernel_lengthscale=0.3,
    kernel_type="rbf",
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
    # kernel = gpx.kernels.RBF(
    #     lengthscale=kernel_lengthscale, variance=px.NonTrainable(jnp.array(1.0))
    # )
    kernel = build_kernel(kernel_type, lengthscale=kernel_lengthscale)
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
    return nlpd_pro(
        y_test, test_basis, test_cov, particles, parameters=pro_params, return_per_point=True
    )


def evaluate(get_instance, fit_function, key, num_instances, *, region_mask_fn=None):
    """Fit/evaluate `num_instances` draws, returning (mean, std, nlpds, region_nlpds).

    `fit_function` now returns *per-test-point* NLPD (not a scalar); the overall
    mean/std/nlpds are unchanged in meaning (mean over all test points per instance).
    If `region_mask_fn(data) -> bool array` (matching `data.x_test`'s order) is given,
    also split each instance's per-point NLPD into a "region" mean and a "background"
    (non-region) mean -- e.g. to check whether a method pays a tax in well-specified
    regions in exchange for doing better in misspecified ones. An instance with zero
    test points on one side of the split contributes NaN for that side there, rather
    than skewing the mean with a fabricated value.
    """
    keys = jr.split(key, num_instances)

    nlpds = []
    region_nlpds = {"region": [], "background": []} if region_mask_fn is not None else None

    for instance_key in progress_bar(keys):
        data = get_instance(instance_key)
        fit_key, instance_key = jr.split(instance_key)
        nlpd_per_point = fit_function(data, fit_key)
        nlpds.append(float(jnp.mean(nlpd_per_point)))

        if region_mask_fn is not None:
            mask = region_mask_fn(data)
            in_region, out_region = nlpd_per_point[mask], nlpd_per_point[~mask]
            region_nlpds["region"].append(
                float(jnp.mean(in_region)) if in_region.size else float("nan")
            )
            region_nlpds["background"].append(
                float(jnp.mean(out_region)) if out_region.size else float("nan")
            )

    mean = float(np.mean(nlpds))
    std = float(np.std(nlpds))

    log.info("NLPD: %.4f±%.4f", mean, std)
    return mean, std, nlpds, region_nlpds


_DATASET_SOURCES = {
    "heteroskedastic": (make_heteroskedastic_instance, HETEROSKEDASTIC_KWARGS),
    "multimodal": (make_multimodal_instance, MULTIMODAL_KWARGS),
    "regime_switch": (make_regime_switch_instance, REGIME_SWITCH_KWARGS),
    "heavy_tailed": (make_heavy_tailed_instance, HEAVY_TAILED_KWARGS),
    "interpolation_gap": (make_interpolation_instance, INTERPOLATION_GAP_KWARGS),
    "huber": (make_huber_instance, HUBER_KWARGS),
    "skewed": (make_skewed_instance, SKEWED_KWARGS),
    "saturation": (make_saturation_instance, SATURATION_KWARGS),
    "linear_mismatch": (make_linear_mismatch_instance, LINEAR_MISMATCH_KWARGS),
    "extrapolation": (make_extrapolation_instance, EXTRAPOLATION_KWARGS),
    "well_specified": (make_well_specified_instance, WELL_SPECIFIED_KWARGS),
}


def _instance_fn(make_instance, kwarg_names, cfg: DictConfig):
    """Build a `key -> data` closure for `make_instance`, forwarding only the subset of
    `kwarg_names` that `cfg` actually sets -- so a `ds` config can override any subset
    of a generator's parameters without a code change here."""
    kwargs = {k: cfg[k] for k in kwarg_names if k in cfg}
    return lambda key: make_instance(key, **kwargs)


def _heteroskedastic_region_mask_fn(data):
    return heteroskedastic_region_mask(data.x_test[:, 0], data.regions)


def _multimodal_region_mask_fn(data):
    return multimodal_region_mask(data.x_test[:, 0], data.regions)


def _regime_switch_region_mask_fn(data):
    return regime_switch_region_mask(data.x_test[:, 0], data.regions)


def _interpolation_gap_region_mask_fn(data):
    return interpolation_gap_mask(data.x_test[:, 0], data.gap_center, data.gap_width)


def _extrapolation_region_mask_fn(data):
    return extrapolation_region_mask(data.x_test[:, 0], data.boundary, data.extrapolate_right)


_REGION_MASK_FNS = {
    "heteroskedastic": _heteroskedastic_region_mask_fn,
    "multimodal": _multimodal_region_mask_fn,
    "regime_switch": _regime_switch_region_mask_fn,
    "interpolation_gap": _interpolation_gap_region_mask_fn,
    "extrapolation": _extrapolation_region_mask_fn,
}


def _get_region_mask_fn(cfg: DictConfig):
    return _REGION_MASK_FNS.get(cfg.source)


def _fit_gp_fn(cfg: DictConfig):
    # `data.kernel_type` only exists on WellSpecifiedCase -- every other dataset fits a
    # fixed generic RBF regardless of (or *because* it doesn't match) how the data was
    # generated, which is the whole point of those datasets being misspecified.
    return lambda data, key=None: fit_gp(
        data,
        key=key,
        kernel_lengthscale=cfg.kernel.lengthscale,
        kernel_type=getattr(data, "kernel_type", "rbf"),
    )


def _fit_pro_fn(cfg: DictConfig):
    return lambda data, key: fit_pro(
        data,
        key,
        kernel_lengthscale=cfg.kernel.lengthscale,
        kernel_type=getattr(data, "kernel_type", "rbf"),
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
        make_instance, kwarg_names = _DATASET_SOURCES[cfg.source]
    except KeyError:
        msg = f"Dataset {cfg.source} not supported"
        raise ValueError(msg) from None
    return _instance_fn(make_instance, kwarg_names, cfg)


def _get_fit_algorithm(cfg: DictConfig):
    try:
        build = _FIT_ALGORITHMS[cfg.algorithm]
    except KeyError:
        msg = f"Algorithm {cfg.algorithm} not supported"
        raise ValueError(msg) from None
    return build(cfg)


def debug_instance(cfg: DictConfig) -> None:
    """Re-run a single instance from an `evaluate` run by index, reproducing exactly
    the same draw and fit `evaluate` used -- for isolating an outlier NLPD spotted in
    a saved metrics.json (`nlpds[i]`) without re-running the whole sweep.

    Pass the *same* seed/source/algorithm/param overrides the original run used (they
    round-trip through the saved `config.json`) plus `mode=instance instance_index=i`;
    `jr.split(key, num_instances)` is deterministic, so `keys[i]` -- and everything
    downstream of it -- reproduces bit-for-bit.
    """
    get_instance = _get_instance_fn(cfg)
    fit_algorithm = _get_fit_algorithm(cfg)
    plot_case_fn = _get_plot_case_fn(cfg)

    key = jr.PRNGKey(cfg.seed)
    keys = jr.split(key, cfg.num_instances)
    instance_key = keys[cfg.instance_index]

    data = get_instance(instance_key)
    fit_key, _ = jr.split(instance_key)
    nlpd_per_point = fit_algorithm(data, fit_key)
    nlpd = float(jnp.mean(nlpd_per_point))
    log.info(
        "Instance %d/%d (%s, %s=%s): NLPD=%.4f",
        cfg.instance_index, cfg.num_instances, cfg.algorithm, cfg.param_name,
        cfg[cfg.param_name], nlpd,
    )

    fig, ax = plt.subplots()
    plot_case_fn(ax, data)
    filename = f"{cfg.source}_instance{cfg.instance_index}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    log.info("Saved instance plot to %s", FIGURES_DIR / filename)


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
    if cfg.mode == "instance":
        debug_instance(cfg)
        return

    get_instance = _get_instance_fn(cfg)
    key = jr.PRNGKey(cfg.seed)

    if cfg.mode == "panel":
        plot_case_fn = _get_plot_case_fn(cfg)
        filename = f"{cfg.source}_panel.png"
        make_dataset_panel(
            key,
            get_instance=get_instance,
            plot_case_fn=plot_case_fn,
            num_instances=cfg.panel.num_instances,
            grid_shape=tuple(cfg.panel.grid_shape),
            filename=filename,
        )
        log.info("Saved panel to %s", FIGURES_DIR / filename)
        return

    fit_algorithm = _get_fit_algorithm(cfg)
    param_value = cfg[cfg.param_name]

    log.info(
        "Evaluating %s on %s (%s=%s)", cfg.algorithm, cfg.source, cfg.param_name, param_value
    )
    region_mask_fn = _get_region_mask_fn(cfg)
    mean, std, nlpds, region_nlpds = evaluate(
        get_instance, fit_algorithm, key, cfg.num_instances, region_mask_fn=region_mask_fn
    )

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
    if region_nlpds is not None:
        # Per-instance mean NLPD split by `_get_region_mask_fn`'s mask: "region" =
        # inside a noise-elevation region (misspecified for a homoscedastic-noise
        # model), "background" = outside all of them (well-specified). NaN entries
        # mean that instance had zero test points on that side of the split.
        metrics["region_nlpds"] = region_nlpds

    results_dir = out_dir(cfg, param_value)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(results_dir / "config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)

    log.info("Saved results to %s", results_dir)


if __name__ == "__main__":
    main()
