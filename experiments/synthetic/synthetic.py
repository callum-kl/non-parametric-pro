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
    extrapolation_region_mask,
    make_extrapolation_instance,
)
from non_parametric_pro.data.heavy_tailed import make_heavy_tailed_instance
from non_parametric_pro.data.heteroskedastic import (
    heteroskedastic_noise_std,
    heteroskedastic_region_mask,
    make_heteroskedastic_instance,
)
from non_parametric_pro.data.huber import make_huber_instance
from non_parametric_pro.data.hyperparameter_ambiguous import (
    make_hyperparameter_ambiguous_instance,
)
from non_parametric_pro.data.interpolation_gap import (
    interpolation_gap_mask,
    make_interpolation_instance,
)
from non_parametric_pro.data.linear_mismatch import make_linear_mismatch_instance
from non_parametric_pro.data.multimodal import (
    make_multimodal_instance,
    multimodal_region_mask,
)
from non_parametric_pro.data.regime_switch import (
    make_regime_switch_instance,
    regime_switch_region_mask,
)
from non_parametric_pro.data.saturation import make_saturation_instance
from non_parametric_pro.data.skewed import make_skewed_instance
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


def _plot_heteroskedastic_case(ax, data):
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


def _plot_multimodal_case(ax, data):
    """
    Plot one MultimodalCase: the shared background curve (blue) and the alternate
    branch (orange) -- `y_alt` is constructed to already coincide with `y_shared`
    outside a region (see `make_multimodal_instance`), so plotting both over their full
    extent shows a genuine fork: the two curves visibly touch at each region's edges
    and diverge only toward its center, rather than needing to be cut off/hidden away
    from the region. Training points are colored by which branch actually generated
    them (recovering that split is exactly what a model *can't* do, since `z` is
    hidden -- this is a diagnostic only).
    """
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_shared_full = jnp.concatenate(
        [data.y_truth_shared_train[:, 0], data.y_truth_shared_test[:, 0]]
    )
    y_alt_full = jnp.concatenate([data.y_truth_alt_train[:, 0], data.y_truth_alt_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted = x_full[order]
    y_shared_sorted = y_shared_full[order]
    y_alt_sorted = y_alt_full[order]

    ax.plot(x_sorted, y_shared_sorted, color="C0", linewidth=1.5)
    ax.plot(x_sorted, y_alt_sorted, color="C1", linewidth=1.5)

    z_train = data.z_train
    ax.scatter(data.x_train[~z_train], data.y_train[~z_train], color="C0", s=8, zorder=3)
    ax.scatter(data.x_train[z_train], data.y_train[z_train], color="C1", s=8, zorder=3)
    ax.set_title(f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}", fontsize=9)


def _plot_regime_switch_case(ax, data):
    """
    Plot one RegimeSwitchCase: the single latent truth curve (no bands, no second
    branch -- it's an unambiguous, single-valued function throughout) and the noisy
    observed training points. Each region's span is lightly shaded so it's visible
    where the shorter-lengthscale (rougher) perturbation applies, since otherwise
    there's no color/branch cue the way there is for `_plot_multimodal_case`.
    """
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    for center, width in zip(data.regions.centers, data.regions.widths, strict=True):
        ax.axvspan(float(center - width), float(center + width), color="C1", alpha=0.15, linewidth=0)

    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\ell_r$={data.ell_region:.2f}  $\\alpha$={data.alpha:.2f}",
        fontsize=9,
    )


def _plot_heavy_tailed_case(ax, data):
    """
    Plot one HeavyTailedCase: the single latent truth curve and observed points, applied
    over the whole domain (no region -- see the module docstring on why this dataset
    tests a *global* rather than local misspecification). Points more than 2 noise stds
    from the curve are highlighted (orange) purely as a visual diagnostic of where the
    heavy tails show up in this particular draw -- it's not a structural split like the
    other datasets' region masks, just `|residual| > 2*noise_std`.
    """
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    residual = data.y_train[:, 0] - data.y_truth_train[:, 0]
    is_outlier = jnp.abs(residual) > 2 * data.noise_std
    ax.scatter(data.x_train[~is_outlier], data.y_train[~is_outlier], color="black", s=8, zorder=3)
    ax.scatter(data.x_train[is_outlier], data.y_train[is_outlier], color="C1", s=8, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  $\\nu$={data.noise_df:.1f}",
        fontsize=9,
    )


def _plot_huber_case(ax, data):
    """
    Plot one HuberCase: the single latent truth curve and observed points, colored by
    the *true* (hidden) contamination indicator -- orange = drawn from the wide
    outlier-scale Gaussian, black = ordinary noise. Applied over the whole domain (no
    region), matching `heavy_tailed.py`'s final form -- gross errors aren't spatially
    correlated with x in the classical Huber model. Unlike `_plot_heavy_tailed_case`
    (which has to guess outliers via a residual threshold), this dataset actually
    generates a ground-truth contamination label, so we use it directly.
    """
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    is_outlier = data.is_outlier_train
    ax.scatter(data.x_train[~is_outlier], data.y_train[~is_outlier], color="black", s=8, zorder=3)
    ax.scatter(data.x_train[is_outlier], data.y_train[is_outlier], color="C1", s=8, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  $\\epsilon$={data.contamination_prob:.2f}",
        fontsize=9,
    )


def _plot_skewed_case(ax, data):
    """
    Plot one SkewedCase: the single latent truth curve and observed points, applied
    over the whole domain (no region -- skewness isn't spatially correlated with x any
    more than heavy tails or contamination are). Points more than 1.5 noise stds out on
    the *long-tail side* (using the instance's known `skew_sign`, not a symmetric
    threshold like `_plot_heavy_tailed_case`'s) are highlighted -- the asymmetry should
    show up as most of the highlighted points sitting on one side of the curve, not
    scattered evenly above and below it.
    """
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    residual = data.y_train[:, 0] - data.y_truth_train[:, 0]
    is_long_tail = (data.skew_sign * residual) > 1.5 * data.noise_std
    ax.scatter(data.x_train[~is_long_tail], data.y_train[~is_long_tail], color="black", s=8, zorder=3)
    ax.scatter(data.x_train[is_long_tail], data.y_train[is_long_tail], color="C1", s=8, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  "
        f"skew={data.skew_sign * data.noise_skewness:+.1f}",
        fontsize=9,
    )


def _plot_saturation_case(ax, data):
    """
    Plot one SaturationCase: the single latent truth curve, dashed lines at the +/-
    saturation threshold, and observed points colored by the *true* (hidden) censoring
    indicator -- orange points sit exactly on one of the threshold lines (the classic
    "flat-lined" signature of a saturated sensor), black points are uncensored.
    """
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.axhline(data.threshold, color="C1", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axhline(-data.threshold, color="C1", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    is_censored = data.is_censored_train
    ax.scatter(data.x_train[~is_censored], data.y_train[~is_censored], color="black", s=8, zorder=3)
    ax.scatter(data.x_train[is_censored], data.y_train[is_censored], color="C1", s=8, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  thr={data.threshold:.2f}",
        fontsize=9,
    )


def _plot_linear_mismatch_case(ax, data):
    """
    Plot one LinearMismatchCase: the single latent truth curve (drawn from the
    RBF/Linear blend -- visibly wider swings toward the domain edges as `linear_mix`
    grows, since the Linear component's marginal variance grows away from `x=0`) and
    the noisy observed points. No special coloring -- there's no hidden branch,
    censoring, or outlier label here, just a single well-defined function whose
    *non-stationarity* is what's being tested.
    """
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  linear={data.linear_mix:.2f}",
        fontsize=9,
    )


def _plot_interpolation_case(ax, data):
    """
    Plot one InterpolationCase: the single latent truth curve, the gap's span shaded,
    training points (black), the interpolation-gap test points (orange, inside the
    shading -- the model never saw any training data anywhere nearby), and the ordinary
    random-holdout test points (gray, scattered like any other train/test split).
    """
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.axvspan(
        data.gap_center - data.gap_width / 2, data.gap_center + data.gap_width / 2,
        color="C1", alpha=0.15, linewidth=0,
    )
    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    in_gap = interpolation_gap_mask(data.x_test[:, 0], data.gap_center, data.gap_width)
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.scatter(data.x_test[~in_gap], data.y_test[~in_gap], color="gray", s=8, zorder=3)
    ax.scatter(data.x_test[in_gap], data.y_test[in_gap], color="C1", s=8, zorder=3)
    ax.set_title(f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}", fontsize=9)


def _plot_extrapolation_case(ax, data):
    """
    Plot one ExtrapolationCase: the single latent truth curve, the held-out edge
    shaded, training points (black), the extrapolation test points (orange, inside the
    shading -- beyond the edge of any training data, unlike `_plot_interpolation_case`'s
    gap which has data on both sides), and the ordinary random-holdout test points
    (gray, scattered like any other train/test split).
    """
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    edge = float(x_sorted.max()) if data.extrapolate_right else float(x_sorted.min())
    ax.axvspan(
        min(data.boundary, edge), max(data.boundary, edge), color="C1", alpha=0.15, linewidth=0
    )
    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)

    in_extrap = extrapolation_region_mask(data.x_test[:, 0], data.boundary, data.extrapolate_right)
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.scatter(data.x_test[~in_extrap], data.y_test[~in_extrap], color="gray", s=8, zorder=3)
    ax.scatter(data.x_test[in_extrap], data.y_test[in_extrap], color="C1", s=8, zorder=3)
    ax.set_title(f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}", fontsize=9)


def _plot_hyperparameter_ambiguous_case(ax, data):
    """
    Plot one HyperparameterAmbiguousCase: the true (short-lengthscale) curve evaluated
    densely over the combined train+test pool, the sparse black training points, and
    the much denser gray test points -- visually, the point is that the black points
    alone give little hint of how wiggly the true curve actually is *between* them.
    """
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.0, alpha=0.6)
    ax.scatter(data.x_test, data.y_test, color="gray", s=6, zorder=2)
    ax.scatter(data.x_train, data.y_train, color="black", s=20, zorder=3)
    ax.set_title(
        f"$\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  n={data.x_train.shape[0]}",
        fontsize=9,
    )


# `source` -> `(ax, data) -> None` plotting function, used by `make_dataset_panel`/
# `run_dataset_instance`. Each dataset's `data` struct has different fields (noise
# regions vs. two candidate functions + hidden mode), so unlike `_DATASET_SOURCES` etc.
# there's no shared generic plot -- each source needs its own.
_PLOT_CASE_FNS = {
    "heteroskedastic": _plot_heteroskedastic_case,
    "multimodal": _plot_multimodal_case,
    "regime_switch": _plot_regime_switch_case,
    "heavy_tailed": _plot_heavy_tailed_case,
    "interpolation_gap": _plot_interpolation_case,
    "huber": _plot_huber_case,
    "skewed": _plot_skewed_case,
    "saturation": _plot_saturation_case,
    "linear_mismatch": _plot_linear_mismatch_case,
    "extrapolation": _plot_extrapolation_case,
    "hyperparameter_ambiguous": _plot_hyperparameter_ambiguous_case,
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
    plot_case_fn=_plot_heteroskedastic_case,
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
    plot_case_fn=_plot_heteroskedastic_case,
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

    return nlpd_gp(y_test, mean, std, return_per_point=True)

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
    # kernel = gpx.kernels.RBF(
    #     lengthscale=kernel_lengthscale, variance=px.NonTrainable(jnp.array(1.0))
    # )
    kernel = gpx.kernels.RBF(
        lengthscale=kernel_lengthscale
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


_MULTIMODAL_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "mix_prob", "min_width", "max_width", "ell_range", "alpha_range", "num_regions",
)


def _multimodal_instance_fn(cfg: DictConfig):
    kwargs = {k: cfg[k] for k in _MULTIMODAL_KWARGS if k in cfg}
    return lambda key: make_multimodal_instance(key, **kwargs)


_REGIME_SWITCH_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "roughness_factor", "min_width", "max_width", "ell_range", "alpha_range", "num_regions",
)


def _regime_switch_instance_fn(cfg: DictConfig):
    kwargs = {k: cfg[k] for k in _REGIME_SWITCH_KWARGS if k in cfg}
    return lambda key: make_regime_switch_instance(key, **kwargs)


_HEAVY_TAILED_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "noise_df", "ell_range", "alpha_range",
)


def _heavy_tailed_instance_fn(cfg: DictConfig):
    kwargs = {k: cfg[k] for k in _HEAVY_TAILED_KWARGS if k in cfg}
    return lambda key: make_heavy_tailed_instance(key, **kwargs)


_INTERPOLATION_GAP_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "gap_frac", "ell_range", "alpha_range",
)


def _interpolation_gap_instance_fn(cfg: DictConfig):
    kwargs = {k: cfg[k] for k in _INTERPOLATION_GAP_KWARGS if k in cfg}
    return lambda key: make_interpolation_instance(key, **kwargs)


_HUBER_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "contamination_prob", "outlier_scale", "ell_range", "alpha_range",
)


def _huber_instance_fn(cfg: DictConfig):
    kwargs = {k: cfg[k] for k in _HUBER_KWARGS if k in cfg}
    return lambda key: make_huber_instance(key, **kwargs)


_SKEWED_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "noise_skewness", "ell_range", "alpha_range",
)


def _skewed_instance_fn(cfg: DictConfig):
    kwargs = {k: cfg[k] for k in _SKEWED_KWARGS if k in cfg}
    return lambda key: make_skewed_instance(key, **kwargs)


_SATURATION_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "threshold_frac", "ell_range", "alpha_range",
)


def _saturation_instance_fn(cfg: DictConfig):
    kwargs = {k: cfg[k] for k in _SATURATION_KWARGS if k in cfg}
    return lambda key: make_saturation_instance(key, **kwargs)


_LINEAR_MISMATCH_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "linear_mix", "ell_range", "alpha_range",
)


def _linear_mismatch_instance_fn(cfg: DictConfig):
    kwargs = {k: cfg[k] for k in _LINEAR_MISMATCH_KWARGS if k in cfg}
    return lambda key: make_linear_mismatch_instance(key, **kwargs)


_EXTRAPOLATION_KWARGS = (
    "n", "test_fraction", "x_min", "x_max", "noise_std_frac",
    "extrapolation_frac", "ell_range", "alpha_range",
)


def _extrapolation_instance_fn(cfg: DictConfig):
    kwargs = {k: cfg[k] for k in _EXTRAPOLATION_KWARGS if k in cfg}
    return lambda key: make_extrapolation_instance(key, **kwargs)


_HYPERPARAMETER_AMBIGUOUS_KWARGS = (
    "n_train", "n_test", "x_min", "x_max", "noise_std_frac", "ell_range", "alpha_range",
)


def _hyperparameter_ambiguous_instance_fn(cfg: DictConfig):
    kwargs = {k: cfg[k] for k in _HYPERPARAMETER_AMBIGUOUS_KWARGS if k in cfg}
    return lambda key: make_hyperparameter_ambiguous_instance(key, **kwargs)


_DATASET_SOURCES = {
    "heteroskedastic": _heteroskedastic_instance_fn,
    "multimodal": _multimodal_instance_fn,
    "regime_switch": _regime_switch_instance_fn,
    "heavy_tailed": _heavy_tailed_instance_fn,
    "interpolation_gap": _interpolation_gap_instance_fn,
    "huber": _huber_instance_fn,
    "skewed": _skewed_instance_fn,
    "saturation": _saturation_instance_fn,
    "linear_mismatch": _linear_mismatch_instance_fn,
    "extrapolation": _extrapolation_instance_fn,
    "hyperparameter_ambiguous": _hyperparameter_ambiguous_instance_fn,
}


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


# `source` -> `data -> bool mask over data.x_test` for region-conditional evaluation
# (see `evaluate`'s `region_mask_fn`). Sources with no natural "region" notion (or not
# yet wired up) are simply absent, and `_get_region_mask_fn` returns None for them --
# `main` treats that as "skip the region split" rather than an error. `heavy_tailed`,
# `huber`, `skewed`, `saturation`, and `linear_mismatch` are deliberately absent: their
# misspecification is global, not regional (see their module docstrings), so there's no
# "region" to split on. `hyperparameter_ambiguous` is deliberately absent too -- its
# test points are drawn independently and densely, not split into a distinguished
# region vs. background.
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
