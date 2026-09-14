import json
import logging
import os
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px
from fastprogress.fastprogress import progress_bar
from omegaconf import DictConfig, OmegaConf

from non_parametric_pro import replica_gibbs, ula
from non_parametric_pro.data.synthetic.block_outliers import (
    BLOCK_OUTLIERS_KWARGS,
    make_block_outlier_instance,
)
from non_parametric_pro.data.synthetic.heteroskedastic import (
    HETEROSKEDASTIC_KWARGS,
    make_heteroskedastic_instance,
)
from non_parametric_pro.data.synthetic.multimodal import (
    MULTIMODAL_KWARGS,
    make_multimodal_instance,
)
from non_parametric_pro.data.synthetic.well_specified import (
    WELL_SPECIFIED_KWARGS,
    build_kernel,
    make_well_specified_instance,
)
from non_parametric_pro.density import (
    ProParameters,
    pro_logdensity_fn,
)
from non_parametric_pro.inducing import PointInducingBasis
from non_parametric_pro.parameter_adaptation import parameter_adaptation
from non_parametric_pro.replica_gibbs import parametric_replica_gibbs, validate_r
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import (
    cholesky_basis,
    nlpd_gp,
    nlpd_pro,
    prediction_basis,
    predictive_moments,
    project_particles,
    run_inference_algorithm_with_burn_in,
    train_val_split,
)

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


class FitResult(NamedTuple):
    mean: jnp.ndarray
    std: jnp.ndarray
    nlpd_per_point: jnp.ndarray
    particle_predictions: jnp.ndarray | None = None
    sigma_eff: jnp.ndarray | None = None


def fit_gp(data, key=None, *, kernel_lengthscale=1.0, kernel_type="rbf") -> FitResult:

    x_train, y_train = data.x_train, data.y_train
    x_test, y_test = data.x_test, data.y_test
    gp_data = gpx.Dataset(X=x_train, y=y_train)

    kernel = build_kernel(kernel_type, lengthscale=kernel_lengthscale)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=x_train.shape[0])
    posterior = likelihood * prior

    opt_posterior, _ = gpx.fit_scipy(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=gp_data,
        verbose=False,
    )

    latent = opt_posterior.predict(x_test, train_data=gp_data)
    predictive = opt_posterior.likelihood(latent)
    mean = predictive.mean
    std = jnp.sqrt(predictive.variance)

    nlpd_per_point = nlpd_gp(y_test, mean, std, return_per_point=True)
    return FitResult(mean=mean, std=std, nlpd_per_point=nlpd_per_point)


def _adaptation_algorithm(algorithm: str):
    if algorithm == "ula":
        return ula
    if algorithm == "replica_gibbs":
        return replica_gibbs
    msg = f"Unknown algorithm={algorithm!r}; expected 'ula' or 'replica_gibbs'."
    raise ValueError(msg)


def _sampling_algorithm(algorithm: str, pro_params: ProParameters):
    if algorithm == "ula":
        return parametric_ula(pro_logdensity_fn, pro_params)
    if algorithm == "replica_gibbs":
        return parametric_replica_gibbs(pro_logdensity_fn, pro_params)
    msg = f"Unknown algorithm={algorithm!r}; expected 'ula' or 'replica_gibbs'."
    raise ValueError(msg)


def fit_pro(
    data,
    key,
    *,
    kernel_lengthscale=1.0,
    kernel_lengthscale_min=1.0e-3,
    kernel_lengthscale_max=1.0e3,
    kernel_type="rbf",
    algorithm="replica_gibbs",
    step_size=0.0001,
    alpha=1.0,
    sigma_init=0.3,
    sigma_min=0.05,
    sigma_max=1.0,
    num_particles=32,
    val_fraction=0.25,
    num_adapt_steps=200,
    warmup_steps=5,
    sigma_adapt_steps=90,
    kernel_adapt_steps=90,
    sigma_lr=0.1,
    kernel_lr=0.05,
    kernel_steps_per_adapt=1,
    num_sample_steps=200,
    burn_fraction=0.75,
    thin=5,
):
    if algorithm == "replica_gibbs":
        validate_r(num_particles, alpha)

    x_train, y_train = data.x_train, data.y_train
    x_test, y_test = data.x_test, data.y_test

    kernel = build_kernel(
        kernel_type,
        lengthscale=gpx.parameters.SigmoidBounded(
            kernel_lengthscale, low=kernel_lengthscale_min, high=kernel_lengthscale_max
        ),
    )
    basis_full = cholesky_basis(kernel, x_train)
    basis_dim = basis_full.shape[1]
    row_selectable_basis = PointInducingBasis(z=x_train)

    key, split_key, pos_key, adapt_key = jr.split(key, 4)
    split = train_val_split(split_key, x_train, y_train, val_fraction=val_fraction)

    fold_params = ProParameters(
        y=split.y_train,
        basis=None,
        step_size=step_size,
        sigma=gpx.parameters.SigmoidBounded(sigma_init, low=sigma_min, high=sigma_max),
        alpha=alpha,
        residual_std=None,
    )
    initial_position = jr.normal(pos_key, (basis_dim, num_particles))

    adaptation = parameter_adaptation(
        _adaptation_algorithm(algorithm),
        pro_logdensity_fn,
        fold_params,
        x_train=split.x_train,
        initial_kernel=kernel,
        warmup_steps=warmup_steps,
        sigma_adapt_steps=sigma_adapt_steps,
        kernel_adapt_steps=kernel_adapt_steps,
        objective_fn=pro_logdensity_fn,
        inducing_basis=row_selectable_basis,
        x_val=split.x_val,
        y_val=split.y_val,
        adapt_target=None,
        sigma_optimizer=ox.adam(sigma_lr),
        kernel_optimizer=ox.adam(kernel_lr),
        kernel_steps_per_adapt=kernel_steps_per_adapt,
        progress_bar=False,
    )
    adaptation_results, adaptation_info = adaptation.run(
        adapt_key, initial_position, num_steps=num_adapt_steps
    )

    if kernel_adapt_steps > 0:
        adapted_kernel = px.unwrap(
            jax.tree.map(lambda x: x[-1], adaptation_info.kernel)
        )
        basis_full = cholesky_basis(adapted_kernel, x_train)
    else:
        adapted_kernel = kernel

    pro_params = ProParameters(
        y=y_train,
        basis=basis_full,
        step_size=step_size,
        sigma=adaptation_results.parameters.sigma,
        alpha=alpha,
        residual_std=None,
    )
    sampling_algorithm = _sampling_algorithm(algorithm, pro_params)
    key, sample_key = jr.split(key)
    _, (states, _) = run_inference_algorithm_with_burn_in(
        rng_key=sample_key,
        inference_algorithm=sampling_algorithm,
        num_steps=num_sample_steps,
        burn_ratio=burn_fraction,
        initial_position=adaptation_results.state.position,
        progress_bar=False,
    )

    particles = states.position[::thin]
    test_basis, test_cov = prediction_basis(adapted_kernel, x_train, x_test, pro_params)
    nlpd_per_point = nlpd_pro(
        y_test,
        test_basis,
        test_cov,
        particles,
        parameters=pro_params,
        return_per_point=True,
    )

    sigma_val = px.unwrap(pro_params.sigma)
    residual_std = jnp.sqrt(
        jnp.maximum(jnp.diag(test_cov) - jnp.sum(test_basis**2, axis=1), 0.0)
    )
    mean, std = predictive_moments(
        test_basis, particles, noise_std=sigma_val, residual_std=residual_std
    )

    particle_predictions = project_particles(test_basis, particles)
    sigma_eff = jnp.sqrt(sigma_val**2 + residual_std**2)

    return FitResult(
        mean=mean,
        std=std,
        nlpd_per_point=nlpd_per_point,
        particle_predictions=particle_predictions,
        sigma_eff=sigma_eff,
    )


def evaluate(get_instance, fit_function, key, num_instances):
    keys = jr.split(key, num_instances)

    nlpds = []
    for instance_key in progress_bar(keys):
        data = get_instance(instance_key)
        fit_key, instance_key = jr.split(instance_key)
        nlpd_per_point = fit_function(data, fit_key).nlpd_per_point
        nlpds.append(float(jnp.mean(nlpd_per_point)))

    mean = float(np.mean(nlpds))
    std = float(np.std(nlpds))

    log.info("NLPD: %.4f±%.4f", mean, std)
    return mean, std, nlpds


_DATASET_SOURCES = {
    "block_outliers": (make_block_outlier_instance, BLOCK_OUTLIERS_KWARGS),
    "heteroskedastic": (make_heteroskedastic_instance, HETEROSKEDASTIC_KWARGS),
    "multimodal": (make_multimodal_instance, MULTIMODAL_KWARGS),
    "well_specified": (make_well_specified_instance, WELL_SPECIFIED_KWARGS),
}


def _instance_fn(make_instance, kwarg_names, cfg: DictConfig):
    """
    Build a `key -> data` closure for `make_instance`, forwarding only the subset of
    `kwarg_names` that `cfg` actually sets -- so a `ds` config can override any subset
    of a generator's parameters without a code change here.
    """
    kwargs = {k: cfg[k] for k in kwarg_names if k in cfg}
    return lambda key: make_instance(key, **kwargs)


def _fit_gp_fn(cfg: DictConfig):
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
        algorithm=cfg.pro.algorithm,
        step_size=cfg.pro.step_size,
        alpha=cfg.pro.alpha,
        sigma_init=cfg.pro.sigma_init,
        sigma_min=cfg.pro.sigma_min,
        sigma_max=cfg.pro.sigma_max,
        num_particles=cfg.pro.num_particles,
        val_fraction=cfg.pro.val_fraction,
        num_adapt_steps=cfg.pro.num_adapt_steps,
        warmup_steps=cfg.pro.warmup_steps,
        sigma_adapt_steps=cfg.pro.sigma_adapt_steps,
        kernel_adapt_steps=cfg.pro.kernel_adapt_steps,
        sigma_lr=cfg.pro.sigma_lr,
        kernel_lr=cfg.pro.kernel_lr,
        kernel_steps_per_adapt=cfg.pro.kernel_steps_per_adapt,
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


def out_dir(cfg: DictConfig, param_value) -> Path:
    return (
        Path(cfg.results_root)
        / cfg.source
        / f"{cfg.param_name}_{param_value}"
        / cfg.algorithm
    )


@hydra.main(version_base=None, config_path="../conf", config_name="synthetic")
def main(cfg: DictConfig) -> None:
    get_instance = _get_instance_fn(cfg)
    fit_algorithm = _get_fit_algorithm(cfg)
    key = jr.PRNGKey(cfg.seed)
    param_value = cfg[cfg.param_name]

    log.info(
        "Evaluating %s on %s (%s=%s)",
        cfg.algorithm,
        cfg.source,
        cfg.param_name,
        param_value,
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
