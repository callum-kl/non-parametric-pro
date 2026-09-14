"""Mirrors experiments/synthetic/scripts/synthetic.py:fit_pro, but also returns the
kernel-lengthscale trajectory so adaptation can be studied directly."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px

from non_parametric_pro import replica_gibbs
from non_parametric_pro.data.synthetic.well_specified import build_kernel
from non_parametric_pro.density import ProParameters, pro_logdensity_fn
from non_parametric_pro.inducing import PointInducingBasis
from non_parametric_pro.parameter_adaptation import parameter_adaptation
from non_parametric_pro.replica_gibbs import parametric_replica_gibbs, validate_r
from non_parametric_pro.util import (
    cholesky_basis,
    nlpd_pro,
    prediction_basis,
    run_inference_algorithm_with_burn_in,
    train_val_split,
)


def run(
    data,
    key,
    *,
    kernel_lengthscale=1.0,
    alpha=1.0,
    num_particles=32,
    num_adapt_steps=200,
    warmup_steps=20,
    sigma_adapt_steps=90,
    kernel_adapt_steps=90,
    sigma_lr=0.1,
    kernel_lr=0.05,
    kernel_steps_per_adapt=1,
    sigma_init=0.2,
    sigma_min=0.01,
    sigma_max=1.0,
    val_fraction=0.25,
    step_size=1e-4,
    num_sample_steps=50,
    burn_fraction=0.5,
    thin=2,
):
    validate_r(num_particles, alpha)
    x_train, y_train = data.x_train, data.y_train

    kernel = build_kernel(
        "rbf",
        lengthscale=gpx.parameters.SigmoidBounded(
            kernel_lengthscale, low=1e-3, high=1e3
        ),
    )
    basis_dim = cholesky_basis(kernel, x_train).shape[1]

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

    adaptation = parameter_adaptation(
        replica_gibbs,
        pro_logdensity_fn,
        fold_params,
        x_train=split.x_train,
        initial_kernel=kernel,
        warmup_steps=warmup_steps,
        sigma_adapt_steps=sigma_adapt_steps,
        kernel_adapt_steps=kernel_adapt_steps,
        objective_fn=pro_logdensity_fn,
        inducing_basis=PointInducingBasis(z=x_train),
        x_val=split.x_val,
        y_val=split.y_val,
        adapt_target=None,
        sigma_optimizer=ox.adam(sigma_lr),
        kernel_optimizer=ox.adam(kernel_lr),
        kernel_steps_per_adapt=kernel_steps_per_adapt,
        progress_bar=False,
    )
    results, info = adaptation.run(
        adapt_key, jr.normal(pos_key, (basis_dim, num_particles)), num_steps=num_adapt_steps
    )

    ell_traj = np.asarray(px.unwrap(info.kernel).lengthscale).reshape(-1)
    adapted = (
        px.unwrap(jax.tree.map(lambda x: x[-1], info.kernel))
        if kernel_adapt_steps > 0
        else kernel
    )

    pro_params = ProParameters(
        y=y_train,
        basis=cholesky_basis(adapted, x_train),
        step_size=step_size,
        sigma=results.parameters.sigma,
        alpha=alpha,
        residual_std=None,
    )
    key, sample_key = jr.split(key)
    _, (states, _) = run_inference_algorithm_with_burn_in(
        rng_key=sample_key,
        inference_algorithm=parametric_replica_gibbs(pro_logdensity_fn, pro_params),
        num_steps=num_sample_steps,
        burn_ratio=burn_fraction,
        initial_position=results.state.position,
        progress_bar=False,
    )

    particles = states.position[::thin]
    test_basis, test_cov = prediction_basis(adapted, x_train, data.x_test, pro_params)
    nlpd = float(
        jnp.mean(
            nlpd_pro(
                data.y_test,
                test_basis,
                test_cov,
                particles,
                parameters=pro_params,
                return_per_point=True,
            )
        )
    )
    return {
        "nlpd": nlpd,
        "ell": float(ell_traj[-1]),
        "ell_traj": ell_traj,
        "sigma": float(px.unwrap(pro_params.sigma)),
    }
