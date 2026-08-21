from collections.abc import Callable
from typing import Literal, NamedTuple

import equinox as eqx
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import paramax
from blackjax.adaptation.base import AdaptationResults
from blackjax.base import AdaptationAlgorithm
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import ArrayLikeTree, PRNGKey
from jax.scipy.linalg import solve_triangular

from non_parametric_pro.density import ProParameters
from non_parametric_pro.inducing import InducingBasis, compute_inducing_basis
from non_parametric_pro.util import TrainValSplit, train_val_split

_WARMUP, _ADAPT_NU, _ADAPT_BASIS, _ADAPT_BOTH = 0, 1, 2, 3


class ParameterAdaptationState(NamedTuple):
    sigma: paramax.AbstractUnwrappable
    sigma_opt_state: optax.OptState
    kernel: gpx.kernels.AbstractKernel
    kernel_opt_state: optax.OptState
    basis: jax.Array
    residual_std: jax.Array | None = None
    val_basis: jax.Array | None = None
    val_residual_std: jax.Array | None = None


class ParameterAdaptationInfo(NamedTuple):
    sampler_info: ArrayLikeTree
    sigma: paramax.AbstractUnwrappable
    kernel: gpx.kernels.AbstractKernel


def merge_parameters(
    base: NamedTuple,
    new: NamedTuple,
    transforms: dict[str, Callable] | None = None,
) -> NamedTuple:
    transforms = transforms or {}
    updates = {
        f: transforms.get(f, lambda v: v)(v)
        for f, v in new._asdict().items()
        if f in base._fields
    }
    return base._replace(**updates)


def _evenly_spaced_steps(warmup_steps: int, num_steps: int, count: int) -> jax.Array:
    if count <= 0 or warmup_steps >= num_steps:
        return jnp.array([], dtype=jnp.int32)
    positions = jnp.linspace(warmup_steps, num_steps - 1, count)
    return jnp.round(positions).astype(jnp.int32)


def build_schedule(
    num_steps: int,
    *,
    warmup_steps: int,
    sigma_adapt_steps: int,
    kernel_adapt_steps: int,
) -> jax.Array:
    idx = jnp.arange(num_steps)
    sigma_steps = _evenly_spaced_steps(warmup_steps, num_steps, sigma_adapt_steps)
    kernel_steps = _evenly_spaced_steps(warmup_steps, num_steps, kernel_adapt_steps)

    is_nu_step = jnp.isin(idx, sigma_steps)
    is_kernel_step = jnp.isin(idx, kernel_steps)

    return jnp.where(
        is_nu_step & is_kernel_step,
        _ADAPT_BOTH,
        jnp.where(
            is_nu_step, _ADAPT_NU, jnp.where(is_kernel_step, _ADAPT_BASIS, _WARMUP)
        ),
    )


def base(
    *,
    x_train: jax.Array,
    jitter: float,
    sigma_optimizer: optax.GradientTransformation,
    kernel_optimizer: optax.GradientTransformation,
    objective_fn: Callable,
    inducing_basis: InducingBasis | None = None,
    x_val: jax.Array | None = None,
    y_val: jax.Array | None = None,
    adapt_target: Literal["train", "val"] | None = None,
) -> tuple[Callable, Callable, Callable]:
    """
    Build the (init, update, final) triple for sigma/basis adaptation.
    """
    if adapt_target == "val" and (x_val is None or y_val is None):
        msg = "adapt_target='val' requires both x_val and y_val to be given."
        raise ValueError(msg)

    def basis_fn(
        kernel: gpx.kernels.AbstractKernel,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Recompute basis (and residual_std for inducing) from kernel."""
        unwrapped = paramax.unwrap(kernel)
        x = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
        if inducing_basis is None:
            k_train = unwrapped.gram(x).as_matrix()
            basis = jnp.linalg.cholesky(k_train + jitter * jnp.eye(k_train.shape[0]))
            return basis, None
        return compute_inducing_basis(inducing_basis, unwrapped, x, jitter)

    def val_basis_fn(
        kernel: gpx.kernels.AbstractKernel,
        train_basis: jax.Array,
    ) -> tuple[jax.Array | None, jax.Array | None]:
        """
        Prediction basis at validation points; (None, None) if no x_val.
        """
        if x_val is None:
            return None, None
        unwrapped = paramax.unwrap(kernel)
        x_v = x_val.reshape(-1, 1) if x_val.ndim == 1 else x_val
        if inducing_basis is None:
            x_tr = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
            L = train_basis
            K_val_train = unwrapped.cross_covariance(x_v, x_tr)
            return solve_triangular(L, K_val_train.T, lower=True).T, None
        return compute_inducing_basis(inducing_basis, unwrapped, x_v, jitter)

    def _objective_parameters(
        base_parameters: ProParameters,
        vb: jax.Array | None,
        vr: jax.Array | None,
    ) -> ProParameters:
        """
        Swap to validation y/basis if `adapt_target` calls for it, else training.

        With `adapt_target=None`, reproduces the original `sigma`-only default:
        validation if `x_val` was given, training otherwise.
        """
        use_val = adapt_target == "val" or (adapt_target is None and x_val is not None)
        if not use_val:
            return base_parameters
        return base_parameters._replace(y=y_val, basis=vb, residual_std=vr)

    def sigma_loss(
        sigma: paramax.AbstractUnwrappable,
        position: ArrayLikeTree,
        base_parameters: ProParameters,
        val_basis: jax.Array | None,
        val_residual_std: jax.Array | None,
    ) -> jax.Array:
        obj_params = _objective_parameters(base_parameters, val_basis, val_residual_std)
        return -objective_fn(position, obj_params._replace(sigma=sigma))

    def kernel_loss(
        kernel: gpx.kernels.AbstractKernel,
        position: ArrayLikeTree,
        base_parameters: ProParameters,
    ) -> jax.Array:
        new_basis, new_residual_std = basis_fn(kernel)
        if adapt_target == "val":
            val_basis, val_residual_std = val_basis_fn(kernel, new_basis)
            obj_params = base_parameters._replace(
                y=y_val, basis=val_basis, residual_std=val_residual_std
            )
        else:
            obj_params = base_parameters._replace(
                basis=new_basis, residual_std=new_residual_std
            )
        return -objective_fn(position, obj_params)

    def init(
        initial_sigma: paramax.AbstractUnwrappable,
        initial_kernel: gpx.kernels.AbstractKernel,
    ) -> ParameterAdaptationState:
        initial_basis, initial_residual_std = basis_fn(initial_kernel)
        initial_val_basis, initial_val_residual_std = val_basis_fn(
            initial_kernel, initial_basis
        )
        return ParameterAdaptationState(
            sigma=initial_sigma,
            sigma_opt_state=sigma_optimizer.init(
                eqx.filter(initial_sigma, eqx.is_array)
            ),
            kernel=initial_kernel,
            kernel_opt_state=kernel_optimizer.init(
                eqx.filter(initial_kernel, eqx.is_array)
            ),
            basis=initial_basis,
            residual_std=initial_residual_std,
            val_basis=initial_val_basis,
            val_residual_std=initial_val_residual_std,
        )

    def no_op(
        state: ParameterAdaptationState,
        position: ArrayLikeTree,
        base_parameters: ProParameters,
    ) -> ParameterAdaptationState:
        del position, base_parameters
        return state

    def sigma_update(
        state: ParameterAdaptationState,
        position: ArrayLikeTree,
        base_parameters: ProParameters,
    ) -> ParameterAdaptationState:
        _, grad = eqx.filter_value_and_grad(sigma_loss)(
            state.sigma,
            position,
            base_parameters,
            state.val_basis,
            state.val_residual_std,
        )
        updates, new_opt_state = sigma_optimizer.update(
            grad, state.sigma_opt_state, eqx.filter(state.sigma, eqx.is_array)
        )
        new_sigma = eqx.apply_updates(state.sigma, updates)
        return state._replace(sigma=new_sigma, sigma_opt_state=new_opt_state)

    def basis_update(
        state: ParameterAdaptationState,
        position: ArrayLikeTree,
        base_parameters: ProParameters,
    ) -> ParameterAdaptationState:
        _, grad = eqx.filter_value_and_grad(kernel_loss)(
            state.kernel, position, base_parameters
        )
        updates, new_opt_state = kernel_optimizer.update(
            grad, state.kernel_opt_state, eqx.filter(state.kernel, eqx.is_array)
        )
        new_kernel = eqx.apply_updates(state.kernel, updates)
        new_basis, new_residual_std = basis_fn(new_kernel)
        new_val_basis, new_val_residual_std = val_basis_fn(new_kernel, new_basis)
        return state._replace(
            kernel=new_kernel,
            kernel_opt_state=new_opt_state,
            basis=new_basis,
            residual_std=new_residual_std,
            val_basis=new_val_basis,
            val_residual_std=new_val_residual_std,
        )

    def both_update(
        state: ParameterAdaptationState,
        position: ArrayLikeTree,
        base_parameters: ProParameters,
    ) -> ParameterAdaptationState:
        state = sigma_update(state, position, base_parameters)
        updated_parameters = merge_parameters(base_parameters, state)
        return basis_update(state, position, updated_parameters)

    def update(
        state: ParameterAdaptationState,
        stage: jax.Array,
        position: ArrayLikeTree,
        base_parameters: ProParameters,
    ) -> ParameterAdaptationState:
        return jax.lax.switch(
            stage,
            (no_op, sigma_update, basis_update, both_update),
            state,
            position,
            base_parameters,
        )

    def final(
        state: ParameterAdaptationState,
    ) -> tuple[jax.Array, jax.Array, gpx.kernels.AbstractKernel]:
        return state.sigma, state.basis, state.kernel

    return init, update, final


def parameter_adaptation(
    algorithm,
    logdensity_fn: Callable,
    base_parameters: ProParameters,
    *,
    x_train: jax.Array,
    initial_kernel: gpx.kernels.AbstractKernel,
    warmup_steps: int,
    sigma_adapt_steps: int,
    kernel_adapt_steps: int,
    objective_fn: Callable,
    inducing_basis: InducingBasis | None = None,
    x_val: jax.Array | None = None,
    y_val: jax.Array | None = None,
    adapt_target: Literal["train", "val"] | None = None,
    sigma_optimizer: optax.GradientTransformation | None = None,
    kernel_optimizer: optax.GradientTransformation | None = None,
    jitter: float = 1e-6,
    progress_bar: bool = False,
) -> AdaptationAlgorithm:
    """
    Adapt `sigma` and the kernel hyperparameters behind `basis`.
    """
    if sigma_optimizer is None:
        sigma_optimizer = optax.adam(1e-2)
    if kernel_optimizer is None:
        kernel_optimizer = optax.adam(1e-2)

    kernel = algorithm.build_kernel(logdensity_fn)
    adapt_init, adapt_step, _ = base(
        x_train=x_train,
        jitter=jitter,
        sigma_optimizer=sigma_optimizer,
        kernel_optimizer=kernel_optimizer,
        objective_fn=objective_fn,
        inducing_basis=inducing_basis,
        x_val=x_val,
        y_val=y_val,
        adapt_target=adapt_target,
    )

    def one_step(carry, xs):
        _, rng_key, stage = xs
        state, adaptation_state = carry

        parameters = merge_parameters(base_parameters, adaptation_state)
        new_state, sampler_info = kernel(rng_key, state, parameters)

        new_adaptation_state = adapt_step(
            adaptation_state, stage, new_state.position, parameters
        )

        new_parameters = merge_parameters(base_parameters, new_adaptation_state)
        new_state = jax.lax.cond(
            stage == _WARMUP,
            lambda s: s,
            lambda s: algorithm.refresh(s, new_parameters, logdensity_fn),
            new_state,
        )

        info = ParameterAdaptationInfo(
            sampler_info=sampler_info,
            sigma=new_adaptation_state.sigma,
            kernel=new_adaptation_state.kernel,
        )

        return (new_state, new_adaptation_state), info

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):
        init_adaptation_state = adapt_init(base_parameters.sigma, initial_kernel)
        init_parameters = merge_parameters(base_parameters, init_adaptation_state)
        init_state = algorithm.init(position, init_parameters, logdensity_fn)

        if progress_bar:
            print("Running parameter adaptation")
        schedule = build_schedule(
            num_steps,
            warmup_steps=warmup_steps,
            sigma_adapt_steps=sigma_adapt_steps,
            kernel_adapt_steps=kernel_adapt_steps,
        )
        keys = jax.random.split(rng_key, num_steps)

        scan_fn = gen_scan_fn(num_steps, progress_bar=progress_bar)
        (last_state, last_adaptation_state), info = scan_fn(
            one_step,
            (init_state, init_adaptation_state),
            (jnp.arange(num_steps), keys, schedule),
        )

        final_parameters = merge_parameters(base_parameters, last_adaptation_state)
        return AdaptationResults(last_state, final_parameters), info

    return AdaptationAlgorithm(run)


class CrossValidationResult(NamedTuple):
    """
    Cross-validated sigma/kernel hyperparameters from :func:`cross_validated_parameter_adaptation`.
    """

    sigma: jax.Array
    kernel: gpx.kernels.AbstractKernel
    fold_sigma: jax.Array
    fold_kernel: gpx.kernels.AbstractKernel


def _kfold_splits(
    key: PRNGKey,
    x_full: jax.Array,
    y_full: jax.Array,
    *,
    num_folds: int,
    val_fraction: float,
):
    """
    Disjoint train/validation partitions for classic k-fold cross-validation.
    """
    if num_folds == 1:
        return [train_val_split(key, x_full, y_full, val_fraction=val_fraction)]

    n = y_full.shape[0]
    idx = jr.permutation(key, n)
    chunks = jnp.array_split(idx, num_folds)

    splits = []
    for i in range(num_folds):
        val_idx = chunks[i]
        train_idx = jnp.concatenate([chunks[j] for j in range(num_folds) if j != i])
        splits.append(
            TrainValSplit(
                x_train=x_full[train_idx],
                y_train=y_full[train_idx],
                x_val=x_full[val_idx],
                y_val=y_full[val_idx],
                train_idx=train_idx,
                val_idx=val_idx,
            )
        )
    return splits


def cross_validated_parameter_adaptation(
    algorithm,
    logdensity_fn: Callable,
    base_parameters: ProParameters,
    *,
    x_full: jax.Array,
    y_full: jax.Array,
    initial_kernel: gpx.kernels.AbstractKernel,
    num_folds: int,
    val_fraction: float = 0.2,
    num_particles: int,
    num_steps: int,
    warmup_steps: int,
    sigma_adapt_steps: int,
    kernel_adapt_steps: int,
    objective_fn: Callable,
    rng_key: PRNGKey,
    inducing_basis: InducingBasis | None = None,
    adapt_target: Literal["train", "val"] | None = None,
    sigma_optimizer: optax.GradientTransformation | None = None,
    kernel_optimizer: optax.GradientTransformation | None = None,
    jitter: float = 1e-6,
    progress_bar: bool = False,
) -> CrossValidationResult:
    """
    Cross-validated wrapper around :func:`parameter_adaptation`.
    """
    split_key, run_key = jr.split(rng_key)
    splits = _kfold_splits(
        split_key, x_full, y_full, num_folds=num_folds, val_fraction=val_fraction
    )
    fold_keys = jr.split(run_key, len(splits))

    @jax.jit
    def run_one_fold(x_train, y_train, x_val, y_val, pos_key, adapt_key):
        fold_parameters = base_parameters._replace(y=y_train)
        basis_dim = (
            inducing_basis.output_dim()
            if inducing_basis is not None
            else x_train.shape[0]
        )
        initial_position = jr.normal(pos_key, (basis_dim, num_particles))

        adaptation = parameter_adaptation(
            algorithm,
            logdensity_fn,
            fold_parameters,
            x_train=x_train,
            initial_kernel=initial_kernel,
            warmup_steps=warmup_steps,
            sigma_adapt_steps=sigma_adapt_steps,
            kernel_adapt_steps=kernel_adapt_steps,
            objective_fn=objective_fn,
            inducing_basis=inducing_basis,
            x_val=x_val,
            y_val=y_val,
            adapt_target=adapt_target,
            sigma_optimizer=sigma_optimizer,
            kernel_optimizer=kernel_optimizer,
            jitter=jitter,
            progress_bar=progress_bar,
        )
        adaptation_results, adaptation_info = adaptation.run(
            adapt_key, initial_position, num_steps=num_steps
        )

        final_sigma = paramax.unwrap(adaptation_results.parameters.sigma)
        final_kernel = paramax.unwrap(
            jax.tree.map(lambda x: x[-1], adaptation_info.kernel)
        )
        return final_sigma, final_kernel

    fold_sigmas = []
    fold_kernels = []

    for split, fold_key in zip(splits, fold_keys, strict=True):
        pos_key, adapt_key = jr.split(fold_key)
        final_sigma, final_kernel = run_one_fold(
            split.x_train, split.y_train, split.x_val, split.y_val, pos_key, adapt_key
        )
        fold_sigmas.append(jnp.asarray(final_sigma))
        fold_kernels.append(final_kernel)

    fold_sigma = jnp.stack(fold_sigmas)
    fold_kernel = jax.tree.map(lambda *leaves: jnp.stack(leaves), *fold_kernels)

    mean_sigma = jnp.mean(fold_sigma, axis=0)
    mean_kernel = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), fold_kernel)

    return CrossValidationResult(
        sigma=mean_sigma,
        kernel=mean_kernel,
        fold_sigma=fold_sigma,
        fold_kernel=fold_kernel,
    )
