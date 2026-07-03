"""
Adaptation of `sigma` and the kernel hyperparameters behind `basis`.

Runs a fixed warmup with no parameter adaptation, then alternates between a
gradient step on `sigma` and a gradient step on a gpjax kernel's hyperparameters
that determine `basis`, on a user-defined schedule.
"""

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import gpjax as gpx
import jax
import jax.numpy as jnp
import optax
import paramax
from blackjax.adaptation.base import AdaptationResults
from blackjax.base import AdaptationAlgorithm
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import ArrayLikeTree, PRNGKey

from jax.scipy.linalg import solve_triangular

from non_parametric_pro.density import ProParameters
from non_parametric_pro.inducing import InducingBasis, compute_inducing_basis

# Schedule stage labels, dispatched via jax.lax.switch.
_WARMUP, _ADAPT_NU, _ADAPT_BASIS, _ADAPT_BOTH = 0, 1, 2, 3


class ParameterAdaptationState(NamedTuple):
    """
    Carry for the sigma/basis adaptation loop.

    `sigma` is expected to be a `paramax.AbstractUnwrappable` (e.g.
    `gpjax.parameters.NonNegativeReal`), optimised in its unconstrained
    space; `sigma_opt_state` carries optax state for that unconstrained
    representation.
    """

    sigma: paramax.AbstractUnwrappable
    sigma_opt_state: optax.OptState
    kernel: gpx.kernels.AbstractKernel
    kernel_opt_state: optax.OptState
    basis: jax.Array
    residual_std: jax.Array | None = None  # None for full GP, (N,1) for inducing
    val_basis: jax.Array | None = None          # prediction basis at x_val
    val_residual_std: jax.Array | None = None   # approximation residual at x_val


class ParameterAdaptationInfo(NamedTuple):
    """
    Per-step diagnostics, stacked by `lax.scan` over the whole run.

    `sampler_info` is whatever `algorithm`'s kernel returns as transition
    info (e.g. `ULAInfo`). `sigma`/`kernel` are the adaptation state *after*
    that step, stacked with a leading `num_steps` axis on every leaf -- both
    are `paramax`-wrapped (`sigma` as a `NonNegativeReal`-like type, `kernel`
    as the gpjax kernel), so read them with `paramax.unwrap(...)` (and, for
    `kernel`, `jax.tree.map(lambda x: x[i], ...)` first to pick a step).
    """

    sampler_info: ArrayLikeTree
    sigma: paramax.AbstractUnwrappable
    kernel: gpx.kernels.AbstractKernel


def merge_parameters(
    base: NamedTuple,
    new: NamedTuple,
    transforms: dict[str, Callable] | None = None,
) -> NamedTuple:
    """Override `base`'s fields with `new`'s, for fields present on both."""
    transforms = transforms or {}
    updates = {
        f: transforms.get(f, lambda v: v)(v)
        for f, v in new._asdict().items()
        if f in base._fields
    }
    return base._replace(**updates)


def build_schedule(
    num_steps: int,
    *,
    warmup_steps: int,
    sigma_adapt_every: int,
    kernel_adapt_every: int,
) -> jax.Array:
    """
    Stage 0 (no-op) during warmup; afterwards adapt sigma/kernel independently.

    After `warmup_steps`, `sigma` is adapted every `sigma_adapt_every` steps and
    the kernel every `kernel_adapt_every` steps -- independently, not
    alternating, so they can have entirely different cadences. Pass -1 for
    either to disable that parameter's adaptation entirely. If both would
    fire on the same step, both run that step (see `_ADAPT_BOTH`/`both_update`).
    """
    idx = jnp.arange(num_steps)
    steps_since_warmup = idx - warmup_steps
    post_warmup = idx >= warmup_steps

    is_nu_step = (
        post_warmup
        & (sigma_adapt_every != -1)
        & (steps_since_warmup % sigma_adapt_every == 0)
    )
    is_kernel_step = (
        post_warmup
        & (kernel_adapt_every != -1)
        & (steps_since_warmup % kernel_adapt_every == 0)
    )

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
) -> tuple[Callable, Callable, Callable]:
    """
    Build the (init, update, final) triple for sigma/basis adaptation.

    Parameters
    ----------
    x_train
        Training inputs used to recompute `basis` whenever the kernel
        hyperparameters change.
    jitter
        Diagonal jitter added before the training Gram matrix's Cholesky.
    sigma_optimizer, kernel_optimizer
        Optax optimisers for the two adapted quantities.
    inducing_basis
        An :class:`InducingBasis` instance (e.g. ``PointInducingBasis(z)``).
        If ``None`` (default), a full GP Cholesky basis is used and
        ``residual_std`` is ``None``. If given, the basis is
        ``K_xz L_zz^{-T}`` and ``residual_std`` is
        ``sqrt(diag(K_xx) - diag(basis @ basis^T))``.

    """

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
    ) -> tuple[jax.Array | None, jax.Array | None]:
        """Prediction basis at validation points; (None, None) if no x_val."""
        if x_val is None:
            return None, None
        unwrapped = paramax.unwrap(kernel)
        x_v = x_val.reshape(-1, 1) if x_val.ndim == 1 else x_val
        if inducing_basis is None:
            x_tr = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
            k_train = unwrapped.gram(x_tr).as_matrix()
            L = jnp.linalg.cholesky(k_train + jitter * jnp.eye(k_train.shape[0]))
            K_val_train = unwrapped.cross_covariance(x_v, x_tr)
            return solve_triangular(L, K_val_train.T, lower=True).T, None
        return compute_inducing_basis(inducing_basis, unwrapped, x_v, jitter)

    def _objective_parameters(
        base_parameters: ProParameters,
        vb: jax.Array | None,
        vr: jax.Array | None,
    ) -> ProParameters:
        """Swap to validation y/basis if available, else use training parameters."""
        if x_val is None:
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
        vb, vr = val_basis_fn(kernel)
        obj_params = _objective_parameters(
            base_parameters._replace(basis=new_basis, residual_std=new_residual_std),
            vb, vr,
        )
        return -objective_fn(position, obj_params)

    def init(
        initial_sigma: paramax.AbstractUnwrappable,
        initial_kernel: gpx.kernels.AbstractKernel,
    ) -> ParameterAdaptationState:
        initial_basis, initial_residual_std = basis_fn(initial_kernel)
        initial_val_basis, initial_val_residual_std = val_basis_fn(initial_kernel)
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
            state.sigma, position, base_parameters,
            state.val_basis, state.val_residual_std,
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
        # Mirrors gpjax's own gpx.fit step: filter to the array (trainable)
        # leaves, differentiate the paramax-wrapped model, apply updates back
        # onto the full pytree (static fields like compute_engine untouched).
        _, grad = eqx.filter_value_and_grad(kernel_loss)(
            state.kernel, position, base_parameters
        )
        updates, new_opt_state = kernel_optimizer.update(
            grad, state.kernel_opt_state, eqx.filter(state.kernel, eqx.is_array)
        )
        new_kernel = eqx.apply_updates(state.kernel, updates)
        new_basis, new_residual_std = basis_fn(new_kernel)
        new_val_basis, new_val_residual_std = val_basis_fn(new_kernel)
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
        # Each loss receives the current merged parameters (including the other's
        # current value), so sigma_update sees the current adapted basis and
        # basis_update sees the current adapted sigma.
        state = sigma_update(state, position, base_parameters)
        return basis_update(state, position, base_parameters)

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

# ToDo: better handling of case where ProParameters does not have basis field precomputed
def parameter_adaptation(  # noqa: PLR0913
    algorithm,
    logdensity_fn: Callable,
    base_parameters: ProParameters,
    *,
    x_train: jax.Array,
    initial_kernel: gpx.kernels.AbstractKernel,
    warmup_steps: int,
    sigma_adapt_every: int,
    kernel_adapt_every: int,
    objective_fn: Callable,
    inducing_basis: InducingBasis | None = None,
    x_val: jax.Array | None = None,
    y_val: jax.Array | None = None,
    sigma_optimizer: optax.GradientTransformation | None = None,
    kernel_optimizer: optax.GradientTransformation | None = None,
    jitter: float = 1e-6,
    progress_bar: bool = False,
) -> AdaptationAlgorithm:
    """
    Adapt `sigma` and the kernel hyperparameters behind `basis`.

    `algorithm` is a module exposing `build_kernel(logdensity_fn)` and
    `init(position, parameters, logdensity_fn)` (e.g. `non_parametric_pro.ula`),
    matching how `blackjax.adaptation.window_adaptation` consumes `blackjax.mala`.

    `sigma_adapt_every`/`kernel_adapt_every` control each parameter's adaptation
    cadence independently (see `build_schedule`); pass -1 to disable a given
    parameter's adaptation entirely.

    The returned `AdaptationAlgorithm.run` yields
    `(AdaptationResults(state, parameters), info)` -- `parameters` is a
    `ProParameters` with `sigma`/`basis` set from the final step, ready to feed
    straight into a sampler. `info` is a `ParameterAdaptationInfo` stacked
    over every step (via `lax.scan`), so `info.sigma`/`info.kernel` are full
    `num_steps`-length traces -- `info.kernel` is not part of `ProParameters`
    since the density never reads it directly, but is needed for downstream
    uses like predicting at new inputs; index `jax.tree.map(lambda x: x[-1],
    info.kernel)` to recover the final optimised kernel on its own.
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
    )

    def one_step(carry, xs):
        _, rng_key, stage = xs
        state, adaptation_state = carry

        parameters = merge_parameters(base_parameters, adaptation_state)
        new_state, sampler_info = kernel(rng_key, state, parameters)

        new_adaptation_state = adapt_step(
            adaptation_state, stage, new_state.position, parameters
        )

        # If this step changed sigma/basis, new_state's cached logdensity/grad
        # (computed above under the *old* parameters) are stale relative to
        # new_adaptation_state -- they'd otherwise drive the Langevin drift
        # on the *next* kernel call under the wrong parameters. Refresh only
        # fires on actual adaptation steps, not every step.
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
            print("Running parameter adaptation")  # noqa: T201
        schedule = build_schedule(
            num_steps,
            warmup_steps=warmup_steps,
            sigma_adapt_every=sigma_adapt_every,
            kernel_adapt_every=kernel_adapt_every,
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
