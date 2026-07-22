"""Stochastic Gradient Langevin Dynamics (SGLD) sampler for the PRO posterior."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import diffusions
from blackjax.types import PRNGKey

from non_parametric_pro.density import pro_score_fn
from non_parametric_pro.ula import ULAInfo, ULAState, init, refresh

__all__ = ["SGLDAlgorithm", "build_kernel", "init", "parametric_sgld", "refresh", "sgld"]


def build_kernel(logdensity_fn: Callable, *, batch_size: int) -> Callable:
    """
    Build an SGLD kernel: the same overdamped Langevin step as
    `ula.build_kernel`, except a fresh random minibatch of `batch_size` rows
    (`parameters.batch_idx`) is resampled every step before evaluating the
    density/gradient. See `density.pro_score_fn` for the `n / batch_size`
    rescale that keeps this an unbiased estimator of the full-data score.

    A fixed `step_size` tuned for exact ULA will generally need to shrink for
    SGLD -- minibatch gradient noise adds to (does not replace) ULA's own
    discretisation noise, and an unadjusted Euler-Maruyama step has no
    built-in safeguard against that extra variance.
    """
    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
    one_step = diffusions.overdamped_langevin(logdensity_and_grad_fn)

    def kernel(
        rng_key: PRNGKey,
        state: ULAState,
        parameters: NamedTuple,
    ) -> tuple[ULAState, ULAInfo]:
        """Generate a new sample with the SGLD kernel."""
        batch_key, step_key = jax.random.split(rng_key)
        n = parameters.basis.shape[0]  # type: ignore  # noqa: PGH003
        batch_idx = jax.random.choice(batch_key, n, (batch_size,), replace=False)
        parameters = parameters._replace(batch_idx=batch_idx)  # type: ignore  # noqa: PGH003

        new_state = one_step(
            step_key,
            state,                          # type: ignore  # noqa: PGH003
            parameters.step_size,
            (parameters,),
        )
        new_state = ULAState(*new_state)    # type: ignore  # noqa: PGH003

        score = pro_score_fn(new_state.position, parameters)
        return new_state, ULAInfo(score)

    return kernel


@dataclass(frozen=True)
class SGLDAlgorithm:
    """
    Adapter satisfying the `algorithm` duck-type expected by
    `non_parametric_pro.adaptation.parameter_adaptation` (and
    `parametric_sgld` below) -- identical to passing the `ula` module
    itself, except `build_kernel` is pre-bound to this instance's
    `batch_size`. Construct via :func:`sgld`, not directly.
    """

    batch_size: int

    def build_kernel(self, logdensity_fn: Callable) -> Callable:
        return build_kernel(logdensity_fn, batch_size=self.batch_size)

    def init(
        self, position: jax.Array, parameters: NamedTuple, logdensity_fn: Callable
    ) -> ULAState:
        return init(position, parameters, logdensity_fn)

    def refresh(
        self, state: ULAState, parameters: NamedTuple, logdensity_fn: Callable
    ) -> ULAState:
        return refresh(state, parameters, logdensity_fn)


def sgld(batch_size: int) -> SGLDAlgorithm:
    """
    Construct an `algorithm`-compatible SGLD sampler with a fixed minibatch size.

    Drop-in replacement for the `ula` module wherever an `algorithm` argument
    is expected, e.g.:

    .. code::

        from non_parametric_pro.sgld import sgld
        from non_parametric_pro.adaptation.parameter_adaptation import (
            cross_validated_parameter_adaptation,
        )

        result = cross_validated_parameter_adaptation(
            sgld(batch_size=256),   # instead of `ula`
            pro_logdensity_fn,
            pro_params,
            ...,
        )
    """
    return SGLDAlgorithm(batch_size=batch_size)


def parametric_sgld(
    logdensity_fn: Callable,
    parameters: NamedTuple,
    *,
    batch_size: int,
) -> SamplingAlgorithm:
    """
    Standalone user interface for the SGLD kernel, mirroring
    `ula.parametric_ula`.

    Examples
    --------
    .. code::

        sampler = parametric_sgld(pro_logdensity_fn, parameters, batch_size=256)
        state = sampler.init(position)
        new_state, info = sampler.step(rng_key, state)
    """
    kernel = build_kernel(logdensity_fn, batch_size=batch_size)

    def init_fn(position: jax.Array, rng_key: PRNGKey | None = None) -> ULAState:
        del rng_key
        return init(position, parameters, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state: ULAState) -> tuple[ULAState, ULAInfo]:
        return kernel(rng_key, state, parameters)

    return SamplingAlgorithm(init_fn, step_fn)
