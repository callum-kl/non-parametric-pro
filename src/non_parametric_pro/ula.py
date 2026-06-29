"""BlackJAX-style ULA sampler specialised for the PRO posterior."""

from collections.abc import Callable
from typing import NamedTuple

import jax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import diffusions
from blackjax.types import PRNGKey

from non_parametric_pro.density import predictive_score


class ULAState(NamedTuple):
    """
    State of the ULA algorithm.

    The ULA algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density as well as the current gradient of
    the log-probability density.
    """

    position: jax.Array
    logdensity: jax.Array
    logdensity_grad: jax.Array


class ULAInfo(NamedTuple):
    """
    Additional information on the ULA transition.

    score
        The average negative predictive log score of the new particles.
    """

    score: jax.Array


def init(
    position: jax.Array,
    parameters: NamedTuple,
    logdensity_fn: Callable,
) -> ULAState:
    """Initialize a ULA state from particle positions."""
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(
        position,
        parameters,
    )
    return ULAState(position, logdensity, logdensity_grad)


def refresh(
    state: ULAState,
    parameters: NamedTuple,
    logdensity_fn: Callable,
) -> ULAState:
    """Refresh cached log density and gradient after parameters change."""
    return init(state.position, parameters, logdensity_fn)


def build_kernel(logdensity_fn: Callable) -> Callable:
    """
    Build a ULA kernel using BlackJAX's overdamped Langevin diffusion.

    ``parameters`` is passed into the kernel explicitly, rather than closed
    over, so that it can be a regular pytree argument that varies between
    calls (e.g. during adaptation) without forcing retracing of the kernel.
    """
    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
    one_step = diffusions.overdamped_langevin(logdensity_and_grad_fn)

    def kernel(
        rng_key: PRNGKey,
        state: ULAState,
        parameters: NamedTuple,
    ) -> tuple[ULAState, ULAInfo]:
        """Generate a new sample with the ULA kernel."""
        new_state = one_step(
            rng_key,
            state,                          # type: ignore  # noqa: PGH003
            parameters.step_size,
            (parameters,),
        )
        new_state = ULAState(*new_state)    # type: ignore  # noqa: PGH003

        score = predictive_score(new_state.position, parameters)
        return new_state, ULAInfo(score)

    return kernel


def parametric_ula(
    logdensity_fn: Callable,
    parameters: NamedTuple,
) -> SamplingAlgorithm:
    """
    Implement the (basic) user interface for the ULA kernel.

    Mirrors BlackJAX's ``mala``/``hmc`` top-level API
    (``mala = blackjax.mala(logdensity_fn, step_size)``), except ``parameters``
    is a pytree rather than individual scalar arguments, since it carries the
    PRO posterior's data and may be replaced wholesale during adaptation.

    Examples
    --------
    .. code::

        ula = parametric_ula(pro_logdensity_fn, parameters)
        state = ula.init(position)
        new_state, info = ula.step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    parameters
        The PRO posterior's parameters, including the step size.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    kernel = build_kernel(logdensity_fn)

    def init_fn(position: jax.Array, rng_key: PRNGKey | None = None) -> ULAState:
        del rng_key
        return init(position, parameters, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state: ULAState) -> tuple[ULAState, ULAInfo]:
        return kernel(rng_key, state, parameters)

    return SamplingAlgorithm(init_fn, step_fn)
