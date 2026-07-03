"""
Adapter making any BlackJAX MCMC sampler compatible with the parametric
protocol expected by ``parameter_adaptation``.

The protocol requires three module-level callables::

    build_kernel(logdensity_fn) -> Callable[[rng_key, state, parameters], (state, info)]
    init(position, parameters, logdensity_fn) -> State
    refresh(state, parameters, logdensity_fn) -> State

Standard BlackJAX samplers close over ``logdensity_fn`` at construction time,
so they cannot be updated mid-scan without recompilation.  The adapter works
around this by constructing the closure *inside* the scan body, where
``parameters`` is a traced JAX pytree — the closure then captures the symbolic
value, so parameter updates flow through correctly without retracing.

Usage
-----
::

    import blackjax
    from non_parametric_pro.blackjax_wrapper import wrap_blackjax
    from non_parametric_pro.adaptation.parameter_adaptation import parameter_adaptation

    mala = wrap_blackjax(blackjax.mala)

    adaptation = parameter_adaptation(
        mala,                       # drop-in for the custom `ula` module
        pro_logdensity_fn,
        params,
        x_train=x_train,
        initial_kernel=kernel,
        ...
    )

Notes
-----
- ``parameters.step_size`` is used as the BlackJAX sampler's ``step_size``
  argument. Ensure the wrapped sampler accepts it as its second positional
  argument (true for ``blackjax.mala``, ``blackjax.sgld``, etc.).
- The returned state type is whatever the wrapped sampler produces.  It must
  have a ``position`` field for ``refresh`` to work; all standard BlackJAX
  MCMC states satisfy this.
- For samplers that require additional static constructor arguments (e.g.
  ``blackjax.nuts`` with ``max_num_doublings``), pass them via ``**kwargs``
  to ``wrap_blackjax``.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
from blackjax.types import ArrayLikeTree, PRNGKey


def wrap(sampler_fn: Callable, **sampler_kwargs: Any):
    """
    Wrap a standard BlackJAX sampler into the parametric protocol.

    Parameters
    ----------
    sampler_fn
        A BlackJAX sampler factory such as ``blackjax.mala`` or
        ``blackjax.nuts``.  It must accept ``(logdensity_fn, step_size,
        **sampler_kwargs)`` and return an object with ``.init`` and ``.step``.
    **sampler_kwargs
        Extra static keyword arguments forwarded to ``sampler_fn`` on every
        call (e.g. ``max_num_doublings=10`` for NUTS).

    Returns
    -------
    A module-like object with ``build_kernel``, ``init``, and ``refresh``
    matching the protocol used by ``parameter_adaptation``.
    """

    def _make_algorithm(logdensity_fn: Callable, parameters: NamedTuple):
        """Construct a fresh sampler instance closing over current parameters."""
        ld = lambda z: logdensity_fn(z, parameters)  # noqa: E731
        return sampler_fn(ld, parameters.step_size, **sampler_kwargs)

    def build_kernel(logdensity_fn: Callable) -> Callable:
        """
        Return a kernel with the signature ``(rng_key, state, parameters)``.

        The closure over ``logdensity_fn`` and ``parameters`` is rebuilt on
        every call, but because this happens *inside* ``lax.scan``, JAX traces
        through the construction and ``parameters`` remains a live traced value.
        """

        def kernel(
            rng_key: PRNGKey,
            state: Any,
            parameters: NamedTuple,
        ) -> tuple[Any, Any]:
            algorithm = _make_algorithm(logdensity_fn, parameters)
            return algorithm.step(rng_key, state)

        return kernel

    def init(
        position: ArrayLikeTree,
        parameters: NamedTuple,
        logdensity_fn: Callable,
    ) -> Any:
        """Initialise sampler state from a starting position."""
        algorithm = _make_algorithm(logdensity_fn, parameters)
        return algorithm.init(position)

    def refresh(
        state: Any,
        parameters: NamedTuple,
        logdensity_fn: Callable,
    ) -> Any:
        """
        Rebuild cached logdensity/grad after parameters change.

        Equivalent to re-initialising from the current position, which is what
        BlackJAX's ``init`` does for gradient-based samplers.
        """
        algorithm = _make_algorithm(logdensity_fn, parameters)
        return algorithm.init(state.position)

    # Return a namespace object so callers can do ``mala.build_kernel(...)``
    # etc., matching the module-level protocol of the custom ``ula`` module.
    class _WrappedSampler:
        pass

    wrapped = _WrappedSampler()
    wrapped.build_kernel = build_kernel
    wrapped.init = init
    wrapped.refresh = refresh
    return wrapped
