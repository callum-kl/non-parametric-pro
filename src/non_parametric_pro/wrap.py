from collections.abc import Callable
from typing import Any, NamedTuple

from blackjax.types import ArrayLikeTree, PRNGKey


def wrap(sampler_fn: Callable, **sampler_kwargs: Any):
    def _make_algorithm(logdensity_fn: Callable, parameters: NamedTuple):
        ld = lambda z: logdensity_fn(z, parameters)
        return sampler_fn(ld, parameters.step_size, **sampler_kwargs)

    def build_kernel(logdensity_fn: Callable) -> Callable:
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
        algorithm = _make_algorithm(logdensity_fn, parameters)
        return algorithm.init(position)

    def refresh(
        state: Any,
        parameters: NamedTuple,
        logdensity_fn: Callable,
    ) -> Any:
        algorithm = _make_algorithm(logdensity_fn, parameters)
        return algorithm.init(state.position)

    class _WrappedSampler:
        pass

    wrapped = _WrappedSampler()
    wrapped.build_kernel = build_kernel
    wrapped.init = init
    wrapped.refresh = refresh
    return wrapped
