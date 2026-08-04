"""
Corrupted Hartmann6 / Friedman1 regression benchmarks.

Ports the synthetic regression setup from Ament et al. 2024, "Robust Gaussian Processes
via Relevance Pursuit" (NeurIPS 2024), Section 6.1 / Fig. 4: the sparse-corruption model
of their Section 2.2, ``y_i = Z_i f(x_i) + (1 - Z_i) W_i`` with ``Z_i ~ Bernoulli(1 - p)``
for an outlier fraction ``p``, applied to the Hartmann6 and Friedman1 test functions.

The paper specifies the corruption *mechanism* and the outlier fractions swept per function
(5%/20% constant corruptions for Hartmann6, 10%/25% uniform corruptions for Friedman10, and
a 0.2-1.0 sweep of Student-t corruptions for both), but does not state the exact corruption
magnitude, training set size, or baseline noise level in the main text or appendix -- those
live in the authors' modified fork of
https://github.com/andrade-stats/TrimmedMarginalLikelihoodGP, which this module does not
reproduce. Defaults below are reasonable stand-ins, not verified against that code.
"""

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

HARTMANN6_DIM = 6

_HARTMANN6_ALPHA = jnp.array([1.0, 1.2, 3.0, 3.2])
_HARTMANN6_A = jnp.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
)
_HARTMANN6_P = 1e-4 * jnp.array(
    [
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
    ]
)


def hartmann6(x: jax.Array) -> jax.Array:
    """
    Evaluate the standard 6-D Hartmann function (Dixon 1978).

    ``x`` has shape ``(..., 6)`` with entries in ``[0, 1]``. The global minimum is
    ``f(x*) ≈ -3.32237`` at
    ``x* ≈ (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)``.
    """
    diff = x[..., None, :] - _HARTMANN6_P  # (..., 4, 6)
    inner = jnp.sum(_HARTMANN6_A * diff**2, axis=-1)  # (..., 4)
    return -jnp.sum(_HARTMANN6_ALPHA * jnp.exp(-inner), axis=-1)


def friedman1(x: jax.Array) -> jax.Array:
    """
    Evaluate the Friedman #1 function using the first 5 columns of ``x``.

    ``f(x) = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5``. Any additional columns
    (``dim > 5``) are nuisance/irrelevant dimensions that don't affect the output -- this
    is the standard "Friedman10" = 5 relevant + 5 nuisance dimensions convention.
    """
    x1, x2, x3, x4, x5 = (x[..., i] for i in range(5))
    return 10 * jnp.sin(jnp.pi * x1 * x2) + 20 * (x3 - 0.5) ** 2 + 10 * x4 + 5 * x5


CorruptionType = Literal["constant", "uniform", "student_t"]


class CorruptedRegressionCase(NamedTuple):
    """A corrupted-training-set regression instance with a clean test set."""

    x_train: jax.Array
    y_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_truth_train: jax.Array
    y_truth_test: jax.Array
    corrupted_idx: jax.Array
    outlier_fraction: float
    corruption_type: CorruptionType


def make_corrupted_regression_case(  # noqa: PLR0913
    key: PRNGKey,
    fn: "callable[[jax.Array], jax.Array]",
    *,
    dim: int,
    n_train: int = 50,
    n_test: int = 1000,
    sigma: float = 0.1,
    outlier_fraction: float = 0.1,
    corruption_type: CorruptionType = "constant",
    constant_value: float = 100.0,
    uniform_range: tuple[float, float] | None = None,
    student_t_df: float = 2.0,
    student_t_scale: float = 1.0,
) -> CorruptedRegressionCase:
    """
    Draw a corrupted regression instance for ``fn`` (e.g. :func:`hartmann6`, :func:`friedman1`).

    Inputs are i.i.d. ``Uniform[0, 1]^dim`` for both train and test (the standard domain for
    both Hartmann6 and Friedman1). Only the *training* set is corrupted: each training point
    is independently corrupted with probability ``outlier_fraction``, following
    ``y_i = Z_i f(x_i) + (1 - Z_i) W_i``. The test set is always clean (just observation
    noise), so predictive log-likelihood on it measures recovery of the true function despite
    training-set corruption -- matching the paper's setup.

    Parameters
    ----------
    corruption_type
        - ``"constant"``: corrupted points get a fixed value (``constant_value``),
          matching Fig. 4's "Constant Corruptions" (used for Hartmann6 at 5%/20%).
        - ``"uniform"``: corrupted points get a value drawn ``Uniform(uniform_range)``
          (defaults to the observed range of ``fn`` on this sample if not given), matching
          "Uniform Corruptions" (used for Friedman10 at 10%/25%).
        - ``"student_t"``: corrupted points get the true value plus heavy-tailed noise
          (``student_t_scale * StudentT(student_t_df)``), matching "Student-t Corruptions"
          (``student_t_df=2`` reproduces the paper's two-degrees-of-freedom sweep).
    """
    x_train_key, x_test_key, noise_key, gate_key, corrupt_key = jr.split(key, 5)

    x_train = jr.uniform(x_train_key, (n_train, dim))
    x_test = jr.uniform(x_test_key, (n_test, dim))

    y_truth_train = fn(x_train)
    y_truth_test = fn(x_test)

    y_train = y_truth_train + sigma * jr.normal(noise_key, (n_train,))
    y_test = y_truth_test + sigma * jr.normal(jr.fold_in(noise_key, 1), (n_test,))

    is_corrupted = jr.bernoulli(gate_key, outlier_fraction, (n_train,))
    corrupted_idx = jnp.flatnonzero(is_corrupted)

    if corruption_type == "constant":
        corrupted_values = jnp.full((n_train,), constant_value)
    elif corruption_type == "uniform":
        lo, hi = (
            uniform_range
            if uniform_range is not None
            else (float(y_truth_train.min()), float(y_truth_train.max()))
        )
        corrupted_values = jr.uniform(corrupt_key, (n_train,), minval=lo, maxval=hi)
    elif corruption_type == "student_t":
        t_noise = student_t_scale * jr.t(corrupt_key, student_t_df, (n_train,))
        corrupted_values = y_truth_train + t_noise
    else:
        msg = f"Unknown corruption_type: {corruption_type!r}"
        raise ValueError(msg)

    y_train = jnp.where(is_corrupted, corrupted_values, y_train)

    return CorruptedRegressionCase(
        x_train=x_train,
        y_train=y_train.reshape(-1, 1),
        x_test=x_test,
        y_test=y_test.reshape(-1, 1),
        y_truth_train=y_truth_train.reshape(-1, 1),
        y_truth_test=y_truth_test.reshape(-1, 1),
        corrupted_idx=corrupted_idx,
        outlier_fraction=outlier_fraction,
        corruption_type=corruption_type,
    )
