"""
Fixed-latent datasets used only for the illustrative fit panels of the synthetic
figure. Unlike the benchmark generators in this package, every regime here shares
one deterministic latent and hand-picked corruption magnitudes, so the panels
differ only in how they are misspecified.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

REGIMES = ("block_outliers", "heteroskedastic", "multimodal", "well_specified")

X_MIN, X_MAX = 0.0, 1.0
NUM_CURVE_POINTS = 400

WELL_SPECIFIED_NOISE_STD = 0.15

OUTLIER_REGION = (0.4, 0.55)
OUTLIER_OFFSET = 2.3
OUTLIER_NOISE_STD = 0.13

# Nearly clean up to `RAMP_START`, then the variance ramps hard to a peak at the
# latent's trough and falls away again, so the noisiest region sits on the minimum
# rather than at the edge of the domain.
HETEROSKEDASTIC_NOISE_STD_RANGE = (0.08, 0.58)
HETEROSKEDASTIC_RAMP_START = 0.5
HETEROSKEDASTIC_PEAK = 0.65

MULTIMODAL_NOISE_STD = 0.10
MULTIMODAL_GAP = 0.8
MULTIMODAL_MERGE_HALF_WIDTH = 0.1
MULTIMODAL_TRANSITION_WIDTH = 0.15

DATA_COLOR = "#707070"
# Bands are drawn over every test input, but that many dots obscures them.
MAX_PLOTTED_POINTS = 150


class IllustrativeCase(NamedTuple):
    x_train: jax.Array
    y_train: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    x_curve: jax.Array
    y_curves: jax.Array
    is_outlier_train: jax.Array


def latent(x: jax.Array) -> jax.Array:
    return jnp.sin(2 * jnp.pi * x) + 0.3 * jnp.cos(4 * jnp.pi * x)


def _jittered_grid(key: PRNGKey, n: int) -> jax.Array:
    grid = jnp.linspace(X_MIN, X_MAX, n)
    spacing = (X_MAX - X_MIN) / (n - 1)
    jitter = jr.uniform(key, (n,), minval=-0.4 * spacing, maxval=0.4 * spacing)
    return jnp.clip(grid + jitter, X_MIN, X_MAX)


def _heteroskedastic_noise_std(x: jax.Array) -> jax.Array:
    """Flat at `low` up to `HETEROSKEDASTIC_RAMP_START`, then the *variance* ramps
    linearly to `high**2` at `HETEROSKEDASTIC_PEAK` and back down to `low**2` at
    `X_MAX`."""
    low, high = HETEROSKEDASTIC_NOISE_STD_RANGE
    rising = (x - HETEROSKEDASTIC_RAMP_START) / (
        HETEROSKEDASTIC_PEAK - HETEROSKEDASTIC_RAMP_START
    )
    falling = (X_MAX - x) / (X_MAX - HETEROSKEDASTIC_PEAK)
    ramp = jnp.clip(jnp.minimum(rising, falling), 0.0, 1.0)
    return jnp.sqrt(low**2 + (high**2 - low**2) * ramp)


def _branch_gap(x: jax.Array) -> jax.Array:
    center = 0.5 * (X_MIN + X_MAX)
    distance = jnp.abs(x - center) - MULTIMODAL_MERGE_HALF_WIDTH
    return MULTIMODAL_GAP * jnp.clip(distance / MULTIMODAL_TRANSITION_WIDTH, 0.0, 1.0)


def _well_specified(key, x_train, x_test, x_curve, test_key=None):
    train_key, derived_test_key = jr.split(key)
    test_key = derived_test_key if test_key is None else test_key
    y_train = latent(x_train) + WELL_SPECIFIED_NOISE_STD * jr.normal(
        train_key, x_train.shape
    )
    y_test = latent(x_test) + WELL_SPECIFIED_NOISE_STD * jr.normal(
        test_key, x_test.shape
    )
    return y_train, y_test, latent(x_curve)[None, :], jnp.zeros_like(x_train, bool)


def _block_outliers(key, x_train, x_test, x_curve, test_key=None):
    train_key, derived_test_key = jr.split(key)
    test_key = derived_test_key if test_key is None else test_key
    lo, hi = OUTLIER_REGION
    is_outlier = (x_train > lo) & (x_train < hi)
    y_train = (
        latent(x_train)
        + OUTLIER_NOISE_STD * jr.normal(train_key, x_train.shape)
        + jnp.where(is_outlier, OUTLIER_OFFSET, 0.0)
    )
    y_test = latent(x_test) + OUTLIER_NOISE_STD * jr.normal(test_key, x_test.shape)
    return y_train, y_test, latent(x_curve)[None, :], is_outlier


def _heteroskedastic(key, x_train, x_test, x_curve, test_key=None):
    train_key, derived_test_key = jr.split(key)
    test_key = derived_test_key if test_key is None else test_key
    y_train = latent(x_train) + _heteroskedastic_noise_std(x_train) * jr.normal(
        train_key, x_train.shape
    )
    y_test = latent(x_test) + _heteroskedastic_noise_std(x_test) * jr.normal(
        test_key, x_test.shape
    )
    return y_train, y_test, latent(x_curve)[None, :], jnp.zeros_like(x_train, bool)


def _multimodal(key, x_train, x_test, x_curve, test_key=None):
    train_key, derived_test_key, train_branch_key, derived_branch_key = jr.split(key, 4)
    test_key, test_branch_key = (
        (derived_test_key, derived_branch_key)
        if test_key is None
        else jr.split(test_key)
    )

    def branch(x, branch_key, noise_key):
        sign = jnp.where(jr.bernoulli(branch_key, 0.5, x.shape), 1.0, -1.0)
        mean = latent(x) + sign * _branch_gap(x)
        return mean + MULTIMODAL_NOISE_STD * jr.normal(noise_key, x.shape)

    y_train = branch(x_train, train_branch_key, train_key)
    y_test = branch(x_test, test_branch_key, test_key)

    gap = _branch_gap(x_curve)
    curves = jnp.stack([latent(x_curve) - gap, latent(x_curve) + gap])
    return y_train, y_test, curves, jnp.zeros_like(x_train, bool)


_REGIME_FNS = {
    "well_specified": _well_specified,
    "block_outliers": _block_outliers,
    "heteroskedastic": _heteroskedastic,
    "multimodal": _multimodal,
}


def make_illustrative_instance(
    key: PRNGKey,
    *,
    regime: str,
    n_train: int = 100,
    n_test: int = 400,
    test_key: PRNGKey | None = None,
) -> IllustrativeCase:
    """`test_key` rerolls the held-out set -- its inputs and its noise -- while leaving
    the training data, and so the fit, untouched. Left as None the whole instance comes
    from `key` alone."""
    try:
        regime_fn = _REGIME_FNS[regime]
    except KeyError:
        msg = f"Unknown regime={regime!r}; expected one of {REGIMES}"
        raise ValueError(msg) from None

    train_x_key, test_x_key, regime_key = jr.split(key, 3)
    regime_test_key = None
    if test_key is not None:
        test_x_key, regime_test_key = jr.split(test_key)

    x_train = _jittered_grid(train_x_key, n_train)
    x_test = jr.uniform(test_x_key, (n_test,), minval=X_MIN, maxval=X_MAX)
    x_curve = jnp.linspace(X_MIN, X_MAX, NUM_CURVE_POINTS)

    y_train, y_test, y_curves, is_outlier = regime_fn(
        regime_key, x_train, x_test, x_curve, test_key=regime_test_key
    )

    return IllustrativeCase(
        x_train=x_train.reshape(-1, 1),
        y_train=y_train.reshape(-1, 1),
        x_test=x_test.reshape(-1, 1),
        y_test=y_test.reshape(-1, 1),
        x_curve=x_curve,
        y_curves=y_curves,
        is_outlier_train=is_outlier,
    )


def plotted_test_points(
    data: IllustrativeCase, max_points: int = MAX_PLOTTED_POINTS
) -> tuple[jax.Array, jax.Array]:
    """An evenly-strided subset of the test set. The inputs are i.i.d. uniform, so
    striding is an unbiased subsample, and being deterministic it keeps the drawn
    points and the display range in step."""
    n = data.x_test.shape[0]
    if n <= max_points:
        return data.x_test, data.y_test
    stride = n // max_points
    return data.x_test[::stride][:max_points], data.y_test[::stride][:max_points]


def plot_illustrative_case(
    ax,
    data: IllustrativeCase,
    *,
    data_alpha: float = 0.45,
    scale: float = 1.0,
    shift: float = 0.0,
    max_points: int = MAX_PLOTTED_POINTS,
) -> None:
    """`scale`/`shift` apply an affine `y -> scale * y + shift` to everything drawn,
    letting a panel be fitted on the generator's own scale but displayed inside a
    range shared with the other panels."""

    def to_display(y):
        return scale * y + shift

    for curve in data.y_curves:
        ax.plot(
            data.x_curve,
            to_display(curve),
            color="black",
            linestyle="--",
            linewidth=1.3,
            alpha=0.7,
        )

    # held-out points, so the block-outlier panel's dots stay clean; only the
    # corrupted training points are overlaid, as markers
    x_plot, y_plot = plotted_test_points(data, max_points)
    ax.scatter(
        x_plot,
        to_display(y_plot),
        color=DATA_COLOR,
        s=12,
        alpha=data_alpha,
        zorder=3,
    )

    outlier_idx = jnp.where(data.is_outlier_train)[0]
    if outlier_idx.size > 0:
        ax.scatter(
            data.x_train[outlier_idx],
            to_display(data.y_train[outlier_idx]),
            color="maroon",
            marker="x",
            s=40,
            linewidths=1.5,
            zorder=4,
        )
