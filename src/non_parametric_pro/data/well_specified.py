"""Well-specified regression instances: fit kernel family matches the true generative one.

Unlike every other dataset in this experiment suite -- which deliberately introduces
some form of misspecification (heteroscedastic noise, hidden regime switches, heavy
tails, ...) that a standard homoscedastic-Gaussian-likelihood GP with a generic fit
kernel can't capture -- this one is the control. The latent function is drawn from a
GP prior whose kernel *family* (RBF, Matern-1/2, Matern-3/2, Matern-5/2) is itself
randomised per instance, and `kernel_type` is exposed on the returned case so the
fitting code (`experiments/synthetic/synthetic.py`'s `fit_gp`/`fit_pro`) can match it
via `build_kernel` -- i.e. fit with the *same* kernel family that generated the data,
not a fixed generic one. This isolates "how much does misspecification cost" (the
other datasets) from "how well do these methods do when nothing is wrong" (this one),
and lets you sweep dataset size (`n`, at a fixed train/test split ratio) to see how
each method's calibration/accuracy scales with data even in the ideal case.
"""

from typing import NamedTuple

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import PRNGKey

from non_parametric_pro.util import train_val_split

# Matern-1/2, -3/2, -5/2 span rough (non-differentiable) to smooth (twice
# differentiable); RBF is the infinitely-smooth limit -- together they cover a
# meaningfully different range of sample-path roughness a "kernel family" sweep should
# exercise, without trying to be an exhaustive list of gpjax's stationary kernels.
KERNEL_TYPES = ("rbf", "matern12", "matern32", "matern52")

_KERNEL_CONSTRUCTORS = {
    "rbf": gpx.kernels.RBF,
    "matern12": gpx.kernels.Matern12,
    "matern32": gpx.kernels.Matern32,
    "matern52": gpx.kernels.Matern52,
}


def build_kernel(kernel_type: str, *, lengthscale, variance=None) -> gpx.kernels.AbstractKernel:
    """Construct a kernel of the given family -- the single place that maps a
    `kernel_type` string (as stored on `WellSpecifiedCase`, or set via a fit config) to
    a gpjax kernel class, so data generation and model fitting can't drift apart."""
    try:
        kernel_cls = _KERNEL_CONSTRUCTORS[kernel_type]
    except KeyError:
        msg = f"Unknown kernel_type={kernel_type!r}; expected one of {KERNEL_TYPES}"
        raise ValueError(msg) from None
    if variance is None:
        return kernel_cls(lengthscale=lengthscale)
    return kernel_cls(lengthscale=lengthscale, variance=variance)


class WellSpecifiedCase(NamedTuple):
    """A synthetic regression instance where the fit model's kernel family can exactly
    match the data-generating kernel -- the "nothing is misspecified" baseline."""

    x_train: jax.Array
    y_train: jax.Array
    y_truth_train: jax.Array
    x_test: jax.Array
    y_truth_test: jax.Array
    y_test: jax.Array
    noise_std: float
    ell: float
    alpha: float
    kernel_type: str
    n: int


def make_well_specified_instance(  # noqa: PLR0913
    key: PRNGKey,
    *,
    n: int = 300,
    train_fraction: float = 0.2,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std_frac: float = 0.1,
    ell_range: tuple[float, float] = (0.15, 0.5),
    alpha_range: tuple[float, float] = (0.5, 2.0),
    kernel_types: tuple[str, ...] = KERNEL_TYPES,
) -> WellSpecifiedCase:
    """Draw a well-specified regression instance.

    `kernel_type` is randomised *per instance* (not just lengthscale/variance): the
    point of this dataset is to check "what happens when the fit kernel family exactly
    matches the truth" across a spread of families, not to validate any one family.
    `n` (the pooled dataset size, at a fixed `train_fraction` split) is the swept
    parameter here, rather than `train_fraction` itself -- the interesting question for
    a well-specified model is "how does it do with more/less data", which is a
    statement about absolute dataset size, not about what fraction of a fixed pool goes
    to training.
    """
    x_key, ell_key, alpha_key, kernel_key, latent_key, split_key, noise_key = jr.split(key, 7)

    ell = jr.uniform(ell_key, (), minval=ell_range[0], maxval=ell_range[1])
    alpha = jr.uniform(alpha_key, (), minval=alpha_range[0], maxval=alpha_range[1])
    kernel_idx = jr.randint(kernel_key, (), 0, len(kernel_types))
    kernel_type = kernel_types[int(kernel_idx)]

    kernel = build_kernel(kernel_type, lengthscale=ell, variance=alpha)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)

    x = jr.uniform(x_key, (n, 1), minval=x_min, maxval=x_max)
    y_truth = prior.predict(x).sample(latent_key)

    noise_std = noise_std_frac * jnp.sqrt(alpha)
    y_obs = y_truth + noise_std * jr.normal(noise_key, y_truth.shape)

    # `train_val_split` only splits one (x, y) pair -- reuse its train/val indices to
    # split y_truth/y_obs consistently rather than calling it twice. `train_fraction`
    # is fixed here (unlike `n`, which is the swept parameter -- see above), so invert
    # it into the `val_fraction` the shared helper expects.
    split = train_val_split(split_key, x, y_truth, val_fraction=1.0 - train_fraction)
    train_idx, test_idx = split.train_idx, split.val_idx

    return WellSpecifiedCase(
        x_train=x[train_idx],
        y_train=y_obs[train_idx].reshape(-1, 1),
        y_truth_train=y_truth[train_idx].reshape(-1, 1),
        x_test=x[test_idx],
        y_truth_test=y_truth[test_idx].reshape(-1, 1),
        y_test=y_obs[test_idx].reshape(-1, 1),
        noise_std=float(noise_std),
        ell=float(ell),
        alpha=float(alpha),
        kernel_type=kernel_type,
        n=n,
    )


def plot_well_specified_case(ax, data: WellSpecifiedCase) -> None:
    """Plot one WellSpecifiedCase: the single latent truth curve and the noisy
    observed training points -- no misspecification cue to highlight (no region, no
    hidden branch, no outliers), just the kernel family/hyperparameters/dataset size
    in the title, since those are the only things that vary "difficulty" here."""
    x_full = jnp.concatenate([data.x_train[:, 0], data.x_test[:, 0]])
    y_full = jnp.concatenate([data.y_truth_train[:, 0], data.y_truth_test[:, 0]])
    order = jnp.argsort(x_full)
    x_sorted, y_sorted = x_full[order], y_full[order]

    ax.plot(x_sorted, y_sorted, color="C0", linewidth=1.5)
    ax.scatter(data.x_train, data.y_train, color="black", s=8, zorder=3)
    ax.set_title(
        f"{data.kernel_type}  $\\ell$={data.ell:.2f}  $\\alpha$={data.alpha:.2f}  "
        f"n={data.n} (train={data.x_train.shape[0]})",
        fontsize=9,
    )


# Allowed keys a `ds` config (experiments/synthetic/conf/ds/*.yaml) can set on top of
# `source` itself; only these are forwarded to `make_well_specified_instance`, so a
# config can override any subset without a code change in synthetic.py.
WELL_SPECIFIED_KWARGS = (
    "n", "train_fraction", "x_min", "x_max", "noise_std_frac",
    "ell_range", "alpha_range", "kernel_types",
)
