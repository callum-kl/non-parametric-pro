"""One example instance each of multimodal, heteroskedastic, skewed, and well_specified,
with a standard GP and a PRO GP fit to each and their predictive mean +- std bands
overlaid on the data, laid out as a 2x2 grid.

Mirrors `synthetic.py`'s `mode=panel`/`mode=instance`: one shared `--seed` splits into
`--num-instances` candidate draws per source (same `jr.split(PRNGKey(seed),
num_instances)` convention), and each panel picks one by index -- default index set
per-source in `_SOURCES` below, overridable from the CLI without touching the file, so
you can eyeball a panel and dial in whichever draw looks best:

    python experiments/synthetic/example_grid.py
    python experiments/synthetic/example_grid.py --well-specified-index 7
"""

import argparse
import os
from pathlib import Path
from typing import Callable, NamedTuple

# Must be set before `import jax` (and before any transitive jax import, e.g. via
# gpjax) -- setting it later doesn't reliably take effect before jax's backend
# initializes.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

# This script only ever calls savefig, never show(); force a non-interactive backend
# so it doesn't depend on a GUI toolkit being usable (e.g. Qt's xcb plugin, which
# aborts the whole process if there's no X server -- as under plain WSL).
matplotlib.use("Agg")

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from non_parametric_pro.data.heteroskedastic import (
    make_heteroskedastic_instance,
    plot_heteroskedastic_case,
)
from non_parametric_pro.data.multimodal import make_multimodal_instance, plot_multimodal_case
from non_parametric_pro.data.skewed import make_skewed_instance, plot_skewed_case
from non_parametric_pro.data.well_specified import (
    make_well_specified_instance,
    plot_well_specified_case,
)

from synthetic import FitResult, fit_gp, fit_pro

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

GP_COLOR = "C2"
PRO_COLOR = "C3"

class SourceSpec(NamedTuple):
    """One panel's dataset source: `kwargs` mirrors that dataset's own
    `conf/ds/*.yaml` defaults, so the panel looks like a "typical" draw; `plot_kwargs`
    turns off cues that clutter this specific "data + two overlaid model fits" view --
    every source's true-value curve/band, since the two model fits are the thing being
    shown -- without touching those functions' defaults, which `synthetic.py`'s
    `mode=panel` still relies on. `instance_index` picks which of the shared
    `--seed`/`--num-instances` candidate draws this panel uses -- hand-pick it here (or
    override from the CLI with --<name>-index) to get a good-looking example."""

    name: str
    title: str
    make_instance: Callable
    plot_case: Callable
    kwargs: dict
    plot_kwargs: dict
    instance_index: int


_SOURCES = [
    SourceSpec(
        "multimodal",
        "Multimodal",
        make_multimodal_instance,
        plot_multimodal_case,
        {
            "num_regions": 1, "mix_prob": 0.5, "min_width": 0.3, "max_width": 0.8,
            "ell_range": (0.5, 1.0), "noise_std_frac": 0.1,
        },
        {"show_curves": False, "color_by_branch": False},
        instance_index=3,
    ),
    SourceSpec(
        "heteroskedastic",
        "Heteroskedastic",
        make_heteroskedastic_instance,
        plot_heteroskedastic_case,
        {"num_regions": 2, "amplitude_frac": 0.8, "min_width": 0.3, "max_width": 0.8},
        {"show_curve": False, "show_noise_bands": False},
        instance_index=9,
    ),
    SourceSpec(
        "skewed",
        "Skewed",
        make_skewed_instance,
        plot_skewed_case,
        {"noise_skewness": 2.0, "noise_std_frac": 0.4},
        {"show_curve": False, "color_by_tail": False},
        instance_index=14,
    ),
    SourceSpec(
        "well_specified",
        "Well-specified",
        make_well_specified_instance,
        plot_well_specified_case,
        {"train_fraction": 0.7, 'noise_std_frac': 0.2},
        {"show_curve": False},
        instance_index=2,
    ),
]


def _overlay_gp_band(ax, x_test, result: FitResult, *, color, label) -> None:
    """Standard GP predictive: single-Gaussian mean +- std band. Sort by x and
    connect -- same convention every `plot_..._case` already uses for the true
    function."""
    order = jnp.argsort(x_test[:, 0])
    x_sorted = x_test[order, 0]
    mean_sorted, std_sorted = result.mean[order], result.std[order]

    ax.plot(x_sorted, mean_sorted, color=color, linewidth=1.5, linestyle="--", label=label)
    ax.fill_between(
        x_sorted, mean_sorted - std_sorted, mean_sorted + std_sorted,
        color=color, alpha=0.2, linewidth=0,
    )


def _overlay_pro_draws(ax, x_test, result: FitResult, *, color, label) -> None:
    """PRO GP predictive: a spaghetti plot of `posterior_function_draws` -- a mean +-
    std band would moment-match away exactly the multimodality PRO's mixture
    predictive is meant to capture (a bimodal mixture and a unimodal Gaussian can share
    the same mean/std). Each line is one smooth, coherent function draw (not per-point
    noise), so a multimodal predictive shows up as the lines visibly forking into
    separate bundles rather than one smeared-out band."""
    order = jnp.argsort(x_test[:, 0])
    x_sorted = x_test[order, 0]
    draws_sorted = result.function_draws[order]

    ax.plot(x_sorted, draws_sorted[:, 0], color=color, linewidth=0.8, alpha=0.4, label=label)
    ax.plot(x_sorted, draws_sorted[:, 1:], color=color, linewidth=0.8, alpha=0.4)


def main(seed: int, num_instances: int, index_overrides: dict[str, int]) -> None:
    # Same convention as `synthetic.py`'s `evaluate`/`debug_instance`: one shared key
    # split `num_instances` ways, so index `i` here is the *same* draw `mode=panel`
    # panel position `i` (or `mode=instance instance_index=i`) would show for that
    # source, given the same seed/num_instances/kwargs.
    keys = jr.split(jr.PRNGKey(seed), num_instances)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    handles, labels = None, None

    for ax, spec in zip(axes.flat, _SOURCES, strict=True):
        index = index_overrides.get(spec.name, spec.instance_index)
        instance_key = keys[index]

        data = spec.make_instance(instance_key, **spec.kwargs)

        # `plot_case` draws the dataset's own data/truth (using "C0"/"C1"/"black" --
        # see each source's own module); GP_COLOR/PRO_COLOR are picked to not clash.
        spec.plot_case(ax, data, **spec.plot_kwargs)

        # well_specified's fit is meant to match the data's own generative kernel
        # family (see non_parametric_pro/data/well_specified.py); every other source
        # has no such field, so this falls back to the standard fixed-RBF fit.
        kernel_type = getattr(data, "kernel_type", "rbf")

        gp_result = fit_gp(data, kernel_type=kernel_type)
        _overlay_gp_band(ax, data.x_test, gp_result, color=GP_COLOR, label="Standard GP")

        fit_key, _ = jr.split(instance_key)
        pro_result = fit_pro(data, fit_key, kernel_type=kernel_type)
        _overlay_pro_draws(ax, data.x_test, pro_result, color=PRO_COLOR, label="PRO GP")

        ax.set_title(spec.title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = FIGURES_DIR / "example_grid.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2421)
    parser.add_argument("--num-instances", type=int, default=20)
    for _spec in _SOURCES:
        parser.add_argument(
            f"--{_spec.name.replace('_', '-')}-index", type=int, default=None,
            help=f"Override this panel's instance index (default from _SOURCES: {_spec.instance_index}).",
        )
    args = parser.parse_args()

    index_overrides = {
        spec.name: getattr(args, f"{spec.name}_index")
        for spec in _SOURCES
        if getattr(args, f"{spec.name}_index") is not None
    }
    main(args.seed, args.num_instances, index_overrides)
