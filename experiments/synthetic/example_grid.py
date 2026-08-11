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
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

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

# GP green vs. PRO blue -- a hue contrast, which holds up far better than a
# lightness-only contrast (e.g. grey vs. red) once PRO is drawn semi-transparent on top
# and the two start blending. GP is opaque, drawn first, so it reads as a plain
# background reference; PRO draws on top with alpha < 1 (see PRO_ALPHA), so GP still
# shows through wherever they overlap instead of being fully occluded -- important
# since the two bands are often nearly coincident (that overlap *is* the interesting
# comparison). Each also gets a crisp outline in its own full color at every band
# boundary (see _masked_density_bands' edge_color), so shapes stay readable even in the
# muddiest overlap regions where the fills alone are ambiguous.
GP_COLOR = "#3f8f5f"
PRO_COLOR = "#3a76c4"
GP_CMAP = LinearSegmentedColormap.from_list("gp_density", ["#e8f4ec", GP_COLOR])
PRO_CMAP = LinearSegmentedColormap.from_list("pro_density", ["#e6eef8", PRO_COLOR])
PRO_ALPHA = 0.8

# Paper-figure formatting: no ticks (this is a qualitative comparison, not meant for
# reading off values), no spines (nothing left for them to frame once ticks are gone),
# a faint grid for scale reference instead.
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

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
        {"num_regions": 2, "min_width": 0.3, "max_width": 0.8},
        {"show_curve": False, "show_noise_bands": False},
        instance_index=15,
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


# Shared by both overlays below, so "similar proportion of predictive density shown"
# is true by construction rather than eyeballed: same credible_k cutoff, same number
# of band levels, same min_density_frac floor, for both models.
_DENSITY_DEFAULTS = {"num_bands": 4, "credible_k": 5.0, "num_y": 150, "min_density_frac": 0.03}


def _masked_density_bands(  # noqa: PLR0913
    ax, x_sorted, mean_sorted, std_sorted, density_fn, *,
    cmap, num_bands, credible_k, num_y, min_density_frac, alpha=1.0, edge_color=None,
) -> None:
    """Render `density_fn(y_grid) -> (N_x, N_y) p(y|x)` as a small number of discrete
    filled bands, masked to each x's own `mean +- credible_k*std` -- so nothing is
    drawn outside the region a moment-based (mean/std) plot would already call
    plausible -- and floored at `min_density_frac` of the panel's peak, which is what
    keeps a handful of discrete bands from looking like one blurry wash rather than a
    smooth continuous heatmap. `alpha < 1` lets whatever's already drawn underneath
    (e.g. the other model's band) show through where the two overlap. `edge_color`
    (typically the model's own full-saturation color, not the light end of `cmap`)
    strokes a thin outline at every band boundary, so shapes stay legible even where
    the fills of two overlapping bands are close enough in color to blur together.
    """
    y_lo = float(jnp.min(mean_sorted - credible_k * std_sorted))
    y_hi = float(jnp.max(mean_sorted + credible_k * std_sorted))
    y_grid = jnp.linspace(y_lo, y_hi, num_y)

    density = density_fn(y_grid)
    in_band = jnp.abs(y_grid[None, :] - mean_sorted[:, None]) <= credible_k * std_sorted[:, None]
    density = np.asarray(jnp.where(in_band, density, jnp.nan))

    peak = np.nanmax(density)
    levels = np.linspace(min_density_frac, 1.0, num_bands) * peak

    x_grid, y_mesh = np.meshgrid(np.asarray(x_sorted), np.asarray(y_grid), indexing="ij")
    ax.contourf(x_grid, y_mesh, density, levels=levels, cmap=cmap, alpha=alpha, extend="neither")
    if edge_color is not None:
        ax.contour(x_grid, y_mesh, density, levels=levels, colors=edge_color, linewidths=0.6, alpha=alpha)
    ax.plot(x_sorted, mean_sorted, color=edge_color or cmap(1.0), linewidth=1.4, alpha=alpha)


def _overlay_gp_density(ax, x_test, result: FitResult, *, cmap, alpha=1.0, **density_kwargs) -> None:
    """Standard GP predictive: p(y|x) for its single Gaussian mean/std, rendered with
    the exact same discrete-band/credible-mask convention as `_overlay_pro_density`
    (identical defaults, see `_DENSITY_DEFAULTS`) -- so the two overlays show a
    genuinely comparable *proportion* of predictive density, not just visually similar
    bands. A plain mean +- 1std band has no way to match that masking convention.
    Drawn fully opaque (default) as the background layer -- see PRO_ALPHA on the other
    overlay, which is what actually lets this show through underneath it."""
    order = jnp.argsort(x_test[:, 0])
    x_sorted = x_test[order, 0]
    mean_sorted, std_sorted = result.mean[order], result.std[order]

    def density_fn(y_grid):
        z = (y_grid[None, :] - mean_sorted[:, None]) / std_sorted[:, None]
        return jnp.exp(-0.5 * z**2) / (std_sorted[:, None] * jnp.sqrt(2 * jnp.pi))

    _masked_density_bands(
        ax, x_sorted, mean_sorted, std_sorted, density_fn, cmap=cmap, alpha=alpha, edge_color=GP_COLOR,
        **{**_DENSITY_DEFAULTS, **density_kwargs},
    )


def _overlay_pro_density(ax, x_test, result: FitResult, *, cmap, alpha=1.0, **density_kwargs) -> None:
    """PRO GP predictive: the exact mixture-of-Gaussians density p(y|x) -- rendered
    directly (not sampled), as a small number of discrete filled bands rather than a
    smooth continuous heatmap, so it stays a light complement to the data rather than a
    visually heavy layer. A mean +- std band would moment-match away exactly the
    multimodality this is meant to show (a bimodal mixture and a unimodal Gaussian can
    share the same mean/std); unlike posterior-function-draws spaghetti, this reads off
    the density directly (no "how many draws is enough" question), at the cost of being
    a per-x marginal -- it can't show whether the upper ridge at one x and the upper
    ridge at another belong to the same coherent function the way draws can. Drawn on
    top of the standard GP's band with `alpha < 1` (see PRO_ALPHA) so the two remain
    distinguishable wherever they overlap, rather than PRO fully occluding GP."""
    order = jnp.argsort(x_test[:, 0])
    x_sorted = x_test[order, 0]
    mean_sorted = result.mean[order]
    std_sorted = result.std[order]
    particle_predictions_sorted = result.particle_predictions[order]  # (N_x, J)
    sigma_eff_sorted = result.sigma_eff[order]  # (N_x,)

    def density_fn(y_grid):
        # Same sigma_eff each particle is scored with in nlpd_pro, just evaluated on a
        # dense y grid instead of at the observed y_test.
        z = (y_grid[None, :, None] - particle_predictions_sorted[:, None, :]) / sigma_eff_sorted[:, None, None]
        normal_pdf = jnp.exp(-0.5 * z**2) / (sigma_eff_sorted[:, None, None] * jnp.sqrt(2 * jnp.pi))
        return jnp.mean(normal_pdf, axis=2)

    _masked_density_bands(
        ax, x_sorted, mean_sorted, std_sorted, density_fn, cmap=cmap, alpha=alpha, edge_color=PRO_COLOR,
        **{**_DENSITY_DEFAULTS, **density_kwargs},
    )


def main(seed: int, num_instances: int, index_overrides: dict[str, int]) -> None:
    keys = jr.split(jr.PRNGKey(seed), num_instances)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

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
        _overlay_gp_density(ax, data.x_test, gp_result, cmap=GP_CMAP)

        fit_key, _ = jr.split(instance_key)
        pro_result = fit_pro(data, fit_key, kernel_type=kernel_type)
        _overlay_pro_density(ax, data.x_test, pro_result, cmap=PRO_CMAP, alpha=PRO_ALPHA)

        ax.set_title(spec.title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    # contourf isn't a labeled artist `ax.get_legend_handles_labels()` picks up, so
    # build the shared legend from explicit proxies instead of collecting from an axes.
    legend_handles = [
        Patch(color=GP_COLOR, label="Standard GP"),
        Patch(color=PRO_COLOR, label="PRO GP"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10, frameon=False)
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
