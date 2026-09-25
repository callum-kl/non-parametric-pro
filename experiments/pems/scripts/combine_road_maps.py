"""Place the predictive-std road maps side by side, one model per column."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from plot_map import STD_VMAX, STD_VMIN

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

QUANTITY = "uncertainty"

# Same palette as the synthetic misspecification figures, via plot_nlpd.py.
GP_COLOR = "#e8974e"
PRO_COLOR = "#2ca58d"
PANELS = [("graph_gp", "Graph GP", GP_COLOR), ("pro_gp", "Graph PrO-GP", PRO_COLOR)]

TITLE_FONTSIZE = 32
CBAR_LABEL = "Predictive std"
CBAR_LABEL_FONTSIZE = 24
CBAR_CMAP = "plasma"  # the cmap_name the vendored plot_PEMS draws these maps with
CBAR_TICK_FONTSIZE = 17
CBAR_WIDTH_FRAC = 0.86  # of its map's width

# inches reserved around the map row
TITLE_IN = 0.70
CBAR_IN = 0.30
CBAR_GAP_IN = 0.14
CBAR_TICKS_IN = 1.10  # tick labels plus the colorbar's own label


OUTER_MARGIN_PX = 6
INNER_GAP_PX = 28


def _trim_whitespace(image, *, outer=OUTER_MARGIN_PX, inner=INNER_GAP_PX):
    """Drop the blank margins and shrink the wide blank gutter each panel leaves
    between its map and its colorbar."""
    ink = ~np.all(image[..., :3] > 0.99, axis=(0, 2))
    (columns,) = np.nonzero(ink)
    keep = np.zeros(image.shape[1], bool)
    keep[max(columns[0] - outer, 0) : columns[-1] + outer + 1] = True

    gaps = np.nonzero(np.diff(columns) > inner)[0]
    for g in gaps:  # leave `inner` blank columns either side of the gutter
        keep[columns[g] + inner : columns[g + 1] - inner] = False

    rows = np.nonzero(~np.all(image[..., :3] > 0.99, axis=(1, 2)))[0]
    top, bottom = max(rows[0] - outer, 0), min(rows[-1] + outer + 1, image.shape[0])
    return image[top:bottom, keep]


def _drop_colorbar(image, *, inner=INNER_GAP_PX):
    """Keep only the map: everything up to the blank gutter before the colorbar."""
    (columns,) = np.nonzero(~np.all(image[..., :3] > 0.99, axis=(0, 2)))
    gaps = np.nonzero(np.diff(columns) > inner)[0]
    if gaps.size == 0:
        return image
    return image[:, : columns[gaps[0]] + inner]


def main(out_name: str) -> None:
    panels = [
        _trim_whitespace(mpimg.imread(FIGURES_DIR / f"{prefix}_road_map_{QUANTITY}.png"))
        for prefix, _, _ in PANELS
    ]
    # both panels share one scale, replaced below by a single horizontal colorbar
    panels = [_drop_colorbar(image) for image in panels]

    widths = [image.shape[1] for image in panels]
    height = panels[0].shape[0]
    scale = 7.0 / max(widths)  # inches per pixel, widest map 7in across
    map_height_in = scale * height

    fig_height = TITLE_IN + map_height_in + CBAR_GAP_IN + CBAR_IN + CBAR_TICKS_IN
    fig = plt.figure(figsize=(scale * sum(widths), fig_height))
    gs = fig.add_gridspec(
        2,
        len(PANELS),
        height_ratios=[map_height_in, CBAR_IN],
        width_ratios=widths,  # so every map renders at one size
        left=0.005,
        right=0.995,
        top=1.0 - TITLE_IN / fig_height,
        bottom=CBAR_TICKS_IN / fig_height,
        hspace=CBAR_GAP_IN / ((map_height_in + CBAR_IN) / 2),
        wspace=0.03,
    )
    axes = [fig.add_subplot(gs[0, col]) for col in range(len(PANELS))]

    for col, (ax, image, (_, title, colour)) in enumerate(
        zip(axes, panels, PANELS, strict=True)
    ):
        ax.imshow(image, interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(title, fontsize=TITLE_FONTSIZE, color=colour, pad=10)

    for col in range(len(PANELS)):
        cax = fig.add_subplot(gs[1, col])
        box = cax.get_position()
        width = CBAR_WIDTH_FRAC * box.width
        cax.set_position(
            [box.x0 + (box.width - width) / 2, box.y0, width, box.height]
        )
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(Normalize(STD_VMIN, STD_VMAX), cmap=CBAR_CMAP),
            cax=cax,
            orientation="horizontal",
        )
        cbar.set_ticks(np.arange(STD_VMIN, STD_VMAX + 1, 2))
        cbar.set_label(CBAR_LABEL, fontsize=CBAR_LABEL_FONTSIZE)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="road_map_grid_std_h.png")
    args = parser.parse_args()
    main(args.out)
