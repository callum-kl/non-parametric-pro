"""Combine `example_grid.png` and `summary_panel.png` side by side into one image --
`example_grid.py`'s qualitative fit comparison on the left, `plot_summary.py`'s
quantitative NLPD comparison on the right. Both already cover the same four datasets in
the same order (see each script's own `_SOURCES`/`SOURCES`), so reading them side by
side ties a panel's visual behavior directly to its NLPD numbers.

Rescales both source images to a common height -- preserving each's own aspect ratio,
since example_grid.png's 2x2 grid of dense fit overlays and summary_panel.png's 2x2
grid of sparse error bars aren't the same shape -- rather than assuming they already
match or stretching one to fit the other.

    python experiments/synthetic/combine_grid_summary.py
"""

from pathlib import Path

from PIL import Image

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

GAP_PX = 24
BACKGROUND = (255, 255, 255, 255)


def combine_side_by_side(left_path: Path, right_path: Path, *, gap_px: int = GAP_PX) -> Image.Image:
    """Paste `left_path`'s image and `right_path`'s image side by side, each rescaled
    to the taller of the two source heights (preserving its own aspect ratio) so
    neither one gets stretched/squashed to match the other."""
    left = Image.open(left_path).convert("RGBA")
    right = Image.open(right_path).convert("RGBA")

    target_height = max(left.height, right.height)
    left = left.resize((round(left.width * target_height / left.height), target_height), Image.LANCZOS)
    right = right.resize((round(right.width * target_height / right.height), target_height), Image.LANCZOS)

    combined = Image.new("RGBA", (left.width + gap_px + right.width, target_height), BACKGROUND)
    combined.paste(left, (0, 0), left)
    combined.paste(right, (left.width + gap_px, 0), right)
    return combined


if __name__ == "__main__":
    combined = combine_side_by_side(FIGURES_DIR / "example_grid.png", FIGURES_DIR / "summary_panel.png")
    out_path = FIGURES_DIR / "example_grid_and_summary.png"
    combined.convert("RGB").save(out_path)
    print(f"Saved to {out_path}")
