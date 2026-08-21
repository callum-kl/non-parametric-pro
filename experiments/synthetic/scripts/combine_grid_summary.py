from pathlib import Path

from PIL import Image

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

GAP_PX = 24
BACKGROUND = (255, 255, 255, 255)


def combine_side_by_side(
    left_path: Path, right_path: Path, *, gap_px: int = GAP_PX
) -> Image.Image:
    left = Image.open(left_path).convert("RGBA")
    right = Image.open(right_path).convert("RGBA")

    target_height = max(left.height, right.height)
    left = left.resize(
        (round(left.width * target_height / left.height), target_height), Image.LANCZOS
    )
    right = right.resize(
        (round(right.width * target_height / right.height), target_height),
        Image.LANCZOS,
    )

    combined = Image.new(
        "RGBA", (left.width + gap_px + right.width, target_height), BACKGROUND
    )
    combined.paste(left, (0, 0), left)
    combined.paste(right, (left.width + gap_px, 0), right)
    return combined


if __name__ == "__main__":
    combined = combine_side_by_side(
        FIGURES_DIR / "example_grid.png", FIGURES_DIR / "summary_panel.png"
    )
    out_path = FIGURES_DIR / "example_grid_and_summary.png"
    combined.convert("RGB").save(out_path)
    print(f"Saved to {out_path}")
