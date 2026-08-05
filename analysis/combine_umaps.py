"""Create a compact poster panel from existing training-time UMAP images."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


BASE_DIR = Path("analysis_results/embedding_visualizations/grokking")

OUTPUT_PATH = Path(
    "analysis_results/embedding_visualizations/"
    "umap_training_progression_compact.png"
)

MANIFOLDS = [
    ("grid", "Grid"),
    ("mobius", "Möbius strip"),
    ("torus", "Flat torus"),
    ("octahedron", "Octahedron"),
]

CHECKPOINTS = [
    (1000, "Early\n1,000 updates"),
    (10000, "Middle\n10,000 updates"),
    (50000, "Final\n50,000 updates"),
]


def get_image_path(world: str, step: int) -> Path:
    """Return the existing UMAP image path."""

    suffix = "_graph" if world == "grid" else ""

    return (
        BASE_DIR
        / world
        / f"{world}_pairs3000_step{step}_umap{suffix}.png"
    )


def crop_umap(image: Image.Image) -> Image.Image:
    """
    Crop an existing UMAP plot down to the central plotting region.

    The source plots include:
      - a long title at the top,
      - axis labels and ticks,
      - a colorbar on the right,
      - outer white margins.

    These fractions are based on the current generated UMAP layout.
    """

    width, height = image.size

    left = int(width * 0.105)
    top = int(height * 0.105)
    right = int(width * 0.805)
    bottom = int(height * 0.885)

    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid crop dimensions for image size {image.size}"
        )

    return image.crop((left, top, right, bottom))


def main() -> None:
    rows = len(CHECKPOINTS)
    columns = len(MANIFOLDS)

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=columns,
        figsize=(11, 7.5),
    )

    for row, (step, step_label) in enumerate(CHECKPOINTS):
        for column, (world, world_label) in enumerate(MANIFOLDS):

            path = get_image_path(world, step)

            with Image.open(path) as source:
                cropped = crop_umap(source.convert("RGB"))

            ax = axes[row, column]
            ax.imshow(cropped)
            ax.set_axis_off()

            # Column titles
            if row == 0:
                ax.set_title(
                    world_label,
                    fontsize=16,
                    fontweight="semibold",
                    pad=5,
                )

            # Row labels
            if column == 0:
                ax.text(
                    -0.05,
                    0.5,
                    step_label,
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=15,
                    fontweight="bold",
                    ha="right",
                    va="center",
                )

    # Remove almost all whitespace between images.
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.008,
        top=0.945,
        wspace=0.005,
        hspace=0.008,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.015,
    )

    plt.close(fig)

    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
