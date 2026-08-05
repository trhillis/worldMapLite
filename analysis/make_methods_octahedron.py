"""Create a clean octahedron illustration of sparse pairwise supervision."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


N_POINTS = 120
N_RED_PAIRS = 7
SEED = 11

OUTPUT = Path(
    "analysis_results/methods/octahedron_training_pairs_clean.png"
)

rng = np.random.default_rng(SEED)


BACKGROUND = "#344a9a"
SURFACE = "#7f95d6"
SURFACE_EDGE = "#b8c6f0"
POINT_FILL = "#ffffff"
POINT_EDGE = "#111827"
PAIR_RED = "#ff1c1c"
HIGHLIGHT = "#ffd84d"
TEXT = "#ffffff"


VERTICES = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)


FACES = [
    (0, 2, 4),
    (2, 1, 4),
    (1, 3, 4),
    (3, 0, 4),
    (2, 0, 5),
    (1, 2, 5),
    (3, 1, 5),
    (0, 3, 5),
]


def triangle_area(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    return 0.5 * np.linalg.norm(
        np.cross(b - a, c - a)
    )


def sample_triangle_uniform(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Uniformly sample one point from a triangle."""

    r1 = rng.random()
    r2 = rng.random()

    sqrt_r1 = np.sqrt(r1)

    return (
        (1.0 - sqrt_r1) * a
        + sqrt_r1 * (1.0 - r2) * b
        + sqrt_r1 * r2 * c
    )


def sample_octahedron_surface(
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points uniformly over the octahedron's triangular faces."""

    face_areas = np.array(
        [
            triangle_area(
                VERTICES[i],
                VERTICES[j],
                VERTICES[k],
            )
            for i, j, k in FACES
        ]
    )

    probabilities = face_areas / face_areas.sum()

    face_indices = rng.choice(
        len(FACES),
        size=count,
        p=probabilities,
    )

    points = []

    for face_index in face_indices:
        i, j, k = FACES[face_index]

        point = sample_triangle_uniform(
            VERTICES[i],
            VERTICES[j],
            VERTICES[k],
        )

        points.append(point)

    return np.array(points), face_indices


def choose_red_pairs(
    points: np.ndarray,
    count: int,
) -> list[tuple[int, int]]:
    """Choose moderately local pairs on the visible surface."""

    candidates: list[tuple[float, int, int]] = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(
                points[i] - points[j]
            )

            if 0.40 <= distance <= 0.90:
                candidates.append((distance, i, j))

    rng.shuffle(candidates)

    selected: list[tuple[int, int]] = []
    used = np.zeros(len(points), dtype=int)

    for _, i, j in candidates:
        if used[i] >= 1 or used[j] >= 1:
            continue

        selected.append((i, j))
        used[i] += 1
        used[j] += 1

        if len(selected) == count:
            break

    if len(selected) < count:
        raise RuntimeError(
            f"Found only {len(selected)} suitable red pairs."
        )

    return selected


def choose_highlight_pair(
    points: np.ndarray,
    excluded: set[int],
) -> tuple[int, int]:
    """Choose a longer visible example pair."""

    candidates: list[tuple[float, int, int]] = []

    for i in range(len(points)):
        if i in excluded:
            continue

        for j in range(i + 1, len(points)):
            if j in excluded:
                continue

            distance = np.linalg.norm(
                points[i] - points[j]
            )

            if 1.0 <= distance <= 1.55:
                candidates.append(
                    (distance, i, j)
                )

    if not candidates:
        raise RuntimeError(
            "Could not find a suitable highlighted pair."
        )

    candidates.sort(reverse=True)

    _, i, j = candidates[0]
    return i, j


def main() -> None:
    points, _ = sample_octahedron_surface(N_POINTS)

    red_pairs = choose_red_pairs(
        points,
        N_RED_PAIRS,
    )

    excluded = {
        index
        for pair in red_pairs
        for index in pair
    }

    highlight_i, highlight_j = choose_highlight_pair(
        points,
        excluded,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(6.4, 4.8),
        facecolor=BACKGROUND,
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.set_facecolor(BACKGROUND)

    # Pale octahedron faces.
    face_vertices = [
        [
            VERTICES[i],
            VERTICES[j],
            VERTICES[k],
        ]
        for i, j, k in FACES
    ]

    surface = Poly3DCollection(
        face_vertices,
        facecolor=SURFACE,
        edgecolor=SURFACE_EDGE,
        linewidth=0.9,
        alpha=0.52,
    )

    ax.add_collection3d(surface)

    # Sampled entities.
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=48,
        facecolors="white",
        edgecolors="#202020",
        linewidths=1.15,
        depthshade=False,
        zorder=5,
    )

    # Sparse observed pairs.
    for i, j in red_pairs:
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            [points[i, 2], points[j, 2]],
            color=PAIR_RED,
            linewidth=2.2,
            alpha=0.99,
            solid_capstyle="round",
            zorder=3,
        )

    # Highlighted example pair.
    ax.plot(
        [
            points[highlight_i, 0],
            points[highlight_j, 0],
        ],
        [
            points[highlight_i, 1],
            points[highlight_j, 1],
        ],
        [
            points[highlight_i, 2],
            points[highlight_j, 2],
        ],
        color=HIGHLIGHT,
        linewidth=4.5,
        alpha=0.98,
        solid_capstyle="round",
        zorder=7,
    )

    midpoint = (
        points[highlight_i]
        + points[highlight_j]
    ) / 2.0

    label_position = midpoint + np.array(
        [0.05, 0.03, 0.30]
    )

    ax.text(
        label_position[0],
        label_position[1],
        label_position[2],
        r"$d(i,j)$",
        color="#174cff",
        fontsize=17,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=8,
    )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-1.15, 1.15)

    ax.view_init(
        elev=24,
        azim=-48,
    )

    ax.set_box_aspect(
        (1.0, 1.0, 1.0)
    )

    ax.set_axis_off()

    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.005,
        top=0.995,
    )

    fig.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.015,
        facecolor=BACKGROUND,
        transparent=False,
    )

    plt.close(fig)

    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()