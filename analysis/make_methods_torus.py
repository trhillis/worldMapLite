"""Create a clean flat-torus illustration of sparse pairwise supervision."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N_POINTS = 120
N_RED_PAIRS = 7
SEED = 7

MAJOR_RADIUS = 2.0
MINOR_RADIUS = 0.70

OUTPUT = Path(
    "analysis_results/methods/torus_training_pairs_clean.png"
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

def sample_torus_points(
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample uniformly in torus parameter coordinates."""

    u = rng.uniform(0.0, 2.0 * np.pi, count)
    v = rng.uniform(0.0, 2.0 * np.pi, count)

    return u, v


def torus_to_3d(
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Embed a flat parameter torus into 3D for visualization."""

    x = (
        MAJOR_RADIUS
        + MINOR_RADIUS * np.cos(v)
    ) * np.cos(u)

    y = (
        MAJOR_RADIUS
        + MINOR_RADIUS * np.cos(v)
    ) * np.sin(u)

    z = MINOR_RADIUS * np.sin(v)

    return np.column_stack((x, y, z))


def wrapped_difference(
    a: float,
    b: float,
) -> float:
    """Smallest absolute angular difference."""

    return abs(
        (a - b + np.pi) % (2.0 * np.pi) - np.pi
    )


def flat_torus_distance(
    u_a: float,
    v_a: float,
    u_b: float,
    v_b: float,
) -> float:
    """
    Flat-torus parameter distance.

    Used only to choose visually suitable example pairs.
    """

    du = wrapped_difference(u_a, u_b)
    dv = wrapped_difference(v_a, v_b)

    return np.hypot(du, dv)


def choose_red_pairs(
    u: np.ndarray,
    v: np.ndarray,
    count: int,
) -> list[tuple[int, int]]:
    """Choose moderately local sparse pairs."""

    candidates: list[tuple[float, int, int]] = []

    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            distance = flat_torus_distance(
                u[i],
                v[i],
                u[j],
                v[j],
            )

            if 0.45 <= distance <= 1.20:
                candidates.append((distance, i, j))

    rng.shuffle(candidates)

    selected: list[tuple[int, int]] = []
    used = np.zeros(len(u), dtype=int)

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
    u: np.ndarray,
    v: np.ndarray,
    points: np.ndarray,
    excluded: set[int],
) -> tuple[int, int]:
    """Choose a longer, clearly visible example pair."""

    candidates: list[tuple[float, int, int]] = []

    for i in range(len(u)):
        if i in excluded:
            continue

        for j in range(i + 1, len(u)):
            if j in excluded:
                continue

            parameter_distance = flat_torus_distance(
                u[i],
                v[i],
                u[j],
                v[j],
            )

            chord_length = np.linalg.norm(
                points[i] - points[j]
            )

            if (
                1.25 <= parameter_distance <= 2.0
                and 1.3 <= chord_length <= 2.7
            ):
                candidates.append(
                    (chord_length, i, j)
                )

    if not candidates:
        raise RuntimeError(
            "Could not find a suitable highlighted pair."
        )

    candidates.sort(reverse=True)

    _, i, j = candidates[0]
    return i, j


def main() -> None:
    u, v = sample_torus_points(N_POINTS)
    points = torus_to_3d(u, v)

    red_pairs = choose_red_pairs(
        u,
        v,
        N_RED_PAIRS,
    )

    excluded = {
        index
        for pair in red_pairs
        for index in pair
    }

    highlight_i, highlight_j = choose_highlight_pair(
        u,
        v,
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

    # Pale torus surface.
    surface_u = np.linspace(
        0.0,
        2.0 * np.pi,
        160,
    )

    surface_v = np.linspace(
        0.0,
        2.0 * np.pi,
        48,
    )

    u_grid, v_grid = np.meshgrid(
        surface_u,
        surface_v,
    )

    surface_points = torus_to_3d(
        u_grid.ravel(),
        v_grid.ravel(),
    )

    surface_x = surface_points[:, 0].reshape(
        u_grid.shape
    )

    surface_y = surface_points[:, 1].reshape(
        u_grid.shape
    )

    surface_z = surface_points[:, 2].reshape(
        u_grid.shape
    )

    ax.plot_surface(
        surface_x,
        surface_y,
        surface_z,
        color=SURFACE,
        alpha=0.35,
        linewidth=0,
        antialiased=True,
        shade=False,
        zorder=1,
    )

    # Sampled entities.
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=48,
        facecolors=POINT_FILL,
        edgecolors=POINT_EDGE,
        linewidths=1.15,
        depthshade=False,
        zorder=5,
    )

    # Sparse observed training pairs.
    for i, j in red_pairs:
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            [points[i, 2], points[j, 2]],
            color=PAIR_RED,
            linewidth=2.2,
            alpha=1,
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
        [0.05, 0.05, 0.42]
    )

    ax.text(
        label_position[0],
        label_position[1],
        label_position[2],
        r"$d(i,j)$",
        color=TEXT,
        fontsize=17,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=8,
    )

    ax.view_init(
        elev=27,
        azim=-55,
    )

    ax.set_box_aspect(
        (1.45, 1.35, 0.72)
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