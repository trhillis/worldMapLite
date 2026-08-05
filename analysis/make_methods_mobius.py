"""Create a clean Möbius-strip illustration of sparse pairwise supervision."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

N_POINTS = 110
N_RED_PAIRS = 7
SEED = 4

RADIUS = 2.0
HALF_WIDTH = 0.55

OUTPUT = Path(
    "analysis_results/methods/mobius_training_pairs_clean.png"
)

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------
# Möbius geometry
# ---------------------------------------------------------------------

def sample_mobius_points(
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points in Möbius parameter coordinates."""

    theta = rng.uniform(0.0, 2.0 * np.pi, count)
    width = rng.uniform(-HALF_WIDTH, HALF_WIDTH, count)

    return theta, width


def mobius_to_3d(
    theta: np.ndarray,
    width: np.ndarray,
) -> np.ndarray:
    """Map Möbius parameter coordinates into 3D."""

    x = (
        RADIUS
        + width * np.cos(theta / 2.0)
    ) * np.cos(theta)

    y = (
        RADIUS
        + width * np.cos(theta / 2.0)
    ) * np.sin(theta)

    z = width * np.sin(theta / 2.0)

    return np.column_stack((x, y, z))


def wrapped_angle_difference(
    theta_a: float,
    theta_b: float,
) -> float:
    """Smallest absolute angular difference on a circle."""

    difference = (
        theta_a - theta_b + np.pi
    ) % (2.0 * np.pi) - np.pi

    return abs(difference)


def parameter_distance(
    theta_a: float,
    width_a: float,
    theta_b: float,
    width_b: float,
) -> float:
    """
    Approximate Möbius parameter-space distance.

    This is used only to choose visually suitable pairs.
    """

    candidates: list[float] = []

    # Direct comparison.
    angular = RADIUS * (theta_a - theta_b)
    transverse = width_a - width_b
    candidates.append(np.hypot(angular, transverse))

    # Compare across the Möbius seam, which flips width.
    for shift in (-2.0 * np.pi, 2.0 * np.pi):
        angular = RADIUS * (
            theta_a - (theta_b + shift)
        )
        transverse = width_a + width_b
        candidates.append(np.hypot(angular, transverse))

    return min(candidates)


# ---------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------

def choose_red_pairs(
    theta: np.ndarray,
    width: np.ndarray,
    count: int,
) -> list[tuple[int, int]]:
    """
    Choose moderately local pairs.

    Keeping the angular separation limited prevents most lines from
    cutting visually through the central hole of the strip.
    """

    candidates: list[tuple[float, int, int]] = []

    for i in range(len(theta)):
        for j in range(i + 1, len(theta)):
            angular_gap = wrapped_angle_difference(
                theta[i],
                theta[j],
            )

            distance = parameter_distance(
                theta[i],
                width[i],
                theta[j],
                width[j],
            )

            # Moderately separated pairs on roughly the same visible region.
            if (
                0.35 <= angular_gap <= 1.05
                and 0.65 <= distance <= 2.5
                and abs(width[i] - width[j]) <= 0.75
            ):
                candidates.append(
                    (distance, i, j)
                )

    rng.shuffle(candidates)

    selected: list[tuple[int, int]] = []
    usage = np.zeros(len(theta), dtype=int)

    for _, i, j in candidates:
        # Avoid repeatedly connecting the same point.
        if usage[i] >= 1 or usage[j] >= 1:
            continue

        selected.append((i, j))
        usage[i] += 1
        usage[j] += 1

        if len(selected) == count:
            break

    if len(selected) < count:
        raise RuntimeError(
            f"Found only {len(selected)} suitable red pairs; "
            f"requested {count}."
        )

    return selected


def choose_highlight_pair(
    theta: np.ndarray,
    width: np.ndarray,
    points: np.ndarray,
    excluded: set[int],
) -> tuple[int, int]:
    """
    Choose a longer, clearly visible pair for the blue example.

    The pair is longer than the ordinary red pairs but not so long that
    it spans the entire strip or cuts dramatically through the hole.
    """

    candidates: list[tuple[float, float, int, int]] = []

    for i in range(len(theta)):
        if i in excluded:
            continue

        for j in range(i + 1, len(theta)):
            if j in excluded:
                continue

            angular_gap = wrapped_angle_difference(
                theta[i],
                theta[j],
            )

            chord_length = np.linalg.norm(
                points[i] - points[j]
            )

            parameter_gap = parameter_distance(
                theta[i],
                width[i],
                theta[j],
                width[j],
            )

            if (
                1.05 <= angular_gap <= 1.65
                and 1.5 <= chord_length <= 3.0
                and 1.8 <= parameter_gap <= 3.8
            ):
                # Prefer a long visible chord.
                candidates.append(
                    (
                        chord_length,
                        parameter_gap,
                        i,
                        j,
                    )
                )

    if not candidates:
        raise RuntimeError(
            "Could not find a suitable highlighted pair."
        )

    candidates.sort(reverse=True)

    _, _, i, j = candidates[0]
    return i, j


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def main() -> None:
    theta, width = sample_mobius_points(N_POINTS)
    points = mobius_to_3d(theta, width)

    red_pairs = choose_red_pairs(
        theta,
        width,
        N_RED_PAIRS,
    )

    used_points = {
        index
        for pair in red_pairs
        for index in pair
    }

    highlight_pair = choose_highlight_pair(
        theta,
        width,
        points,
        used_points,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(6.4, 4.8),
        facecolor="white",
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.set_facecolor("white")

    # -----------------------------------------------------------------
    # Pale surface
    # -----------------------------------------------------------------

    surface_theta = np.linspace(
        0.0,
        2.0 * np.pi,
        180,
    )

    surface_width = np.linspace(
        -HALF_WIDTH,
        HALF_WIDTH,
        30,
    )

    theta_grid, width_grid = np.meshgrid(
        surface_theta,
        surface_width,
    )

    surface_points = mobius_to_3d(
        theta_grid.ravel(),
        width_grid.ravel(),
    )

    surface_x = surface_points[:, 0].reshape(
        theta_grid.shape
    )

    surface_y = surface_points[:, 1].reshape(
        theta_grid.shape
    )

    surface_z = surface_points[:, 2].reshape(
        theta_grid.shape
    )

    ax.plot_surface(
        surface_x,
        surface_y,
        surface_z,
        color="#b9ddf5",
        alpha=0.45,
        linewidth=0,
        antialiased=True,
        shade=False,
        zorder=1,
    )

    # -----------------------------------------------------------------
    # Sampled points
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Ordinary observed pairs
    # -----------------------------------------------------------------

    for i, j in red_pairs:
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            [points[i, 2], points[j, 2]],
            color="#d62728",
            linewidth=2.2,
            alpha=0.42,
            solid_capstyle="round",
            zorder=3,
        )

    # -----------------------------------------------------------------
    # Highlighted observed pair
    # -----------------------------------------------------------------

    example_i, example_j = highlight_pair

    ax.plot(
        [
            points[example_i, 0],
            points[example_j, 0],
        ],
        [
            points[example_i, 1],
            points[example_j, 1],
        ],
        [
            points[example_i, 2],
            points[example_j, 2],
        ],
        color="#174cff",
        linewidth=4.5,
        alpha=0.98,
        solid_capstyle="round",
        zorder=7,
    )

    midpoint = (
        points[example_i]
        + points[example_j]
    ) / 2.0

    # Move label away from the line and nearby points.
    label_position = midpoint + np.array(
        [0.12, 0.08, 0.40]
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

    # -----------------------------------------------------------------
    # Camera and export
    # -----------------------------------------------------------------

    ax.view_init(
        elev=27,
        azim=-60,
    )

    ax.set_box_aspect(
        (1.45, 1.35, 0.65)
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
        facecolor="white",
        transparent=False,
    )

    plt.close(fig)

    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()