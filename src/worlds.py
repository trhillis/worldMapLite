# Provides a compact way to define a class whose main purpose is storing data.
from dataclasses import dataclass

# os is used to create/check the geodesic-distance cache directory.
import os

# NumPy provides arrays, random-number generation, and numerical operations.
import numpy as np


# Define a general container for every type of world.
@dataclass
class World:
    # Human-readable name for every entity or point.
    names: list[str]

    # Numeric coordinates for every point.
    #
    # Shape:
    #   [number_of_points, coordinate_dimensions]
    #
    # Examples:
    #   manifold (octahedron): [n, 3]  (ambient embedding)
    #   manifold (flat torus): [n, 4]  (ambient embedding)
    coordinates: np.ndarray

    # Connections between points.
    #
    # Each tuple contains the integer indices of two connected points.
    # Manifold worlds have no graph edges (geodesic distance is precomputed
    # densely instead), so this is always an empty list for them.
    edges: list[tuple[int, int]]

    # Additional information about the world.
    #
    # Example:
    #   {
    #       "type": "manifold",
    #       "manifold_name": "octahedron",
    #       "n": 800,
    #       "seed": 0,
    #       "distance_matrix": <n, n> geodesic distances,
    #   }
    meta: dict


def make_manifold_world(
    manifold,
    n=800,
    seed=0,
    cache_dir="data/manifold_cache",
) -> World:
    """
    Sample n fixed points from a manifolds.Manifold and precompute every
    pairwise geodesic distance once.

    The points are the manifold equivalent of Park et al.'s fixed set of
    cities: a random but reproducible (seeded) set of entities whose indices
    become the embedding-table indices the model is trained on. Geodesic
    distance on these manifolds is a combinatorial search (see
    manifolds/polyhedra.py), so the full n-by-n distance matrix is computed
    once here and cached to disk - training and analysis both then treat
    ground-truth distance as an O(1) lookup rather than recomputing it.
    """

    # Reproducible sampling, independent of every other random source.
    rng = np.random.default_rng(seed)

    # Chart-coordinate points on the manifold, shape (n, *point_shape).
    chart_points = manifold.sample(n, rng=rng)

    # Ambient coordinates, used as the World's true "ground truth" geometry
    # for plotting and for probing the model's learned representations.
    ambient = manifold.embed(chart_points)

    # Cache key: which manifold, how many points, which seed.
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"{manifold.name}_n{n}_seed{seed}.npz",
    )

    if os.path.exists(cache_path):
        # Reuse the previously computed geodesic distances.
        cached = np.load(cache_path)
        distance_matrix = cached["distance_matrix"]
    else:
        # Every pairwise geodesic distance - this is the expensive,
        # combinatorial part, so it only happens once per
        # (manifold, n, seed) combination.
        distance_matrix = manifold.distance_matrix(chart_points)

        np.savez(
            cache_path,
            chart_points=chart_points,
            ambient=ambient,
            distance_matrix=distance_matrix,
        )

    # Give every sampled point a name and wrap everything into a World.
    names = [f"m_{i}" for i in range(n)]

    return World(
        names=names,
        coordinates=ambient.astype(np.float32),
        edges=[],
        meta={
            "type": "manifold",
            "manifold_name": manifold.name,
            "n": n,
            "seed": seed,
            "chart_points": chart_points,
            "distance_matrix": distance_matrix,
        },
    )
