# NumPy is used to read face indices out of chart points.
import numpy as np

# Import the shared World type.
from src.worlds import World


def distance(world: World, i: int, j: int):
    """
    Compute the ground-truth geodesic distance between points i and j.

    Manifold worlds (src.worlds.make_manifold_world) precompute the full
    pairwise geodesic distance matrix once, so this is a lookup, not a call
    into manifolds.Manifold.distance().
    """

    # Read the world type from metadata.
    world_type = world.meta["type"]

    if world_type == "manifold":
        return float(world.meta["distance_matrix"][i, j])

    # Reject unsupported world types.
    raise ValueError(f"Unknown world type: {world_type}")


def same_triangle(world: World, i: int, j: int) -> bool:
    """
    Ground truth: do points i and j lie on the same triangular face?

    Polyhedron-specific: relies on the (face, b0, b1, b2) chart-point
    convention used by manifolds.polyhedra.PolyhedralSurface (octahedron()/
    icosahedron()), the same column src.datasets.points_on_faces reads -
    not meaningful for FlatTorus or FlatMobiusStrip worlds.
    """

    world_type = world.meta["type"]

    if world_type == "manifold":
        chart_points = world.meta["chart_points"]
        faces = np.round(chart_points[:, 0]).astype(np.int64)
        return bool(faces[i] == faces[j])

    raise ValueError(f"Unknown world type: {world_type}")
