# NumPy is used for vector operations and angle calculations.
import numpy as np

# Import the shared World type and world-specific distance functions.
from src.worlds import (
    World,
    spherical_distance,
    haversine_km,
    shortest_path_distance,
)


def euclidean_distance(a, b):
    """
    Compute straight-line Euclidean distance between two coordinate vectors.
    """

    # Subtract one coordinate from the other.
    difference = a - b

    # Compute the length of the difference vector.
    return float(np.linalg.norm(difference))


def distance(world: World, i: int, j: int):
    """
    Compute the appropriate ground-truth distance between points i and j.

    The meaning of distance depends on the world type.
    """

    # Read the world type from metadata.
    world_type = world.meta["type"]

    if world_type == "grid":
        # IMPORTANT:
        # This uses straight-line Euclidean distance.
        #
        # Examples:
        #   horizontal neighbor: 1
        #   diagonal neighbor: sqrt(2)
        #
        # This is not graph distance and not Manhattan distance.
        return euclidean_distance(
            world.coordinates[i],
            world.coordinates[j],
        )

    if world_type == "sphere":
        # Use the angle between points on the unit sphere.
        return float(
            spherical_distance(
                world.coordinates[i],
                world.coordinates[j],
            )
        )

    if world_type == "earth":
        # Use surface distance in kilometers.
        return float(
            haversine_km(
                world.coordinates[i],
                world.coordinates[j],
            )
        )

    if world_type == "graph":
        # Use the number of graph edges in the shortest path.
        return shortest_path_distance(world, i, j)

    # Reject unsupported world types.
    raise ValueError(f"Unknown world type: {world_type}")


def nearest(world: World, i: int):
    """
    Find one nearest point to point i.

    If multiple points are tied, this returns the first one found.
    """

    # Store the best point found so far.
    best_j = None

    # Begin with an infinite best distance.
    best_distance = float("inf")

    # Compare point i against every point in the world.
    for j in range(len(world.names)):
        # A point cannot be its own nearest neighbor.
        if i == j:
            continue

        # Compute the true world distance.
        current_distance = distance(world, i, j)

        # Replace the current best point if this one is closer.
        if current_distance < best_distance:
            best_distance = current_distance
            best_j = j

    # Return both the point index and its distance.
    return best_j, best_distance


def angle(world: World, a: int, b: int, c: int):
    """
    Compute the angle A-B-C.

    Point b is the vertex of the angle.

        a
         \
          b --- c
    """

    # Create the vector pointing from b toward a.
    vector_ba = (
        world.coordinates[a]
        - world.coordinates[b]
    )

    # Create the vector pointing from b toward c.
    vector_bc = (
        world.coordinates[c]
        - world.coordinates[b]
    )

    # The denominator is the product of the vector lengths.
    denominator = (
        np.linalg.norm(vector_ba)
        * np.linalg.norm(vector_bc)
    )

    # Avoid division by zero if one of the vectors has zero length.
    if denominator == 0:
        return 0.0

    # Compute the cosine of the angle using the dot-product formula.
    cosine = (
        np.dot(vector_ba, vector_bc)
        / denominator
    )

    # Protect arccos against small floating-point errors.
    cosine = np.clip(cosine, -1.0, 1.0)

    # Convert the cosine into an angle in radians.
    return float(np.arccos(cosine))