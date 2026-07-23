# Provides a compact way to define a class whose main purpose is storing data.
from dataclasses import dataclass

# NumPy provides arrays, random-number generation, and numerical operations.
import numpy as np

# deque provides an efficient queue for breadth-first search.
from collections import deque


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
    #   grid:  [400, 2]
    #   sphere: [200, 3]
    coordinates: np.ndarray

    # Connections between points.
    #
    # Each tuple contains the integer indices of two connected points.
    # For example, (3, 4) means point 3 is connected to point 4.
    edges: list[tuple[int, int]]

    # Additional information about the world.
    #
    # Example:
    #   {
    #       "type": "grid",
    #       "width": 20,
    #       "height": 20,
    #   }
    meta: dict


def make_grid(width=10, height=10) -> World:
    """
    Create a rectangular two-dimensional grid.

    Every grid location becomes one point.

    Example for width=3 and height=2:

        p_0_1 -- p_1_1 -- p_2_1
          |        |        |
        p_0_0 -- p_1_0 -- p_2_0
    """

    # Store the name of every point.
    names = []

    # Store the true [x, y] coordinate of every point.
    coordinates = []

    # Store connections between horizontally and vertically adjacent points.
    edges = []

    # Visit every row.
    for y in range(height):
        # Visit every column in the current row.
        for x in range(width):
            # Give the point a name containing its coordinate.
            names.append(f"p_{x}_{y}")

            # Store its true two-dimensional position.
            coordinates.append([x, y])

    def idx(x, y):
        """
        Convert a two-dimensional grid coordinate into one flat integer index.

        Example for width=10:
            (0, 0) -> 0
            (1, 0) -> 1
            (0, 1) -> 10
        """

        # Each completed row contains `width` points.
        return y * width + x

    # Visit every point again to create graph edges.
    for y in range(height):
        for x in range(width):
            # Connect the current point to the point on its right.
            #
            # Only do this when the current point is not already
            # on the right edge of the grid.
            if x < width - 1:
                edges.append(
                    (
                        idx(x, y),
                        idx(x + 1, y),
                    )
                )

            # Connect the current point to the point above it.
            #
            # Only do this when the current point is not already
            # on the top edge of the grid.
            if y < height - 1:
                edges.append(
                    (
                        idx(x, y),
                        idx(x, y + 1),
                    )
                )

    # Convert the collected data into a World object.
    return World(
        names=names,
        coordinates=np.array(coordinates, dtype=np.float32),
        edges=edges,
        meta={
            "type": "grid",
            "width": width,
            "height": height,
        },
    )


def make_sphere(n=200, seed=0) -> World:
    """
    Create points distributed across the surface of a unit sphere.

    Each point has a three-dimensional coordinate [x, y, z].
    Every coordinate has length approximately 1.
    """

    # Create a reproducible random-number generator.
    rng = np.random.default_rng(seed)

    # Generate n random three-dimensional vectors.
    x = rng.normal(size=(n, 3))

    # Compute the length of every vector and divide by that length.
    #
    # This places every point on the surface of the unit sphere.
    x /= np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    # Assign a name to every sphere point.
    names = [f"s_{i}" for i in range(n)]

    # Sphere points currently have no graph edges.
    return World(
        names=names,
        coordinates=x.astype(np.float32),
        edges=[],
        meta={
            "type": "sphere",
            "n": n,
            "seed": seed,
        },
    )


def spherical_distance(a, b):
    """
    Return the great-circle distance between two unit-sphere points.

    Since the sphere radius is 1, the result is an angle in radians.
    """

    # The dot product of two unit vectors equals the cosine
    # of the angle between them.
    dot = np.dot(a, b)

    # Numerical errors can produce values slightly below -1
    # or slightly above 1, which arccos cannot accept.
    dot = np.clip(dot, -1.0, 1.0)

    # Convert cosine back into the angle between the vectors.
    return np.arccos(dot)


def make_random_graph(n=100, p=0.05, seed=0) -> World:
    """
    Create an undirected random graph.

    Every possible pair of nodes is connected independently
    with probability p.
    """

    # Create a reproducible random-number generator.
    rng = np.random.default_rng(seed)

    # Give every graph node a random two-dimensional coordinate.
    #
    # These coordinates are only for visualization.
    # Graph distance is computed from the edges, not these coordinates.
    coords = rng.normal(size=(n, 2))

    # Store every randomly created graph edge.
    edges = []

    # Visit every unique unordered pair of graph nodes.
    for i in range(n):
        for j in range(i + 1, n):
            # Create an edge with probability p.
            if rng.random() < p:
                edges.append((i, j))

    # Give every graph node a name.
    names = [f"g_{i}" for i in range(n)]

    # Return the completed graph world.
    return World(
        names=names,
        coordinates=coords.astype(np.float32),
        edges=edges,
        meta={
            "type": "graph",
            "n": n,
            "p": p,
            "seed": seed,
        },
    )


def shortest_path_distance(
    world: World,
    start: int,
    goal: int,
) -> int:
    """
    Compute the smallest number of graph edges needed to travel
    from `start` to `goal`.

    This uses breadth-first search.
    """

    # Create one empty neighbor list for every graph node.
    adj = [[] for _ in world.names]

    # Convert the edge list into an adjacency list.
    for a, b in world.edges:
        # The graph is undirected, so both directions are added.
        adj[a].append(b)
        adj[b].append(a)

    # Queue items contain:
    #   (current_node, distance_from_start)
    q = deque([(start, 0)])

    # Record nodes that have already been visited.
    seen = {start}

    # Continue until the queue is empty.
    while q:
        # Remove the oldest queued node.
        node, current_distance = q.popleft()

        # If this is the target node, return the discovered distance.
        if node == goal:
            return current_distance

        # Visit every neighbor of the current node.
        for neighbor in adj[node]:
            # Ignore neighbors that were already visited.
            if neighbor in seen:
                continue

            # Mark the neighbor as visited immediately so that it
            # cannot be added to the queue multiple times.
            seen.add(neighbor)

            # The neighbor is one edge farther away.
            q.append(
                (
                    neighbor,
                    current_distance + 1,
                )
            )

    # The graph is disconnected and no path exists.
    #
    # A very large integer is used here as an "infinite" distance.
    return 10**9


def haversine_km(a, b):
    """
    Compute distance along Earth's surface between two latitude-longitude points.

    Inputs:
        a = [latitude, longitude] in degrees
        b = [latitude, longitude] in degrees

    Output:
        distance in kilometers
    """

    # Convert the first coordinate from degrees to radians.
    lat1, lon1 = np.radians(a)

    # Convert the second coordinate from degrees to radians.
    lat2, lon2 = np.radians(b)

    # Compute latitude and longitude changes.
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula.
    h = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    # Convert the central angle into kilometers.
    #
    # 6371 is the approximate mean radius of Earth in kilometers.
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(h))