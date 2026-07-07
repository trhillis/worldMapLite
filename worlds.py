# Generate earth/grid/sphere graph

from dataclasses import dataclass
import numpy as np
from collections import deque

@dataclass
class World:
    names: list[str]
    coordinates: np.ndarray # [n, dim]
    edges: list[tuple[int, int]] # graph structures
    meta: dict


# Grid world
# This is a simple 2D grid world where each point is connected to its immediate neighbors (up, down, left, right).
# The coordinates of each point are its (x, y) position in the grid. The edges represent the connections between neighboring points.
# We need this world to test the distance model, as it provides a simple and structured environment where distances can be easily computed and verified.
def make_grid(width = 10, height = 10):
    names = []
    coordinates = []
    edges = []

    for y in range(height):
        for x in range(width):
            names.append(f"p_{x}_{y}")
            coordinates.append([x, y])
    
    def idx (x, y):
        return y * width + x
    
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                edges.append((idx(x, y), idx(x + 1, y)))
            if y < height - 1:
                edges.append((idx(x, y), idx(x, y + 1)))

    return World(
        names = names,
        coordinates = np.array(coordinates),
        edges = edges,
        meta = {"type": "grid", "width": width, "height": height}
    )


# Sphere world
# This world represents points uniformly distributed on the surface of a sphere.
# The coordinates are 3D Cartesian coordinates (x, y, z) that lie on the unit sphere.
# There are no edges in this world, as it is not a graph structure.
# We need this world to test the distance model in a more complex and continuous space, where distances are computed as angles between points on the sphere.
def make_sphere(n=200, seed=0) -> World:
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(n,3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    names = [f"s_{i}" for i in range(n)]

    return World(
        names = names,
        coordinates = x,
        edges = [],
        meta = {"type": "sphere", "n": n, "seed": seed}
    )


# This function computes the spherical distance between two points on the unit sphere using the arccosine of their dot product.
# The result is the angle in radians between the two points, which corresponds to the great-circle distance on the sphere's surface.
# We need this function to provide a ground truth distance metric for the sphere world, allowing us to evaluate the performance of the distance model in this continuous space.
def spherical_distance(a, b):
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.arccos(dot)


# This function generates a random graph with n nodes and edges created with probability p.
# We need this world to test the distance model in a more general graph structure, where distances are computed as the shortest path between nodes.
def make_random_graph(n=100, p=0.05, seed=0) -> World:
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(n, 2))

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))

    names = [f"g_{i}" for i in range(n)]

    return World(
        names=names,
        coordinates=coords,
        edges=edges,
        meta={"type": "graph", "n": n, "p": p},
    )

    
# This function computes the shortest path distance between two nodes in a graph using breadth-first search (BFS).
# We need this function to provide a ground truth distance metric for the random graph world,
# allowing us to evaluate the performance of the distance model in this discrete space.
def shortest_path_distance(world: World, start: int, goal: int) -> int:
    adj = [[] for _ in world.names]
    for a, b in world.edges:
        adj[a].append(b)
        adj[b].append(a)

    q = deque([(start, 0)])
    seen = {start}

    while q:
        node, d = q.popleft()
        if node == goal:
            return d
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))

    return 10**9

# This function generates a fake Earth-like world with random latitude and longitude coordinates for n cities.
# We need this world to test the distance model in a geospatial context,
# where distances are computed using the Haversine formula to account for the curvature of the Earth.
def make_fake_earth(n=200, seed=0) -> World:
    rng = np.random.default_rng(seed)

    lat = rng.uniform(-70, 70, size=n)
    lon = rng.uniform(-180, 180, size=n)

    coords = np.stack([lat, lon], axis=1)
    names = [f"city_{i}" for i in range(n)]

    return World(
        names=names,
        coordinates=coords,
        edges=[],
        meta={"type": "earth", "coords": "lat_lon"},
    )

# This function computes the Haversine distance between two points given their latitude and longitude coordinates.
# We need this function to provide a ground truth distance metric for the fake Earth world,
#  allowing us to evaluate the performance of the distance model in a geospatial context.
def haversine_km(a, b):
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    h = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 6371 * 2 * np.arcsin(np.sqrt(h))