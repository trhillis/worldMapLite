import numpy as np
from worlds import World, spherical_distance, haversine_km, shortest_path_distance

def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))

def distance(world: World, i: int, j: int):
    t = world.meta["type"]

    if t == "grid":
        return euclidean_distance(world.coordinates[i], world.coordinates[j])

    if t == "sphere":
        return float(spherical_distance(world.coordinates[i], world.coordinates[j]))

    if t == "earth":
        return float(haversine_km(world.coordinates[i], world.coordinates[j]))

    if t == "graph":
        return shortest_path_distance(world, i, j)

    raise ValueError(f"Unknown world type: {t}")

def nearest(world: World, i: int):
    best_j = None
    best_d = float("inf")

    for j in range(len(world.names)):
        if i == j:
            continue

        d = distance(world, i, j)
        if d < best_d:
            best_d = d
            best_j = j

    return best_j, best_d

def angle(world: World, a: int, b: int, c: int):
    x = world.coordinates[a] - world.coordinates[b]
    y = world.coordinates[c] - world.coordinates[b]

    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0

    cos = np.clip(np.dot(x, y) / denom, -1.0, 1.0)
    return float(np.arccos(cos))