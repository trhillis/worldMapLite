import numpy as np
from tasks import distance, nearest, angle

def build_nearest_and_negative_cache(world):
    nearest_cache = build_grid_nearest_cache(world)
    all_points = set(range(len(world.names)))

    negative_cache = {}

    for i, positives in nearest_cache.items():
        excluded = {i, *positives}
        negative_cache[i] = np.array(
            sorted(all_points - excluded),
            dtype=np.int64,
        )

    return nearest_cache, negative_cache

def build_grid_nearest_cache(world):
    width = world.meta["width"]
    height = world.meta["height"]

    def idx(x, y):
        return y * width + x

    cache = {}

    for y in range(height):
        for x in range(width):
            i = idx(x, y)
            neighbors = []

            if x > 0:
                neighbors.append(idx(x - 1, y))

            if x < width - 1:
                neighbors.append(idx(x + 1, y))

            if y > 0:
                neighbors.append(idx(x, y - 1))

            if y < height - 1:
                neighbors.append(idx(x, y + 1))

            cache[i] = neighbors

    return cache

def distance_scale(world):
    world_type = world.meta["type"]

    if world_type == "grid":
        return float(
            (world.meta["width"] - 1)
            + (world.meta["height"] - 1)
        )

    if world_type == "sphere":
        return float(np.pi)

    if world_type == "earth":
        return float(np.pi * 6371.0)

    # For graphs, compute the maximum finite shortest-path distance
    # or estimate it from sampled pairs.
    raise ValueError(f"No scale defined for {world_type}")



# This function generates a list of examples for the distance task, where each example consists of a pair of points and their corresponding distance.
# We need this function to create training and evaluation datasets for the distance model, 
# allowing it to learn the relationship between point pairs and their distances in various worlds (grid, sphere, graph, etc)
def make_distance_examples(world, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    examples = []
    scale = distance_scale(world)

    for _ in range(n):
        i, j = rng.choice(len(world.names), size=2, replace=False)
        raw_distance = float(distance(world, i ,j))

        examples.append({
            "indices": (int(i), int(j)),
            "answer": raw_distance / scale,
            "raw_answer": raw_distance,
            "task": "distance",
        })
    
    return examples

def all_nearest(world, i, atol=1e-8):
    """Return every point tied for minimum nonzero distance from i"""

    candidates = []

    for j in range(len(world.names)):
        if j == i:
            continue
        candidates.append((j, float(distance(world, i, j))))

    min_distance = min(d for _, d in candidates)

    return [
        j for j, d in candidates
        if np.isclose(d, min_distance, atol=atol, rtol=0.0)
    ]

# This function generates a list of examples for the nearest neighbor task, where each example consists of a point and its nearest neighbor.
# We need this function to create training and evaluation datasets for the nearest neighbor model,
# allowing it to learn the relationship between a point and its closest point in various worlds (grid, sphere, graph, etc).
def make_nearest_examples(world, n=1000, seed=0, nearest_cache=None, negative_cache=None):
    rng = np.random.default_rng(seed)
    examples = []

    if nearest_cache is None or negative_cache is None:
        nearest_cache, negative_cache = (
            build_nearest_and_negative_cache(world)
        )

    for _ in range(n):
        i = int(rng.integers(len(world.names)))

        positive_j = int(rng.choice(nearest_cache[i]))
        negative_j = int(rng.choice(negative_cache[i]))

        examples.append({
            "indices": (i, positive_j),
            "answer": 1.0,
            "task": "nearest",
        })

        examples.append({
            "indices": (i, negative_j),
            "answer": 0.0,
            "task": "nearest",
        })

    return examples

# This function generates a list of examples for the angle task, where each example consists of 
# three points and the angle formed at the second point by the lines connecting it to the first and third points.
# We need this function to create training and evaluation datasets for the angle model,
# allowing it to learn the relationship between triplets of points and the angles they form in various worlds (grid, sphere, graph, etc).
def make_angle_examples(world, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    examples = []

    for _ in range(n):
        a, b, c = rng.choice(len(world.names), size=3, replace=False)

        examples.append({
            "input": f"angle {world.names[a]} {world.names[b]} {world.names[c]}",
            "answer": angle(world, a, b, c),
            "indices": (int(a), int(b), int(c)),
            "task": "angle",
        })

    return examples