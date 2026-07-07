import numpy as np
from tasks import distance, nearest, angle


# This function generates a list of examples for the distance task, where each example consists of a pair of points and their corresponding distance.
# We need this function to create training and evaluation datasets for the distance model, 
# allowing it to learn the relationship between point pairs and their distances in various worlds (grid, sphere, graph, etc)
def make_distance_examples(world, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    examples = []

    for _ in range(n):
        i, j = rng.choice(len(world.names), size=2, replace=False)
        examples.append({
            "input": f"distance {world.names[i]} {world.names[j]}",
            "answer": distance(world, i, j),
            "indices": (int(i), int(j)),
            "task": "distance",
        })

    return examples

# This function generates a list of examples for the nearest neighbor task, where each example consists of a point and its nearest neighbor.
# We need this function to create training and evaluation datasets for the nearest neighbor model,
# allowing it to learn the relationship between a point and its closest point in various worlds (grid, sphere, graph, etc).
def make_nearest_examples(world, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    examples = []

    for _ in range(n):
        i = int(rng.integers(len(world.names)))
        j, d = nearest(world, i)

        examples.append({
            "input": f"nearest {world.names[i]}",
            "answer": world.names[j],
            "indices": (i, j),
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