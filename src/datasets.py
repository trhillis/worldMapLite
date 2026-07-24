# NumPy provides random sampling and arrays.
import numpy as np

# Import ground-truth task functions.
from src.tasks import distance, angle


def build_grid_nearest_cache(world):
    """
    Precompute every immediate grid neighbor for every grid point.

    This function only supports grid worlds.
    """

    # Reject other world types instead of silently producing bad results.
    if world.meta["type"] != "grid":
        raise ValueError(
            "build_grid_nearest_cache only supports grid worlds"
        )

    # Read the grid dimensions.
    width = world.meta["width"]
    height = world.meta["height"]

    def idx(x, y):
        # Convert [x, y] into a flat point index.
        return y * width + x

    # Map:
    #   point_index -> list of immediate neighbor indices
    cache = {}

    # Visit every grid coordinate.
    for y in range(height):
        for x in range(width):
            # Convert the current coordinate into a point index.
            current_index = idx(x, y)

            # Store the current point's valid neighbors.
            neighbors = []

            # Add the point on the left.
            if x > 0:
                neighbors.append(idx(x - 1, y))

            # Add the point on the right.
            if x < width - 1:
                neighbors.append(idx(x + 1, y))

            # Add the point below.
            if y > 0:
                neighbors.append(idx(x, y - 1))

            # Add the point above.
            if y < height - 1:
                neighbors.append(idx(x, y + 1))

            # Save this point's complete nearest-neighbor list.
            cache[current_index] = neighbors

    return cache


def build_manifold_nearest_cache(
    world,
):
    if world.meta["type"] != "manifold":
        raise ValueError(
            "Expected a manifold world"
        )

    if world.manifold is None:
        raise ValueError(
            "Manifold world is missing its manifold object"
        )

    distance_matrix = (
        world.manifold.distance_matrix(
            world.coordinates
        )
    )

    distance_matrix = np.asarray(
        distance_matrix,
        dtype=np.float64,
    )

    np.fill_diagonal(
        distance_matrix,
        np.inf,
    )

    nearest_cache = {}
    negative_cache = {}

    all_indices = np.arange(
        len(world.names),
        dtype=np.int64,
    )

    for i in range(
        len(world.names)
    ):
        minimum_distance = (
            distance_matrix[i].min()
        )

        nearest = np.flatnonzero(
            np.isclose(
                distance_matrix[i],
                minimum_distance,
                atol=1e-8,
                rtol=0.0,
            )
        ).astype(np.int64)

        excluded = np.zeros(
            len(world.names),
            dtype=bool,
        )

        excluded[i] = True
        excluded[nearest] = True

        nearest_cache[i] = (
            nearest.tolist()
        )

        negative_cache[i] = (
            all_indices[~excluded]
        )

    return (
        nearest_cache,
        negative_cache,
    )

def build_nearest_and_negative_cache(world):
    """
    Precompute positive and negative nearest-task candidates.

    Positive candidates:
        immediate grid neighbors

    Negative candidates:
        every point except the query point and its immediate neighbors
    """

    # Build the positive-neighbor cache.
    nearest_cache = build_grid_nearest_cache(world)

    # Create a set containing every point index.
    all_points = set(range(len(world.names)))

    # Map:
    #   point_index -> NumPy array of valid negative point indices
    negative_cache = {}

    # Process every point and its positive neighbors.
    for point_index, positive_neighbors in nearest_cache.items():
        # Exclude the point itself and every valid positive neighbor.
        excluded = {
            point_index,
            *positive_neighbors,
        }

        # Everything remaining is a valid negative example.
        valid_negatives = all_points - excluded

        # Sorting makes the cache deterministic.
        negative_cache[point_index] = np.array(
            sorted(valid_negatives),
            dtype=np.int64,
        )

    return nearest_cache, negative_cache


def distance_scale(world):
    """
    Return a value used to normalize distance targets.

    The goal is to make distance targets approximately fall between 0 and 1.
    """

    # Read the world type.
    world_type = world.meta["type"]

    if world_type == "grid":
        width = world.meta["width"]
        height = world.meta["height"]

        # IMPORTANT:
        # Your distance function uses Euclidean distance.
        #
        # Therefore the maximum grid distance is the diagonal:
        #
        #   sqrt((width - 1)^2 + (height - 1)^2)
        #
        # Your previous code used:
        #
        #   (width - 1) + (height - 1)
        #
        # which is the maximum Manhattan distance and does not match
        # your Euclidean distance task.
        return float(
            np.sqrt(
                (width - 1) ** 2
                + (height - 1) ** 2
            )
        )

    if world_type == "sphere":
        # The largest possible angle between two sphere points is pi.
        return float(np.pi)

    if world_type == "earth":
        # Half of Earth's circumference is approximately pi * radius.
        return float(np.pi * 6371.0)

    if world_type == "graph":
        # Graph scaling needs a separately computed graph diameter.
        raise ValueError(
            "Graph distance scaling is not implemented"
        )

    if world_type == "manifold":
        return float(
            world.meta["diameter"]
        )

    raise ValueError(
        f"Unknown world type: {world_type}"
    )


def make_distance_examples(
    world,
    n=1000,
    seed=0,
):
    """
    Generate supervised examples for the distance task.

    Each example contains:
        point i
        point j
        normalized distance between i and j
    """

    # Create a reproducible random-number generator.
    rng = np.random.default_rng(seed)

    # Store generated examples.
    examples = []

    # Compute the normalization scale once.
    scale = distance_scale(world)

    # Generate n examples.
    for _ in range(n):
        # Select two different point indices.
        i, j = rng.choice(
            len(world.names),
            size=2,
            replace=False,
        )

        # Compute the true unnormalized distance.
        raw_distance = float(
            distance(
                world,
                int(i),
                int(j),
            )
        )

        # Add the training example.
        examples.append({
            # Input point indices used by the embedding layer.
            "indices": (
                int(i),
                int(j),
            ),

            # Normalized training target.
            "answer": raw_distance / scale,

            # Original distance retained for debugging or reporting.
            "raw_answer": raw_distance,

            # Identifies which task produced this example.
            "task": "distance",
        })

    return examples


def all_nearest(
    world,
    i,
    atol=1e-8,
):
    """
    Find every point tied for the smallest nonzero distance from point i.

    This generic implementation works for continuous worlds, but it is slower
    than the grid-specific cache.
    """

    # Store each candidate point and its distance.
    candidates = []

    # Compare point i with every other point.
    for j in range(len(world.names)):
        if j == i:
            continue

        candidates.append(
            (
                j,
                float(distance(world, i, j)),
            )
        )

    # Find the smallest candidate distance.
    minimum_distance = min(
        candidate_distance
        for _, candidate_distance in candidates
    )

    # Return every point whose distance is approximately equal
    # to the minimum distance.
    return [
        j
        for j, candidate_distance in candidates
        if np.isclose(
            candidate_distance,
            minimum_distance,
            atol=atol,
            rtol=0.0,
        )
    ]


def make_nearest_examples(
    world,
    n=1000,
    seed=0,
    nearest_cache=None,
    negative_cache=None,
):
    """
    Generate balanced binary examples for the nearest-neighbor task.

    Every loop creates:
        one positive pair
        one negative pair

    Therefore n=50,000 creates 100,000 total examples.
    """

    # Create a reproducible random-number generator.
    rng = np.random.default_rng(seed)

    # Store the examples.
    examples = []

    # Build caches if they were not supplied.
    #
    # IMPORTANT:
    # This cache-building function currently only supports grids.
    if nearest_cache is None or negative_cache is None:
        if world.meta["type"] == "grid":
            nearest_cache, negative_cache = (
                build_nearest_and_negative_cache(
                    world
                )
            )

        elif world.meta["type"] == "manifold":
            nearest_cache, negative_cache = (
                build_manifold_nearest_cache(
                    world
                )
            )

        else:
            raise ValueError(
                "Nearest cache is not implemented for "
                f"{world.meta['type']}"
            )

    # Generate n positive-negative pairs.
    for _ in range(n):
        # Select a random query point.
        i = int(
            rng.integers(len(world.names))
        )

        # Randomly select one true immediate neighbor.
        positive_j = int(
            rng.choice(nearest_cache[i])
        )

        # Randomly select one point that is not an immediate neighbor.
        negative_j = int(
            rng.choice(negative_cache[i])
        )

        # Add one positive classification example.
        examples.append({
            "indices": (i, positive_j),
            "answer": 1.0,
            "task": "nearest",
        })

        # Add one negative classification example.
        examples.append({
            "indices": (i, negative_j),
            "answer": 0.0,
            "task": "nearest",
        })

    return examples


def make_angle_examples(
    world,
    n=1000,
    seed=0,
):
    """
    Generate examples for the angle task.

    Each example contains three point indices:
        a, b, c

    The target is the angle at b.
    """

    # Create a reproducible random-number generator.
    rng = np.random.default_rng(seed)

    # Store generated examples.
    examples = []

    # Generate n triplets.
    for _ in range(n):
        # Choose three different points.
        a, b, c = rng.choice(
            len(world.names),
            size=3,
            replace=False,
        )

        # Add the angle example.
        examples.append({
            "indices": (
                int(a),
                int(b),
                int(c),
            ),
            "answer": angle(
                world,
                int(a),
                int(b),
                int(c),
            ),
            "task": "angle",
        })

    return examples