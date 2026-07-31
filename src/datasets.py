# Python's random module is seeded for reproducibility.
import random

# NumPy provides random sampling and arrays.
import numpy as np

# PyTorch provides tensors used by PairDataset.
import torch

# Dataset defines a PyTorch-compatible dataset.
from torch.utils.data import Dataset

# Import ground-truth task functions.
from src.tasks import distance, same_triangle


def distance_scale(world):
    """
    Return a value used to normalize distance targets.

    The goal is to make distance targets approximately fall between 0 and 1.
    """

    # Read the world type.
    world_type = world.meta["type"]

    if world_type == "manifold":
        # The largest geodesic distance in the precomputed pairwise matrix.
        return float(world.meta["distance_matrix"].max())

    raise ValueError(
        f"Unknown world type: {world_type}"
    )


def _build_distance_example(world, i, j, scale):
    """
    Build one supervised distance-task example for point pair (i, j).

    Shared by make_distance_examples and make_distance_examples_from_pairs
    so both sampling strategies produce identically-shaped examples.
    """

    # Compute the true unnormalized distance.
    raw_distance = float(
        distance(
            world,
            int(i),
            int(j),
        )
    )

    return {
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
    }


def split_points(
    num_points,
    n_holdout,
    seed,
    candidate_points=None,
):
    """
    Deterministically partition point indices 0..num_points-1 into a
    train pool and a holdout pool of size n_holdout.

    Reproducible from (num_points, n_holdout, seed, candidate_points)
    alone, mirroring split_pairs. Used to carve out entire points (not
    just pairs) that no training pair may touch - see
    make_holdout_point_pairs for how their probe/eval pairs are then
    built, and src/holdout_probe.py for how their embeddings are
    recovered afterward from a handful of probe distances.

    When candidate_points is given (an int array of allowed indices,
    e.g. from points_on_faces), the n_holdout holdout points are drawn
    from within that subset only - every other index (including any
    point not in candidate_points) goes to train_points. This is what
    lets holdout points be restricted to one region of the manifold
    (e.g. one octahedron face) instead of scattered uniformly at random.
    """

    pool = (
        np.arange(num_points, dtype=np.int64)
        if candidate_points is None
        else np.asarray(candidate_points, dtype=np.int64)
    )

    if n_holdout > len(pool):
        raise ValueError(
            f"n_holdout ({n_holdout}) must not exceed the number of "
            f"candidate points ({len(pool)})"
        )

    rng = np.random.default_rng(seed)

    # A random ordering of the candidate pool; the first slice becomes
    # the holdout set.
    shuffled_pool = pool[rng.permutation(len(pool))]

    holdout_points = np.sort(shuffled_pool[:n_holdout])

    all_points = np.arange(num_points, dtype=np.int64)
    train_points = np.sort(
        all_points[~np.isin(all_points, holdout_points)]
    )

    return train_points, holdout_points


def points_on_faces(world, face_indices):
    """
    Return the indices of world's points whose polyhedral chart face is
    in face_indices.

    Polyhedron-specific: relies on the (face, b0, b1, b2) chart-point
    convention used by manifolds.polyhedra.PolyhedralSurface (and hence
    octahedron()/icosahedron()) - not meaningful for FlatTorus or
    FlatMobiusStrip worlds. Intended use: restrict split_points'
    candidate_points to "one triangle side" of the octahedron, e.g.
    points_on_faces(world, [0]).

    Since World points are sampled uniformly over the whole surface, a
    single face only gets a fraction of the n total points (about
    n / n_faces on average) - callers asking for more holdout points
    than a face actually has will hit split_points' own ValueError.
    """

    chart_points = world.meta["chart_points"]
    faces = np.round(chart_points[:, 0]).astype(np.int64)

    return np.flatnonzero(np.isin(faces, face_indices))


def split_pairs(
    num_points,
    test_fraction,
    seed,
    exclude_points=None,
):
    """
    Deterministically split every unordered point pair (i < j) into a
    train pool and a held-out test pool.

    Reproducible from (num_points, test_fraction, seed) alone, so callers
    never need to persist the split - they just call this again with the
    same arguments. Used to carve out a genuinely unseen set of pairs for
    a held-out generalization check, kept disjoint from whatever pool
    make_distance_examples then samples training pairs from.

    exclude_points (optional): an iterable of point indices that must not
    appear in either returned pool - e.g. the holdout points from
    split_points, so the ordinary pair-level held-out check never
    touches a point that is being held out entirely. Leaving this at its
    default (None) reproduces today's behavior exactly.
    """

    if exclude_points is None:
        # Every unordered pair, i < j.
        all_pairs = np.array(
            [
                (i, j)
                for i in range(num_points)
                for j in range(i + 1, num_points)
            ],
            dtype=np.int64,
        )
    else:
        excluded = set(int(p) for p in exclude_points)

        # Every unordered pair, i < j, with neither endpoint excluded.
        all_pairs = np.array(
            [
                (i, j)
                for i in range(num_points)
                for j in range(i + 1, num_points)
                if i not in excluded and j not in excluded
            ],
            dtype=np.int64,
        )

    rng = np.random.default_rng(seed)

    # A random ordering of pair indices; the first slice becomes the
    # held-out test pool, the rest becomes the train pool.
    shuffled_indices = rng.permutation(len(all_pairs))

    num_test = round(len(all_pairs) * test_fraction)

    test_pairs = all_pairs[shuffled_indices[:num_test]]
    train_pairs = all_pairs[shuffled_indices[num_test:]]

    return train_pairs, test_pairs


def make_holdout_point_pairs(
    num_points,
    holdout_points,
    n_probes,
    seed,
):
    """
    For each holdout point h, build every (h, j) pair with j a
    non-holdout point, then split them into a small probe set (used to
    recover h's embedding, see src/holdout_probe.py) and a disjoint eval
    set (used to check how well the recovered embedding generalizes).

    Pairs between two holdout points are deliberately excluded here -
    both endpoints would still be untrained noise at recovery time, so
    they can't serve as probes; src/holdout_probe.py evaluates
    holdout-to-holdout pairs separately, after both points have already
    been recovered.

    Returns {h: {"probe_pairs": int64[n_probes, 2],
                 "eval_pairs": int64[k, 2]}}, one entry per holdout
    point. Reproducible from (num_points, holdout_points, n_probes, seed)
    alone - each holdout point gets its own independent shuffle, seeded
    from (seed, h), so adding/removing holdout points doesn't reshuffle
    the others' splits.
    """

    holdout_set = set(int(h) for h in holdout_points)
    non_holdout = np.array(
        [p for p in range(num_points) if p not in holdout_set],
        dtype=np.int64,
    )

    if n_probes >= len(non_holdout):
        raise ValueError(
            f"n_probes ({n_probes}) must be less than the number of "
            f"available non-holdout partner points ({len(non_holdout)})"
        )

    result = {}

    for h in holdout_points:
        h = int(h)

        # A fresh, reproducible shuffle per holdout point, so probe/eval
        # membership doesn't depend on the order holdout_points is given.
        rng = np.random.default_rng((seed, h))
        shuffled = non_holdout[rng.permutation(len(non_holdout))]

        probe_j = shuffled[:n_probes]
        eval_j = shuffled[n_probes:]

        result[h] = {
            "probe_pairs": np.stack(
                [np.full(n_probes, h, dtype=np.int64), probe_j], axis=1,
            ),
            "eval_pairs": np.stack(
                [np.full(len(eval_j), h, dtype=np.int64), eval_j], axis=1,
            ),
        }

    return result


def make_distance_examples(
    world,
    n=1000,
    seed=0,
    allowed_pairs=None,
):
    """
    Generate supervised examples for the distance task.

    Each example contains:
        point i
        point j
        normalized distance between i and j

    By default, each of the n examples independently draws two distinct
    point indices from the full entity set. When allowed_pairs is given
    (an [k, 2] array of (i, j) pairs, e.g. the train pool returned by
    split_pairs), examples are instead drawn with replacement from that
    restricted pool - this keeps n (e.g. training_cfg["train_examples"])
    unchanged while excluding a held-out set of pairs from training.
    """

    # Create a reproducible random-number generator.
    rng = np.random.default_rng(seed)

    # Store generated examples.
    examples = []

    # Compute the normalization scale once.
    scale = distance_scale(world)

    # Generate n examples.
    for _ in range(n):
        if allowed_pairs is None:
            # Select two different point indices from the full entity set.
            i, j = rng.choice(
                len(world.names),
                size=2,
                replace=False,
            )
        else:
            # Select one pair (with replacement across draws) from the
            # restricted pool.
            i, j = allowed_pairs[
                rng.integers(len(allowed_pairs))
            ]

        examples.append(
            _build_distance_example(world, i, j, scale)
        )

    return examples


def make_distance_examples_from_pairs(world, pairs):
    """
    Build exactly one distance-task example per given pair, in order.

    Unlike make_distance_examples, this does not sample - every pair in
    `pairs` (e.g. the held-out pool from split_pairs) appears exactly
    once, which is what a held-out evaluation set needs.
    """

    scale = distance_scale(world)

    return [
        _build_distance_example(world, i, j, scale)
        for i, j in pairs
    ]


def _build_same_triangle_example(world, i, j):
    """
    Build one supervised same_triangle-task example for point pair (i, j).

    Shared by make_same_triangle_examples and
    make_same_triangle_examples_from_pairs, mirroring
    _build_distance_example.
    """

    same = same_triangle(world, int(i), int(j))

    return {
        # Input point indices used by the embedding layer.
        "indices": (
            int(i),
            int(j),
        ),

        # Binary training target: 1.0 if i and j share a triangular face.
        "answer": 1.0 if same else 0.0,

        # Identifies which task produced this example.
        "task": "same_triangle",
    }


def make_same_triangle_examples(
    world,
    n=1000,
    seed=0,
    allowed_pairs=None,
):
    """
    Generate supervised examples for the same_triangle task.

    Each example contains:
        point i
        point j
        1.0 if i and j lie on the same triangular face, else 0.0

    Sampling behaves exactly like make_distance_examples (uniform draws
    from either the full entity set or a restricted allowed_pairs pool),
    so the same train/test pair split can be reused unchanged across
    both tasks - no class balancing is applied here.
    """

    rng = np.random.default_rng(seed)

    examples = []

    for _ in range(n):
        if allowed_pairs is None:
            i, j = rng.choice(
                len(world.names),
                size=2,
                replace=False,
            )
        else:
            i, j = allowed_pairs[
                rng.integers(len(allowed_pairs))
            ]

        examples.append(
            _build_same_triangle_example(world, i, j)
        )

    return examples


def make_same_triangle_examples_from_pairs(world, pairs):
    """
    Build exactly one same_triangle-task example per given pair, in order.

    Mirrors make_distance_examples_from_pairs - every pair in `pairs`
    appears exactly once.
    """

    return [
        _build_same_triangle_example(world, i, j)
        for i, j in pairs
    ]


class PairDataset(Dataset):
    """
    Convert a list of example dictionaries into PyTorch tensors.
    """

    def __init__(self, examples):
        # Store the first point index from every example.
        self.i = torch.tensor(
            [
                example["indices"][0]
                for example in examples
            ],
            dtype=torch.long,
        )

        # Store the second point index from every example.
        self.j = torch.tensor(
            [
                example["indices"][1]
                for example in examples
            ],
            dtype=torch.long,
        )

        # Store the target value from every example.
        self.y = torch.tensor(
            [
                example["answer"]
                for example in examples
            ],
            dtype=torch.float32,
        )

    def __len__(self):
        # Return the number of examples.
        return len(self.y)

    def __getitem__(self, index):
        # Return one pair and its target.
        return (
            self.i[index],
            self.j[index],
            self.y[index],
        )


def set_seed(seed):
    """
    Seed every random-number generator used by this program.
    """

    # Seed Python randomness.
    random.seed(seed)

    # Seed NumPy randomness.
    np.random.seed(seed)

    # Seed PyTorch CPU randomness.
    torch.manual_seed(seed)

    # Seed every CUDA device when CUDA is available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite_loader(loader):
    """
    Repeatedly iterate over a DataLoader forever.

    When one epoch ends, iteration begins again from a newly shuffled epoch.
    """

    while True:
        # Yield every batch from the current epoch.
        yield from loader
