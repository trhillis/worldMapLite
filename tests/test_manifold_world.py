"""Smoke tests for the World/task/dataset plumbing around manifolds.

These catch wiring bugs (index mismatches, wrong dispatch branch) cheaply,
before a multi-minute full-scale training run.
"""

from __future__ import annotations

import numpy as np

from manifolds import octahedron, get_manifold
from src.worlds import make_manifold_world
from src.tasks import distance
from src.datasets import (
    distance_scale,
    make_distance_examples,
    make_distance_examples_from_pairs,
    split_pairs,
)


def test_manifold_world_distance_matches_manifold_distance(tmp_path):
    manifold = octahedron()
    world = make_manifold_world(
        manifold, n=20, seed=0, cache_dir=str(tmp_path),
    )

    chart_points = world.meta["chart_points"]

    for i, j in [(0, 1), (2, 5), (0, 19), (7, 7)]:
        expected = manifold.distance(chart_points[i], chart_points[j])
        actual = distance(world, i, j)
        assert actual == expected


def test_distance_scale_matches_matrix_max(tmp_path):
    manifold = octahedron()
    world = make_manifold_world(
        manifold, n=20, seed=0, cache_dir=str(tmp_path),
    )

    assert distance_scale(world) == world.meta["distance_matrix"].max()


def test_manifold_world_cache_is_reused(tmp_path):
    manifold = octahedron()

    world_a = make_manifold_world(
        manifold, n=15, seed=1, cache_dir=str(tmp_path),
    )
    cache_files = list(tmp_path.glob("*.npz"))
    assert len(cache_files) == 1

    world_b = make_manifold_world(
        manifold, n=15, seed=1, cache_dir=str(tmp_path),
    )

    # Same seed -> identical sampled points and distances, served from cache.
    np.testing.assert_array_equal(
        world_a.meta["chart_points"], world_b.meta["chart_points"],
    )
    np.testing.assert_array_equal(
        world_a.meta["distance_matrix"], world_b.meta["distance_matrix"],
    )


def test_make_distance_examples_on_manifold_world(tmp_path):
    manifold = get_manifold("octahedron")
    world = make_manifold_world(
        manifold, n=20, seed=0, cache_dir=str(tmp_path),
    )

    examples = make_distance_examples(world, n=10, seed=0)
    assert len(examples) == 10

    for example in examples:
        i, j = example["indices"]
        assert 0 <= example["answer"] <= 1.0
        assert example["raw_answer"] == distance(world, i, j)


def test_split_pairs_is_disjoint_and_covers_all_pairs():
    num_points = 20
    train_pairs, test_pairs = split_pairs(
        num_points=num_points, test_fraction=0.2, seed=0,
    )

    total_pairs = num_points * (num_points - 1) // 2
    assert len(train_pairs) + len(test_pairs) == total_pairs

    train_set = {tuple(pair) for pair in train_pairs}
    test_set = {tuple(pair) for pair in test_pairs}

    assert train_set.isdisjoint(test_set)

    all_pairs = {
        (i, j)
        for i in range(num_points)
        for j in range(i + 1, num_points)
    }
    assert train_set | test_set == all_pairs


def test_split_pairs_is_reproducible():
    train_a, test_a = split_pairs(num_points=20, test_fraction=0.2, seed=0)
    train_b, test_b = split_pairs(num_points=20, test_fraction=0.2, seed=0)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)


def test_split_pairs_respects_fraction():
    num_points = 20
    train_pairs, test_pairs = split_pairs(
        num_points=num_points, test_fraction=0.2, seed=0,
    )

    total_pairs = num_points * (num_points - 1) // 2
    assert len(test_pairs) == round(0.2 * total_pairs)


def test_make_distance_examples_respects_allowed_pairs(tmp_path):
    manifold = get_manifold("octahedron")
    world = make_manifold_world(
        manifold, n=20, seed=0, cache_dir=str(tmp_path),
    )

    train_pairs, _ = split_pairs(num_points=20, test_fraction=0.2, seed=0)
    allowed_set = {tuple(pair) for pair in train_pairs}

    examples = make_distance_examples(
        world, n=50, seed=0, allowed_pairs=train_pairs,
    )
    assert len(examples) == 50

    for example in examples:
        i, j = example["indices"]
        assert tuple(sorted((i, j))) in allowed_set


def test_make_distance_examples_from_pairs_matches_exact_pairs(tmp_path):
    manifold = get_manifold("octahedron")
    world = make_manifold_world(
        manifold, n=20, seed=0, cache_dir=str(tmp_path),
    )

    pairs = [(0, 1), (2, 5), (0, 19)]
    examples = make_distance_examples_from_pairs(world, pairs)

    assert len(examples) == len(pairs)

    for example, (i, j) in zip(examples, pairs):
        assert example["indices"] == (i, j)
        assert example["raw_answer"] == distance(world, i, j)
