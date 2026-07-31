import numpy as np
import pytest

from manifolds.mobius import FlatMobiusStrip
from src.datasets import (
    distance_scale,
    make_disjoint_distance_splits,
    sample_unique_distance_pairs,
)
from src.tasks import distance
from src.worlds import make_grid, make_manifold_world


def example_pairs(examples):
    return np.asarray(
        [example["indices"] for example in examples],
        dtype=np.int64,
    )


def test_distance_pairs_are_unique_and_unordered():
    pairs = sample_unique_distance_pairs(
        num_points=20,
        n_pairs=100,
        seed=7,
    )

    assert pairs.shape == (100, 2)
    assert np.all(pairs[:, 0] < pairs[:, 1])
    assert len({tuple(pair) for pair in pairs}) == 100


def test_pair_budgets_are_nested_for_same_seed():
    small = sample_unique_distance_pairs(40, 100, seed=3)
    large = sample_unique_distance_pairs(40, 250, seed=3)

    np.testing.assert_array_equal(
        small,
        large[: len(small)],
    )


def test_pair_budget_cannot_exceed_all_possible_pairs():
    with pytest.raises(ValueError, match="between 1 and 10"):
        sample_unique_distance_pairs(5, 11, seed=0)


def test_disjoint_splits_keep_eval_fixed_and_train_budgets_nested():
    world = make_grid(10, 10)

    small_train, small_eval = make_disjoint_distance_splits(
        world,
        n_train=100,
        n_eval=50,
        seed=9,
    )
    large_train, large_eval = make_disjoint_distance_splits(
        world,
        n_train=250,
        n_eval=50,
        seed=9,
    )

    small_train_pairs = example_pairs(small_train)
    large_train_pairs = example_pairs(large_train)
    small_eval_pairs = example_pairs(small_eval)
    large_eval_pairs = example_pairs(large_eval)

    np.testing.assert_array_equal(
        small_eval_pairs,
        large_eval_pairs,
    )
    np.testing.assert_array_equal(
        small_train_pairs,
        large_train_pairs[: len(small_train_pairs)],
    )
    assert set(map(tuple, large_train_pairs)).isdisjoint(
        set(map(tuple, large_eval_pairs))
    )
    assert np.all(large_train_pairs[:, 0] < large_train_pairs[:, 1])
    assert np.all(large_eval_pairs[:, 0] < large_eval_pairs[:, 1])


@pytest.mark.parametrize(
    ("n_train", "n_eval", "message"),
    [
        (0, 1, "n_train must be a positive integer"),
        (1, 0, "n_eval must be a positive integer"),
        (8, 3, "only 10 unique pairs are available"),
    ],
)
def test_disjoint_split_rejects_invalid_pair_counts(
    n_train,
    n_eval,
    message,
):
    with pytest.raises(ValueError, match=message):
        make_disjoint_distance_splits(
            make_grid(5, 1),
            n_train=n_train,
            n_eval=n_eval,
        )


def test_disjoint_grid_examples_use_world_distance_and_scale():
    world = make_grid(4, 3)
    train, evaluation = make_disjoint_distance_splits(
        world,
        n_train=4,
        n_eval=3,
        seed=2,
    )
    scale = distance_scale(world)

    for example in [*train, *evaluation]:
        i, j = example["indices"]
        expected = distance(world, i, j)
        assert example["raw_answer"] == pytest.approx(expected)
        assert example["answer"] == pytest.approx(expected / scale)


def test_disjoint_manifold_examples_use_geodesic_distance_and_diameter():
    manifold = FlatMobiusStrip()
    world = make_manifold_world(
        manifold,
        n=12,
        seed=4,
        diameter=np.pi,
    )
    train, evaluation = make_disjoint_distance_splits(
        world,
        n_train=5,
        n_eval=4,
        seed=3,
    )

    for example in [*train, *evaluation]:
        i, j = example["indices"]
        expected = manifold.distance(
            world.coordinates[i],
            world.coordinates[j],
        )
        assert example["raw_answer"] == pytest.approx(expected)
        assert example["answer"] == pytest.approx(expected / np.pi)
