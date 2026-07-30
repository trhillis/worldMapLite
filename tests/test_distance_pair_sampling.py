import numpy as np
import pytest

from src.datasets import sample_unique_distance_pairs


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
