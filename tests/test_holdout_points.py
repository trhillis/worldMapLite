"""Tests for the point-level holdout generalization feature.

Covers src/datasets.py's split_points / points_on_faces / split_pairs's
exclude_points extension / make_holdout_point_pairs, the
pair_representation refactor in src/multitask_model.py, and
src/holdout_probe.py's recover_and_evaluate_holdout_points.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from manifolds import octahedron
from src.worlds import make_manifold_world
from src.datasets import (
    split_points,
    points_on_faces,
    split_pairs,
    make_holdout_point_pairs,
)
from src.multitask_model import MultiTaskWorldModel
from src.holdout_probe import recover_and_evaluate_holdout_points, _EVAL_BUILDERS


# --------------------------------------------------
# split_points
# --------------------------------------------------

def test_split_points_disjoint_and_covers_all_indices():
    train_points, holdout_points = split_points(
        num_points=20, n_holdout=5, seed=0,
    )

    train_set = set(train_points.tolist())
    holdout_set = set(holdout_points.tolist())

    assert train_set.isdisjoint(holdout_set)
    assert train_set | holdout_set == set(range(20))


def test_split_points_respects_n_holdout():
    train_points, holdout_points = split_points(
        num_points=20, n_holdout=5, seed=0,
    )

    assert len(holdout_points) == 5
    assert len(train_points) == 15


def test_split_points_is_reproducible():
    train_a, holdout_a = split_points(num_points=20, n_holdout=5, seed=0)
    train_b, holdout_b = split_points(num_points=20, n_holdout=5, seed=0)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(holdout_a, holdout_b)


def test_split_points_candidate_points_restricts_holdout():
    candidate_points = np.array([2, 5, 7, 9, 11])

    train_points, holdout_points = split_points(
        num_points=20, n_holdout=3, seed=0, candidate_points=candidate_points,
    )

    assert set(holdout_points.tolist()).issubset(set(candidate_points.tolist()))
    assert set(train_points.tolist()) == set(range(20)) - set(
        holdout_points.tolist()
    )


def test_split_points_candidate_points_is_reproducible():
    candidate_points = np.array([2, 5, 7, 9, 11])

    train_a, holdout_a = split_points(
        num_points=20, n_holdout=3, seed=0, candidate_points=candidate_points,
    )
    train_b, holdout_b = split_points(
        num_points=20, n_holdout=3, seed=0, candidate_points=candidate_points,
    )

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(holdout_a, holdout_b)


def test_split_points_raises_when_n_holdout_exceeds_candidates():
    candidate_points = np.array([2, 5, 7])

    with pytest.raises(ValueError):
        split_points(
            num_points=20, n_holdout=5, seed=0,
            candidate_points=candidate_points,
        )


# --------------------------------------------------
# points_on_faces
# --------------------------------------------------

def test_points_on_faces_returns_matching_face_indices(tmp_path):
    manifold = octahedron()
    world = make_manifold_world(
        manifold, n=50, seed=0, cache_dir=str(tmp_path),
    )

    indices = points_on_faces(world, [0])

    faces = np.round(world.meta["chart_points"][:, 0]).astype(np.int64)
    assert np.all(faces[indices] == 0)
    assert set(indices.tolist()) == set(np.flatnonzero(faces == 0).tolist())


def test_points_on_faces_empty_face_returns_empty_array(tmp_path):
    manifold = octahedron()
    world = make_manifold_world(
        manifold, n=50, seed=0, cache_dir=str(tmp_path),
    )

    # Face index 999 doesn't exist on an 8-face octahedron.
    indices = points_on_faces(world, [999])

    assert len(indices) == 0


# --------------------------------------------------
# split_pairs(..., exclude_points=...)
# --------------------------------------------------

def test_split_pairs_exclude_points_never_touches_excluded_indices():
    exclude_points = [2, 5, 9]

    train_pairs, test_pairs = split_pairs(
        num_points=20, test_fraction=0.2, seed=0, exclude_points=exclude_points,
    )

    for pair in np.concatenate([train_pairs, test_pairs]):
        assert pair[0] not in exclude_points
        assert pair[1] not in exclude_points


def test_split_pairs_exclude_points_covers_all_remaining_pairs():
    num_points = 20
    exclude_points = {2, 5, 9}

    train_pairs, test_pairs = split_pairs(
        num_points=num_points, test_fraction=0.2, seed=0,
        exclude_points=exclude_points,
    )

    expected_pairs = {
        (i, j)
        for i in range(num_points)
        for j in range(i + 1, num_points)
        if i not in exclude_points and j not in exclude_points
    }

    got_pairs = {tuple(pair) for pair in train_pairs} | {
        tuple(pair) for pair in test_pairs
    }

    assert got_pairs == expected_pairs


def test_split_pairs_exclude_points_is_reproducible():
    exclude_points = [2, 5, 9]

    train_a, test_a = split_pairs(
        num_points=20, test_fraction=0.2, seed=0, exclude_points=exclude_points,
    )
    train_b, test_b = split_pairs(
        num_points=20, test_fraction=0.2, seed=0, exclude_points=exclude_points,
    )

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)


def test_split_pairs_backward_compatible_when_exclude_points_omitted():
    # Positional-call (today's original signature) vs. an explicit
    # exclude_points=None keyword call must be identical - the
    # backward-compatibility guarantee every existing caller relies on.
    train_a, test_a = split_pairs(20, 0.2, 0)
    train_b, test_b = split_pairs(20, 0.2, 0, exclude_points=None)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)


# --------------------------------------------------
# make_holdout_point_pairs
# --------------------------------------------------

def test_make_holdout_point_pairs_probe_and_eval_disjoint_and_cover_all_partners():
    num_points = 20
    holdout_points = np.array([1, 2])

    result = make_holdout_point_pairs(
        num_points=num_points, holdout_points=holdout_points, n_probes=3, seed=0,
    )

    non_holdout = set(range(num_points)) - set(holdout_points.tolist())

    for h in holdout_points:
        probe_partners = set(result[h]["probe_pairs"][:, 1].tolist())
        eval_partners = set(result[h]["eval_pairs"][:, 1].tolist())

        assert probe_partners.isdisjoint(eval_partners)
        assert probe_partners | eval_partners == non_holdout


def test_make_holdout_point_pairs_respects_n_probes():
    holdout_points = np.array([1, 2])

    result = make_holdout_point_pairs(
        num_points=20, holdout_points=holdout_points, n_probes=4, seed=0,
    )

    for h in holdout_points:
        assert len(result[h]["probe_pairs"]) == 4


def test_make_holdout_point_pairs_is_reproducible():
    holdout_points = np.array([1, 2])

    result_a = make_holdout_point_pairs(
        num_points=20, holdout_points=holdout_points, n_probes=4, seed=0,
    )
    result_b = make_holdout_point_pairs(
        num_points=20, holdout_points=holdout_points, n_probes=4, seed=0,
    )

    for h in holdout_points:
        np.testing.assert_array_equal(
            result_a[h]["probe_pairs"], result_b[h]["probe_pairs"],
        )
        np.testing.assert_array_equal(
            result_a[h]["eval_pairs"], result_b[h]["eval_pairs"],
        )


def test_make_holdout_point_pairs_raises_when_n_probes_too_large():
    with pytest.raises(ValueError):
        make_holdout_point_pairs(
            num_points=5, holdout_points=np.array([0]), n_probes=10, seed=0,
        )


# --------------------------------------------------
# pair_representation refactor regression
# --------------------------------------------------

def test_pair_representation_from_embeddings_matches_pair_representation():
    torch.manual_seed(0)
    model = MultiTaskWorldModel(num_points=10, emb_dim=8, hidden_dim=16,
                                 num_heads=2, num_layers=1)
    model.eval()

    i = torch.tensor([0, 1, 2])
    j = torch.tensor([3, 4, 5])

    with torch.no_grad():
        via_indices = model.pair_representation(i, j, model.distance_token)
        via_embeddings = model.pair_representation_from_embeddings(
            model.encode(i), model.encode(j), model.distance_token,
        )

    assert torch.equal(via_indices, via_embeddings)


def test_forward_distance_from_embeddings_matches_forward_distance():
    torch.manual_seed(0)
    model = MultiTaskWorldModel(num_points=10, emb_dim=8, hidden_dim=16,
                                 num_heads=2, num_layers=1)
    model.eval()

    i = torch.tensor([0, 1, 2])
    j = torch.tensor([3, 4, 5])

    with torch.no_grad():
        via_indices = model.forward_distance(i, j)
        via_embeddings = model.forward_distance_from_embeddings(
            model.encode(i), model.encode(j),
        )

    assert torch.equal(via_indices, via_embeddings)


# --------------------------------------------------
# recover_and_evaluate_holdout_points
# --------------------------------------------------

def _tiny_world_and_model(tmp_path, n=20, seed=0):
    manifold = octahedron()
    world = make_manifold_world(manifold, n=n, seed=seed, cache_dir=str(tmp_path))

    torch.manual_seed(seed)
    model = MultiTaskWorldModel(
        num_points=n, emb_dim=8, hidden_dim=16, num_heads=2, num_layers=1,
    )

    return world, model


def test_recover_and_evaluate_holdout_points_freezes_everything_else(tmp_path):
    world, model = _tiny_world_and_model(tmp_path)

    holdout_points = np.array([1, 2])
    holdout_point_pairs = make_holdout_point_pairs(
        num_points=len(world.names), holdout_points=holdout_points,
        n_probes=3, seed=0,
    )

    non_holdout_mask = np.ones(len(world.names), dtype=bool)
    non_holdout_mask[holdout_points] = False

    original_emb = model.emb.weight.detach().clone()
    original_other_params = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name != "emb.weight"
    }
    original_requires_grad = {
        name: p.requires_grad for name, p in model.named_parameters()
    }

    recover_and_evaluate_holdout_points(
        model=model,
        world=world,
        holdout_point_pairs=holdout_point_pairs,
        probe_steps=20,
        probe_lr=0.05,
        probe_weight_decay=1e-4,
        batch_size=64,
        device=torch.device("cpu"),
    )

    # Non-holdout embedding rows are bit-identical.
    assert torch.equal(
        model.emb.weight.data[non_holdout_mask],
        original_emb[non_holdout_mask],
    )

    # Holdout rows actually changed.
    assert not torch.equal(
        model.emb.weight.data[holdout_points],
        original_emb[holdout_points],
    )

    # Every other parameter is untouched.
    for name, p in model.named_parameters():
        if name == "emb.weight":
            continue
        assert torch.equal(p.detach(), original_other_params[name])

    # requires_grad flags are unchanged.
    for name, p in model.named_parameters():
        assert p.requires_grad == original_requires_grad[name]


def test_recover_and_evaluate_holdout_points_returns_sane_finite_output(tmp_path):
    world, model = _tiny_world_and_model(tmp_path)

    # 3 (not 2) holdout points, so holdout_to_holdout has 3 pairs -
    # Spearman correlation is mathematically undefined (NaN) for a
    # single pair, which 2 holdout points would produce.
    holdout_points = np.array([1, 2, 3])
    holdout_point_pairs = make_holdout_point_pairs(
        num_points=len(world.names), holdout_points=holdout_points,
        n_probes=3, seed=0,
    )

    result = recover_and_evaluate_holdout_points(
        model=model,
        world=world,
        holdout_point_pairs=holdout_point_pairs,
        probe_steps=20,
        probe_lr=0.05,
        probe_weight_decay=1e-4,
        batch_size=64,
        device=torch.device("cpu"),
    )

    assert result["n_holdout_points"] == 3
    assert result["n_probes"] == 3
    # Default eval_tasks=("distance",), so every result dict is nested
    # one level under the task name.
    assert result["eval_tasks"] == ["distance"]
    assert set(result["per_point"]["distance"].keys()) == {1, 2, 3}

    for stats in list(result["per_point"]["distance"].values()) + [
        result["aggregate"]["distance"], result["holdout_to_holdout"]["distance"],
    ]:
        assert np.isfinite(stats["loss"])
        assert np.isfinite(stats["spearman"])
        assert -1.0 <= stats["spearman"] <= 1.0


def test_recover_and_evaluate_holdout_points_holdout_to_holdout_none_for_single_point(tmp_path):
    world, model = _tiny_world_and_model(tmp_path)

    holdout_points = np.array([1])
    holdout_point_pairs = make_holdout_point_pairs(
        num_points=len(world.names), holdout_points=holdout_points,
        n_probes=3, seed=0,
    )

    result = recover_and_evaluate_holdout_points(
        model=model,
        world=world,
        holdout_point_pairs=holdout_point_pairs,
        probe_steps=10,
        probe_lr=0.05,
        probe_weight_decay=1e-4,
        batch_size=64,
        device=torch.device("cpu"),
    )

    assert result["holdout_to_holdout"]["distance"] is None


def test_recover_and_evaluate_holdout_points_eval_tasks_covers_same_triangle(tmp_path):
    # Recovery always fits holdout_param against distance probes only
    # (see src/holdout_probe.py's module docstring); eval_tasks controls
    # which task(s) the already-recovered embedding is then evaluated on
    # - this is the mechanism behind
    # configs/octahedron_holdout_same_triangle_eval.yaml's "tuned only on
    # distance, evaluated on same_triangle too" experiment.
    world, model = _tiny_world_and_model(tmp_path)

    holdout_points = np.array([1, 2, 3])
    holdout_point_pairs = make_holdout_point_pairs(
        num_points=len(world.names), holdout_points=holdout_points,
        n_probes=3, seed=0,
    )

    result = recover_and_evaluate_holdout_points(
        model=model,
        world=world,
        holdout_point_pairs=holdout_point_pairs,
        probe_steps=20,
        probe_lr=0.05,
        probe_weight_decay=1e-4,
        batch_size=64,
        device=torch.device("cpu"),
        eval_tasks=("distance", "same_triangle"),
    )

    assert result["eval_tasks"] == ["distance", "same_triangle"]
    assert set(result["per_point"].keys()) == {"distance", "same_triangle"}
    assert set(result["per_point"]["same_triangle"].keys()) == {1, 2, 3}

    distance_aggregate = result["aggregate"]["distance"]
    same_triangle_aggregate = result["aggregate"]["same_triangle"]

    assert np.isfinite(distance_aggregate["loss"])
    assert np.isfinite(distance_aggregate["spearman"])
    assert np.isfinite(same_triangle_aggregate["loss"])
    assert 0.0 <= same_triangle_aggregate["accuracy"] <= 1.0

    assert result["holdout_to_holdout"]["distance"] is not None
    assert result["holdout_to_holdout"]["same_triangle"] is not None


def test_eval_builders_registry_covers_distance_and_same_triangle():
    assert set(_EVAL_BUILDERS.keys()) == {"distance", "same_triangle"}
