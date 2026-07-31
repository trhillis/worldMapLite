"""Tests for the same_triangle task: ground truth, dataset generation,
model wiring, and evaluation - mirrors tests/test_manifold_world.py's
coverage of the distance task.
"""

from __future__ import annotations

import numpy as np
import torch

from manifolds import octahedron, get_manifold
from src.worlds import make_manifold_world
from src.tasks import same_triangle
from src.datasets import (
    make_same_triangle_examples,
    make_same_triangle_examples_from_pairs,
    split_pairs,
)
from src.multitask_model import MultiTaskWorldModel
from src.eval_utils import evaluate_same_triangle_examples


def test_same_triangle_matches_manual_face_comparison(tmp_path):
    manifold = octahedron()
    world = make_manifold_world(
        manifold, n=50, seed=0, cache_dir=str(tmp_path),
    )

    faces = np.round(world.meta["chart_points"][:, 0]).astype(np.int64)

    for i, j in [(0, 1), (2, 5), (0, 19), (7, 7), (3, 40)]:
        expected = bool(faces[i] == faces[j])
        assert same_triangle(world, i, j) == expected


def test_same_triangle_true_for_identical_point(tmp_path):
    manifold = octahedron()
    world = make_manifold_world(
        manifold, n=20, seed=0, cache_dir=str(tmp_path),
    )

    assert same_triangle(world, 3, 3) is True


def test_make_same_triangle_examples_on_manifold_world(tmp_path):
    manifold = get_manifold("octahedron")
    world = make_manifold_world(
        manifold, n=20, seed=0, cache_dir=str(tmp_path),
    )

    examples = make_same_triangle_examples(world, n=10, seed=0)
    assert len(examples) == 10

    for example in examples:
        i, j = example["indices"]
        assert example["task"] == "same_triangle"
        assert example["answer"] in (0.0, 1.0)
        assert example["answer"] == (
            1.0 if same_triangle(world, i, j) else 0.0
        )


def test_make_same_triangle_examples_respects_allowed_pairs(tmp_path):
    manifold = get_manifold("octahedron")
    world = make_manifold_world(
        manifold, n=20, seed=0, cache_dir=str(tmp_path),
    )

    train_pairs, _ = split_pairs(num_points=20, test_fraction=0.2, seed=0)
    allowed_set = {tuple(pair) for pair in train_pairs}

    examples = make_same_triangle_examples(
        world, n=50, seed=0, allowed_pairs=train_pairs,
    )
    assert len(examples) == 50

    for example in examples:
        i, j = example["indices"]
        assert tuple(sorted((i, j))) in allowed_set


def test_make_same_triangle_examples_from_pairs_matches_exact_pairs(tmp_path):
    manifold = get_manifold("octahedron")
    world = make_manifold_world(
        manifold, n=20, seed=0, cache_dir=str(tmp_path),
    )

    pairs = [(0, 1), (2, 5), (0, 19)]
    examples = make_same_triangle_examples_from_pairs(world, pairs)

    assert len(examples) == len(pairs)

    for example, (i, j) in zip(examples, pairs):
        assert example["indices"] == (i, j)
        assert example["answer"] == (
            1.0 if same_triangle(world, i, j) else 0.0
        )


def test_forward_same_triangle_is_pair_symmetric_and_shape_correct():
    torch.manual_seed(0)
    model = MultiTaskWorldModel(
        num_points=10, emb_dim=8, hidden_dim=16, num_heads=2, num_layers=1,
    )
    model.eval()

    i = torch.tensor([0, 1, 2])
    j = torch.tensor([3, 4, 5])

    with torch.no_grad():
        forward_logits = model.forward_same_triangle(i, j)
        reverse_logits = model.forward_same_triangle(j, i)

    assert forward_logits.shape == (3,)
    assert torch.allclose(forward_logits, reverse_logits, atol=1e-6)


def test_forward_routes_same_triangle_task():
    torch.manual_seed(0)
    model = MultiTaskWorldModel(
        num_points=10, emb_dim=8, hidden_dim=16, num_heads=2, num_layers=1,
    )
    model.eval()

    i = torch.tensor([0, 1])
    j = torch.tensor([2, 3])

    with torch.no_grad():
        via_forward = model.forward_same_triangle(i, j)
        via_router = model("same_triangle", i, j)

    assert torch.equal(via_forward, via_router)


def test_evaluate_same_triangle_examples_returns_sane_finite_output(tmp_path):
    manifold = get_manifold("octahedron")
    world = make_manifold_world(
        manifold, n=20, seed=0, cache_dir=str(tmp_path),
    )

    torch.manual_seed(0)
    model = MultiTaskWorldModel(
        num_points=20, emb_dim=8, hidden_dim=16, num_heads=2, num_layers=1,
    )

    examples = make_same_triangle_examples(world, n=30, seed=0)

    result = evaluate_same_triangle_examples(
        model, examples, device=torch.device("cpu"), batch_size=16,
    )

    assert result["n_pairs"] == 30
    assert np.isfinite(result["loss"])
    assert 0.0 <= result["accuracy"] <= 1.0
