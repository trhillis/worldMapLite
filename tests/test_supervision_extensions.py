import csv
import sys

import numpy as np
import pytest
import torch

from src.multitask_model import MultiTaskWorldModel
from src.point_recovery import recover_held_out_points
from src.run_sweep import expand_sweep
from src.splits import (
    make_pair_split,
    make_point_split,
    make_recovery_observation_splits,
    validate_pair_split,
)
from src.train_multitask import TrainConfig, main, parse_args
from src.worlds import make_grid, subset_world


def test_pair_split_is_deterministic_fixed_nested_and_disjoint():
    small = make_pair_split(30, 20, 10, seed=4)
    large = make_pair_split(30, 50, 10, seed=4)
    repeat = make_pair_split(30, 20, 10, seed=4)
    np.testing.assert_array_equal(small.held_out_pairs, large.held_out_pairs)
    np.testing.assert_array_equal(small.held_out_pairs, repeat.held_out_pairs)
    assert small.held_out_digest == large.held_out_digest
    np.testing.assert_array_equal(small.train_pairs, large.train_pairs[:20])
    assert set(map(tuple, large.train_pairs)).isdisjoint(map(tuple, large.held_out_pairs))


def test_pair_split_validation_detects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_pair_split(np.array([[0, 1]]), np.array([[0, 1]]))


def test_point_split_deterministic_and_excluded_from_base_pairs():
    first = make_point_split(20, n_held_out=4, seed=9)
    second = make_point_split(20, n_held_out=4, seed=9)
    np.testing.assert_array_equal(first.held_out_points, second.held_out_points)
    pairs = make_pair_split(20, 20, 5, seed=3, excluded_points=first.held_out_points)
    assert set(pairs.train_pairs.ravel()).isdisjoint(first.held_out_points)
    assert set(pairs.held_out_pairs.ravel()).isdisjoint(first.held_out_points)


def test_recovery_anchor_and_evaluation_observations_are_disjoint():
    splits = make_recovery_observation_splits(12, [2, 7], 3, seed=5)
    for split in splits.values():
        anchors = set(split["anchor_pairs"][:, 1])
        evaluation = set(split["evaluation_pairs"][:, 1])
        assert anchors.isdisjoint(evaluation)


def test_recovery_changes_new_vectors_but_not_base_model():
    full_world = make_grid(4, 3)
    point_split = make_point_split(12, n_held_out=2, seed=2)
    retained_world = subset_world(full_world, point_split.retained_points)
    torch.manual_seed(0)
    model = MultiTaskWorldModel(
        len(retained_world.names), emb_dim=8, hidden_dim=16, num_heads=2, num_layers=1,
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    result = recover_held_out_points(
        model, full_world, point_split.retained_points, point_split.held_out_points,
        anchor_count=2, steps=2, learning_rate=0.02, seed=3,
        device=torch.device("cpu"),
    )
    assert all(row["recovered_embedding_delta"] > 0 for row in result["rows"])
    for name, value in model.named_parameters():
        assert torch.equal(value.detach(), before[name])


def test_requested_checkpoint_evaluations_and_tidy_smoke_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = main(TrainConfig(
        width=3, height=3, train_examples_per_task=4, val_examples_per_task=2,
        batch_size=2, steps=3, evaluation_checkpoints=(1, 3),
        held_out_points=1, held_out_point_seed=2, recovery_anchor_counts=(2,),
        recovery_steps=1, pair_split_seed=5, seed=6, results_dir="results",
    ))
    checkpoint = torch.load(output["checkpoint"], map_location="cpu", weights_only=False)
    assert [row["step"] for row in checkpoint["evaluation"]["distance_history"]] == [1, 3]
    assert checkpoint["pair_split"]["pair_split_seed"] == 5
    held_out = set(checkpoint["point_split"]["held_out_points"])
    assert set(checkpoint["distance_train_pairs"].numpy().ravel()).isdisjoint(held_out)
    with open(output["learning_curve"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["optimizer_updates"]) for row in rows] == [1, 3]
    assert output["recovery_rows"]


def test_existing_config_and_cli_remain_backward_compatible(monkeypatch):
    cfg = TrainConfig()
    assert cfg.evaluation_checkpoints is None
    assert cfg.held_out_points == 0
    monkeypatch.setattr(sys, "argv", ["train_multitask.py", "--pair-seed", "8"])
    args = parse_args()
    assert args.pair_seed == 8
    assert args.pair_split_seed is None


def test_sweep_only_adds_curves_and_recovery_to_selected_budgets():
    runs = expand_sweep({
        "worlds": [{"world_type": "grid", "width": 3, "height": 3}],
        "supervision_budgets": [2, 4],
        "model_seeds": [0, 1],
        "fixed": {"steps": 3, "eval_pairs": 1, "pair_split_seed": 7, "world_seed": 8, "data_order_seed": 9},
        "learning_curves": {"selected_budgets": [4], "checkpoints": [1, 3]},
        "held_out_point_recovery": {"selected_budgets": [4], "held_out_points": 1, "anchor_counts": [2]},
    })
    assert len(runs) == 6
    for run in runs:
        assert run.pair_split_seed == 7 and run.world_seed == 8 and run.data_order_seed == 9
        assert (run.evaluation_checkpoints == (1, 3)) == (run.train_examples_per_task == 4)
    recovery_runs = [run for run in runs if run.recovery_anchor_counts]
    assert len(recovery_runs) == 2
    assert all(run.train_examples_per_task == 4 for run in recovery_runs)
