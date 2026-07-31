import math
import sys

import pytest
import torch

from src.train_multitask import (
    TrainConfig,
    evaluate_distance,
    main,
    parse_args,
)


class LookupDistanceModel(torch.nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.register_buffer(
            "predictions",
            torch.tensor(predictions, dtype=torch.float32),
        )

    def forward_distance(self, point_i, point_j):
        del point_j
        return self.predictions[point_i]


def examples(targets):
    return [
        {
            "indices": (index, index + 1),
            "answer": target,
        }
        for index, target in enumerate(targets)
    ]


def test_evaluate_distance_reports_normalized_and_raw_metrics():
    model = LookupDistanceModel([0.0, 0.4, 1.0])
    model.train()

    metrics = evaluate_distance(
        model=model,
        examples=examples([0.0, 0.5, 1.0]),
        scale=10.0,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert metrics["n_examples"] == 3
    assert metrics["normalized_mae"] == pytest.approx(1.0 / 30.0)
    assert metrics["normalized_rmse"] == pytest.approx(
        math.sqrt(0.01 / 3.0)
    )
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(
        10.0 * math.sqrt(0.01 / 3.0)
    )
    assert metrics["r2"] == pytest.approx(0.98)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert model.training


def test_evaluate_distance_preserves_eval_mode():
    model = LookupDistanceModel([0.0, 1.0])
    model.eval()

    evaluate_distance(
        model=model,
        examples=examples([0.0, 1.0]),
        scale=1.0,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert not model.training


def test_evaluate_distance_handles_constant_values():
    model = LookupDistanceModel([0.5, 0.5])

    metrics = evaluate_distance(
        model=model,
        examples=examples([0.5, 0.5]),
        scale=1.0,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert math.isnan(metrics["r2"])
    assert math.isnan(metrics["spearman"])


def test_evaluate_distance_rejects_empty_examples():
    with pytest.raises(
        ValueError,
        match="requires at least one example",
    ):
        evaluate_distance(
            model=LookupDistanceModel([]),
            examples=[],
            scale=1.0,
            device=torch.device("cpu"),
            batch_size=2,
        )


def test_periodic_checkpoints_preserve_schema_and_do_not_trigger_eval(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    cfg = TrainConfig(
        width=3,
        height=3,
        train_examples_per_task=3,
        val_examples_per_task=2,
        batch_size=2,
        steps=3,
        eval_every=10,
        checkpoint_every=1,
        distance_pair_seed=6,
        seed=7,
    )

    main(cfg)

    model_dir = tmp_path / "models"
    base_name = "grid_distance_pairs3_pairseed6_seed7"
    step_one_path = model_dir / f"{base_name}_step1_model.pt"
    step_two_path = model_dir / f"{base_name}_step2_model.pt"
    final_path = model_dir / f"{base_name}_model.pt"

    assert step_one_path.is_file()
    assert step_two_path.is_file()
    assert final_path.is_file()
    assert not (model_dir / f"{base_name}_step3_model.pt").exists()

    step_two = torch.load(
        step_two_path,
        map_location="cpu",
        weights_only=False,
    )
    final = torch.load(
        final_path,
        map_location="cpu",
        weights_only=False,
    )

    existing_schema = {
        "model_state_dict",
        "config",
        "world_meta",
        "distance_train_pairs",
        "distance_eval_pairs",
        "evaluation",
        "world_coordinates",
        "ambient_coordinates",
    }
    assert existing_schema <= step_two.keys()
    assert step_two["training_step"] == 2
    assert "state" in step_two["optimizer_state_dict"]
    assert "param_groups" in step_two["optimizer_state_dict"]
    assert [
        entry["step"]
        for entry in step_two["evaluation"]["distance_history"]
    ] == [1]
    assert final["training_step"] == 3
    assert [
        entry["step"]
        for entry in final["evaluation"]["distance_history"]
    ] == [1, 3]


def test_checkpoint_cli_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_multitask.py", "--checkpoint-every", "250"],
    )

    assert parse_args().checkpoint_every == 250


def test_checkpoint_frequency_must_be_positive():
    with pytest.raises(
        ValueError,
        match="checkpoint_every must be a positive integer or None",
    ):
        main(
            TrainConfig(
                checkpoint_every=0,
            )
        )
