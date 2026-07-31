import math

import pytest
import torch

from src.train_multitask import evaluate_distance


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
