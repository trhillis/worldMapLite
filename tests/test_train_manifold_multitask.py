"""End-to-end smoke test for src/train_manifold.py's multi-task support:
confirms a `tasks: [distance, same_triangle]` config trains without error
and produces a checkpoint with the expected per-task nested structure.

Runs in a temp working directory (models/ and data/ are relative paths in
train_manifold.py) so it doesn't touch the real project's models/ output.
"""

from __future__ import annotations

import yaml
import torch

from src.train_manifold import main


def test_train_manifold_multitask_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = {
        "manifold": {"name": "octahedron", "n_points": 30, "seed": 0},
        "model": {
            "emb_dim": 8, "hidden_dim": 16, "num_heads": 2, "num_layers": 1,
            "dropout": 0.0, "normalize_embeddings": False,
        },
        "training": {
            "tasks": ["distance", "same_triangle"],
            "train_examples": 200,
            "batch_size": 16,
            "steps": 20,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "seed": 0,
            "progress_interval": 10,
            "test_fraction": 0.2,
        },
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    main(str(config_path))

    checkpoint_path = tmp_path / "models" / "octahedron_distance_same_triangle_model.pt"
    assert checkpoint_path.exists()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint["config"]["training"]["tasks"] == ["distance", "same_triangle"]

    held_out_eval = checkpoint["held_out_eval"]
    assert set(held_out_eval["per_task"].keys()) == {"distance", "same_triangle"}
    assert "loss" in held_out_eval["per_task"]["distance"]["held_out"]
    assert "spearman" in held_out_eval["per_task"]["distance"]["held_out"]
    assert "accuracy" in held_out_eval["per_task"]["same_triangle"]["held_out"]

    # Progress snapshots carry both tasks' held-out/in-distribution metrics.
    assert len(checkpoint["progress"]) > 0
    snapshot = checkpoint["progress"][0]
    assert "distance_held_out_loss" in snapshot
    assert "same_triangle_held_out_loss" in snapshot
    assert "same_triangle_held_out_accuracy" in snapshot

    # Both task heads exist in the saved state dict.
    state_dict_keys = checkpoint["model_state_dict"].keys()
    assert any("same_triangle_head" in key for key in state_dict_keys)
    assert any("distance_head" in key for key in state_dict_keys)
