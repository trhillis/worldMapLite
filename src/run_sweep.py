"""Run the existing TrainConfig pipeline over a controlled sweep grid."""

from __future__ import annotations

import argparse
import json

from src.train_multitask import TrainConfig, main as train


def expand_sweep(config):
    fixed = config.get("fixed", {})
    curve = config.get("learning_curves", {})
    recovery = config.get("held_out_point_recovery", {})
    curve_budgets = set(curve.get("selected_budgets", []))
    recovery_budgets = set(recovery.get("selected_budgets", []))
    runs = []
    for world in config["worlds"]:
        for budget in config["supervision_budgets"]:
            for seed in config["model_seeds"]:
                values = {
                    "train_examples_per_task": budget,
                    "seed": seed,
                    "world_type": world["world_type"],
                    "width": world.get("width", TrainConfig.width),
                    "height": world.get("height", TrainConfig.height),
                    "manifold": world.get("manifold", TrainConfig.manifold),
                    "manifold_points": world.get("manifold_points", TrainConfig.manifold_points),
                    "world_seed": fixed.get("world_seed"),
                    "data_order_seed": fixed.get("data_order_seed"),
                    "pair_split_seed": fixed.get("pair_split_seed"),
                    "val_examples_per_task": fixed.get("eval_pairs", TrainConfig.val_examples_per_task),
                    "steps": fixed.get("steps", TrainConfig.steps),
                    "learning_rate": fixed.get("learning_rate", TrainConfig.learning_rate),
                    "weight_decay": fixed.get("weight_decay", TrainConfig.weight_decay),
                    "batch_size": fixed.get("batch_size", TrainConfig.batch_size),
                    "emb_dim": fixed.get("emb_dim", TrainConfig.emb_dim),
                    "hidden_dim": fixed.get("hidden_dim", TrainConfig.hidden_dim),
                    "results_dir": fixed.get("results_dir", TrainConfig.results_dir),
                }
                if budget in curve_budgets:
                    values["evaluation_checkpoints"] = tuple(curve["checkpoints"])
                runs.append(TrainConfig(**values))
                if budget in recovery_budgets:
                    recovery_values = dict(values)
                    recovery_values.update({
                        "held_out_points": recovery.get("held_out_points", 0),
                        "held_out_point_fraction": recovery.get("held_out_point_fraction"),
                        "held_out_point_seed": recovery.get("held_out_point_seed", 0),
                        "recovery_anchor_counts": tuple(recovery.get("anchor_counts", [])),
                        "recovery_steps": recovery.get("steps", TrainConfig.recovery_steps),
                        "recovery_learning_rate": recovery.get("learning_rate", TrainConfig.recovery_learning_rate),
                        "recovery_seed": recovery.get("seed", 0),
                    })
                    runs.append(TrainConfig(**recovery_values))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    runs = expand_sweep(config)
    print(f"Expanded {len(runs)} independent training runs")
    for run in runs:
        if args.dry_run:
            print(run)
        else:
            train(run)


if __name__ == "__main__":
    main()
