import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# dataclass makes it convenient to define a training-configuration object.
from dataclasses import dataclass
import argparse
import csv
import json

# os is used to create the output directory.
import os

# Python's random module is seeded for reproducibility.
import random

# NumPy is also seeded for reproducibility.
import numpy as np

# PyTorch provides tensors, devices, optimization, and training.
import torch

# Functional loss functions.
import torch.nn.functional as F

# Dataset defines a PyTorch-compatible dataset.
# DataLoader creates shuffled minibatches.
from torch.utils.data import Dataset, DataLoader

# Create the experimental grid world.
from src.worlds import make_grid, make_manifold_world, subset_world
from manifolds.mobius import FlatMobiusStrip
from manifolds.polyhedra import octahedron
from manifolds.flat_torus import FlatTorus

# Import task-specific dataset generators and utilities.
from src.datasets import (
    _make_distance_examples_from_pairs,
    distance_scale,
    make_nearest_examples,
    build_nearest_and_negative_cache,
)
from src.evaluation import (
    evaluate_distance_examples,
    representation_metrics,
)
from src.point_recovery import recover_held_out_points
from src.splits import make_pair_split, make_point_split

# Import the shared-embedding multitask model.
from src.multitask_model import MultiTaskWorldModel
from scipy.stats import spearmanr


@dataclass
class TrainConfig:
    """
    Store every training setting in one object.
    """

    # Select which task heads receive training updates.
    #
    # Distance only:
    #   ("distance",)
    #
    # Nearest only:
    #   ("nearest",)
    #
    # Multitask:
    #   ("distance", "nearest")
    tasks: tuple[str, ...] = (
        "distance",
    )

    # World type to train on
    world_type: str = "grid"

    # Number of sampled manifold points
    manifold_points: int = 400

    # Name of the manifold
    manifold: str = "mobius"

    # Grid width.
    width: int = 20

    # Grid height.
    height: int = 20

    # Number of learned values in each point embedding.
    emb_dim: int = 32

    # Number of units in each task-head hidden layer.
    hidden_dim: int = 128

    # Number of generated distance examples.
    #
    # For nearest, this parameter creates twice as many actual examples,
    # because every iteration creates one positive and one negative example.
    train_examples_per_task: int = 50_000

    # Seed used only to choose and order the unique distance relations.
    # Keeping this fixed makes pair-budget conditions nested and comparable.
    distance_pair_seed: int = 0

    # Alias with clearer experimental meaning. None preserves old configs.
    pair_split_seed: int | None = None

    # Separating this from model seed fixes sampled worlds across model seeds.
    world_seed: int | None = None

    # Fixed shuffle stream for comparable runs; None preserves legacy behavior.
    data_order_seed: int | None = None

    # Number of fixed held-out distance relations used for evaluation.
    val_examples_per_task: int = 5_000

    # Frequency of held-out evaluation during training.
    eval_every: int = 1_000

    # Number of examples processed by each task per training step.
    batch_size: int = 256

    # Number of optimizer updates.
    steps: int = 5000

    # Optional frequency for saving resumable intermediate checkpoints.
    checkpoint_every: int | None = None

    # Exact optimizer updates at which one uninterrupted run is evaluated.
    # None retains the legacy eval_every behavior.
    evaluation_checkpoints: tuple[int, ...] | None = None

    # Optional unseen-entity protocol. Both count and fraction default off.
    held_out_points: int = 0
    held_out_point_fraction: float | None = None
    held_out_point_seed: int = 0
    recovery_anchor_counts: tuple[int, ...] = ()
    recovery_steps: int = 200
    recovery_learning_rate: float = 0.05
    recovery_seed: int = 0

    # Tidy per-run tables are written beneath this directory.
    results_dir: str = "results"

    # AdamW learning rate.
    learning_rate: float = 1e-3

    # Strength of parameter shrinkage used by AdamW.
    weight_decay: float = 1e-4

    # Contribution of the distance loss to total loss.
    distance_weight: float = 1.0

    # Contribution of nearest loss to total loss.
    nearest_weight: float = 1.0

    # Random seed for reproducibility.
    seed: int = 4


class PairDataset(Dataset):
    """
    Convert a list of example dictionaries into PyTorch tensors.
    """

    def __init__(self, examples):
        # Store the first point index from every example.
        self.i = torch.tensor(
            [
                example["indices"][0]
                for example in examples
            ],
            dtype=torch.long,
        )

        # Store the second point index from every example.
        self.j = torch.tensor(
            [
                example["indices"][1]
                for example in examples
            ],
            dtype=torch.long,
        )

        # Store the target value from every example.
        self.y = torch.tensor(
            [
                example["answer"]
                for example in examples
            ],
            dtype=torch.float32,
        )

    def __len__(self):
        # Return the number of examples.
        return len(self.y)

    def __getitem__(self, index):
        # Return one pair and its target.
        return (
            self.i[index],
            self.j[index],
            self.y[index],
        )


def set_seed(seed):
    """
    Seed every random-number generator used by this program.
    """

    # Seed Python randomness.
    random.seed(seed)

    # Seed NumPy randomness.
    np.random.seed(seed)

    # Seed PyTorch CPU randomness.
    torch.manual_seed(seed)

    # Seed every CUDA device when CUDA is available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite_loader(loader):
    """
    Repeatedly iterate over a DataLoader forever.

    When one epoch ends, iteration begins again from a newly shuffled epoch.
    """

    while True:
        # Yield every batch from the current epoch.
        yield from loader


def evaluate_distance(
    model,
    examples,
    scale,
    device,
    batch_size,
):
    """Backward-compatible public wrapper around shared evaluation."""

    return evaluate_distance_examples(model, examples, scale, device, batch_size)


def checkpoint_task_name(cfg):
    """Return the stable experiment identifier used in checkpoint names."""

    task_name = "_".join(cfg.tasks)
    pair_seed = (
        cfg.distance_pair_seed if cfg.pair_split_seed is None else cfg.pair_split_seed
    )

    if cfg.world_type == "grid":
        name = (
            f"grid_{task_name}_pairs"
            f"{cfg.train_examples_per_task}_pairseed"
            f"{pair_seed}_seed{cfg.seed}"
        )
    elif cfg.world_type == "manifold":
        name = (
            f"{cfg.manifold}_{task_name}_pairs"
            f"{cfg.train_examples_per_task}_pairseed"
            f"{pair_seed}_seed{cfg.seed}"
        )
    else:
        raise ValueError(f"Unknown world type: {cfg.world_type}")
    if cfg.held_out_points or cfg.held_out_point_fraction:
        amount = (
            f"frac{cfg.held_out_point_fraction:g}"
            if cfg.held_out_point_fraction is not None
            else str(cfg.held_out_points)
        )
        name += f"_pointholdout{amount}_pointseed{cfg.held_out_point_seed}"
    return name


def save_training_checkpoint(
    model,
    optimizer,
    cfg,
    world,
    distance_train,
    distance_eval,
    distance_history,
    step,
    periodic=False,
    pair_split=None,
    point_split=None,
    recovery=None,
):
    """Save the current training state using the established schema."""

    os.makedirs(
        "models",
        exist_ok=True,
    )

    task_name = checkpoint_task_name(cfg)
    step_suffix = (
        f"_step{step}"
        if periodic
        else ""
    )
    save_path = (
        f"models/{task_name}{step_suffix}_model.pt"
    )
    distance_metrics = (
        distance_history[-1]["held_out"]
        if "distance" in cfg.tasks
        else None
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(cfg),
            "world_meta": world.meta,
            "distance_train_pairs": (
                torch.tensor(
                    pair_split.train_pairs if pair_split is not None else [
                        example["indices"] for example in distance_train
                    ],
                    dtype=torch.long,
                )
                if "distance" in cfg.tasks
                else None
            ),
            "distance_eval_pairs": (
                torch.tensor(
                    pair_split.held_out_pairs if pair_split is not None else [
                        example["indices"] for example in distance_eval
                    ],
                    dtype=torch.long,
                )
                if "distance" in cfg.tasks
                else None
            ),
            "evaluation": {
                "distance_final": distance_metrics,
                "distance_history": distance_history,
                "held_out_point_recovery": recovery,
            },
            "pair_split": (
                {
                    **pair_split.metadata(),
                    "training_pairs": pair_split.train_pairs.tolist(),
                    "held_out_pairs": pair_split.held_out_pairs.tolist(),
                }
                if pair_split is not None else None
            ),
            "point_split": (
                point_split.metadata() if point_split is not None else None
            ),
            "world_coordinates": np.asarray(
                world.coordinates
            ),
            "ambient_coordinates": (
                np.asarray(
                    world.ambient_coordinates
                )
                if world.ambient_coordinates is not None
                else None
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_step": step,
        },
        save_path,
    )

    print(f"Saved model to {save_path}")

    return save_path


def _remap_examples(examples, original_to_local):
    remapped = []
    for example in examples:
        copied = dict(example)
        copied["indices"] = tuple(
            original_to_local[int(point)] for point in example["indices"]
        )
        remapped.append(copied)
    return remapped


def _flatten_evaluation_row(cfg, world_name, step, train_metrics, held_out_metrics, rep_metrics, pair_split, point_split):
    row = {
        "world": world_name,
        "supervision_budget": cfg.train_examples_per_task,
        "checkpoint": step,
        "optimizer_updates": step,
        "model_seed": cfg.seed,
        "world_seed": cfg.world_seed if cfg.world_seed is not None else cfg.seed,
        "data_order_seed": cfg.data_order_seed if cfg.data_order_seed is not None else cfg.seed,
        "pair_split_seed": pair_split.seed,
        "held_out_point_seed": point_split.seed,
        "held_out_point_count": len(point_split.held_out_points),
        "evaluation_protocol": (
            "held_out_points" if len(point_split.held_out_points) else "held_out_pairs"
        ),
        "recovery_anchor_count": None,
        "pair_split_digest": pair_split.digest,
        "held_out_pair_digest": pair_split.held_out_digest,
        "point_split_digest": point_split.digest,
    }
    row.update({f"training_pair_{key}": value for key, value in train_metrics.items()})
    row.update({f"held_out_pair_{key}": value for key, value in held_out_metrics.items()})
    row.update(rep_metrics)
    return row


def _write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(cfg=None):
    # Create a configuration object with the values defined above.
    if cfg is None:
        cfg = TrainConfig()

    if cfg.eval_every <= 0:
        raise ValueError(
            "eval_every must be a positive integer"
        )

    if (
        cfg.checkpoint_every is not None
        and cfg.checkpoint_every <= 0
    ):
        raise ValueError(
            "checkpoint_every must be a positive integer or None"
        )

    if cfg.steps <= 0:
        raise ValueError("steps must be a positive integer")
    if cfg.evaluation_checkpoints is not None:
        evaluation_steps = tuple(sorted(set(cfg.evaluation_checkpoints)))
        if not evaluation_steps or evaluation_steps[0] <= 0 or evaluation_steps[-1] > cfg.steps:
            raise ValueError("evaluation_checkpoints must be positive and no greater than steps")
    else:
        evaluation_steps = None
    if cfg.recovery_anchor_counts and not (cfg.held_out_points or cfg.held_out_point_fraction):
        raise ValueError("recovery anchors require held-out points")
    if cfg.held_out_points and cfg.held_out_point_fraction is not None:
        raise ValueError("set only one of held_out_points or held_out_point_fraction")
    if cfg.recovery_anchor_counts and (
        cfg.recovery_steps <= 0
        or cfg.recovery_learning_rate <= 0
        or any(count <= 0 for count in cfg.recovery_anchor_counts)
    ):
        raise ValueError("recovery steps, learning rate, and anchor counts must be positive")
    if (cfg.held_out_points or cfg.held_out_point_fraction) and tuple(cfg.tasks) != ("distance",):
        raise ValueError("held-out-point recovery currently supports distance-only base training")

    pair_seed = cfg.distance_pair_seed if cfg.pair_split_seed is None else cfg.pair_split_seed
    world_seed = cfg.seed if cfg.world_seed is None else cfg.world_seed

    # Make the run reproducible.
    set_seed(cfg.seed)

    # Use the GPU when CUDA is available.
    # Otherwise use the CPU.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Build the full world once. A compact retained-point world is derived
    # below when whole entities are held out.
    if cfg.world_type == "grid":
        full_world = make_grid(
            cfg.width,
            cfg.height,
        )

    elif cfg.world_type == "manifold":

        if cfg.manifold == "mobius":
            manifold = FlatMobiusStrip()

        #elif cfg.manifold == "torus":
        #    manifold = FlatTorus()

        elif cfg.manifold == "octahedron":
           manifold = octahedron()

        elif cfg.manifold == "torus":
            manifold = FlatTorus()

        else:
            raise ValueError(
                f"Unknown manifold: {cfg.manifold}"
            )

        full_world = make_manifold_world(
            manifold,
            n=cfg.manifold_points,
            seed=world_seed,
            diameter=np.pi,
        )

    else:
        raise ValueError(
            f"Unknown world type: {cfg.world_type}"
        )

    point_split = make_point_split(
        len(full_world.names),
        n_held_out=(None if cfg.held_out_point_fraction is not None else cfg.held_out_points),
        held_out_fraction=cfg.held_out_point_fraction,
        seed=cfg.held_out_point_seed,
    )
    world = subset_world(full_world, point_split.retained_points)
    # These are only necessary when training the nearest task.
    nearest_cache = None
    negative_cache = None

    if "nearest" in cfg.tasks:
        print("Building nearest-neighbor cache...")

        # Precompute positive and negative nearest candidates.
        nearest_cache, negative_cache = (
            build_nearest_and_negative_cache(world)
        )

    # These variables remain None when their corresponding task
    # is not being trained.
    distance_iterator = None
    nearest_iterator = None
    distance_train = None
    distance_eval = None
    pair_split = None

    if "distance" in cfg.tasks:
        print(
            "Generating fixed held-out and nested training "
            "distance pairs..."
        )

        pair_split = make_pair_split(
            len(full_world.names),
            n_train=cfg.train_examples_per_task,
            n_held_out=cfg.val_examples_per_task,
            seed=pair_seed,
            excluded_points=point_split.held_out_points,
        )
        original_train = _make_distance_examples_from_pairs(
            full_world, pair_split.train_pairs,
        )
        original_eval = _make_distance_examples_from_pairs(
            full_world, pair_split.held_out_pairs,
        )
        original_to_local = {
            int(original): local
            for local, original in enumerate(point_split.retained_points)
        }
        distance_train = _remap_examples(original_train, original_to_local)
        distance_eval = _remap_examples(original_eval, original_to_local)

        print(
            f"Using {len(distance_train):,} training pairs and "
            f"{len(distance_eval):,} fixed held-out pairs "
            f"(pair seed {pair_seed}, split {pair_split.digest[:12]})"
        )

        # Convert the examples into shuffled minibatches.
        distance_train_loader = DataLoader(
            PairDataset(distance_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(
                cfg.seed if cfg.data_order_seed is None else cfg.data_order_seed
            ),
        )

        # Turn the DataLoader into an endless stream of batches.
        distance_iterator = infinite_loader(
            distance_train_loader
        )

    if "nearest" in cfg.tasks:
        print("Generating nearest training examples...")

        # Generate balanced positive and negative nearest examples.
        #
        # Pass the caches here so they are not rebuilt.
        nearest_train = make_nearest_examples(
            world,
            n=cfg.train_examples_per_task,
            seed=cfg.seed + 1,
            nearest_cache=nearest_cache,
            negative_cache=negative_cache,
        )

        # Convert the examples into shuffled minibatches.
        nearest_train_loader = DataLoader(
            PairDataset(nearest_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(
                cfg.seed if cfg.data_order_seed is None else cfg.data_order_seed
            ),
        )

        # Turn the DataLoader into an endless stream of batches.
        nearest_iterator = infinite_loader(
            nearest_train_loader
        )

    # Create one model containing:
    #   one shared embedding table
    #   one distance head
    #   one nearest head
    model = MultiTaskWorldModel(
        num_points=len(world.names),
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    # AdamW updates model parameters using calculated gradients.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    distance_history = []
    learning_curve_rows = []
    world_name = "grid" if cfg.world_type == "grid" else cfg.manifold

    # Perform the requested number of optimizer updates.
    for step in range(1, cfg.steps + 1):
        # Enable training behavior.
        model.train()

        # Store whichever task losses are active during this step.
        losses = {}

        if "distance" in cfg.tasks:
            # Read one distance minibatch.
            di, dj, dy = next(distance_iterator)

            # Move point indices and labels onto the selected device.
            di = di.to(device)
            dj = dj.to(device)
            dy = dy.to(device)

            # Predict normalized distances.
            distance_prediction = (
                model.forward_distance(di, dj)
            )

            # Smooth L1 is similar to squared error for small mistakes
            # but is less sensitive to large errors.
            losses["distance"] = F.smooth_l1_loss(
                distance_prediction,
                dy,
            )

        if "nearest" in cfg.tasks:
            # Read one nearest-neighbor minibatch.
            ni, nj, ny = next(nearest_iterator)

            # Move point indices and labels onto the selected device.
            ni = ni.to(device)
            nj = nj.to(device)
            ny = ny.to(device)

            # Predict raw binary-classification logits.
            nearest_logits = model.forward_nearest(
                ni,
                nj,
            )

            # Compare logits against binary labels.
            #
            # This function includes the sigmoid operation internally.
            losses["nearest"] = (
                F.binary_cross_entropy_with_logits(
                    nearest_logits,
                    ny,
                )
            )

        # Begin total loss as a scalar zero on the correct device.
        loss = torch.zeros(
            (),
            device=device,
        )

        # Add distance loss when distance is active.
        if "distance" in losses:
            loss = (
                loss
                + cfg.distance_weight
                * losses["distance"]
            )

        # Add nearest loss when nearest is active.
        if "nearest" in losses:
            loss = (
                loss
                + cfg.nearest_weight
                * losses["nearest"]
            )

        # Delete gradients left over from the previous step.
        optimizer.zero_grad(
            set_to_none=True
        )

        # Compute gradients for every trained parameter.
        loss.backward()

        # Update the model parameters.
        optimizer.step()

        should_evaluate = (
            step in evaluation_steps
            if evaluation_steps is not None
            else (step == 1 or step % cfg.eval_every == 0 or step == cfg.steps)
        )
        if "distance" in cfg.tasks and should_evaluate:
            scale = distance_scale(world)
            train_metrics = evaluate_distance(
                model=model,
                examples=distance_train,
                scale=scale,
                device=device,
                batch_size=cfg.batch_size,
            )
            held_out_metrics = evaluate_distance(
                model=model,
                examples=distance_eval,
                scale=scale,
                device=device,
                batch_size=cfg.batch_size,
            )
            history_entry = {
                "step": step,
                "train": train_metrics,
                "held_out": held_out_metrics,
                "representation": representation_metrics(
                    model, world, seed=cfg.seed,
                ),
            }
            distance_history.append(
                history_entry
            )
            learning_curve_rows.append(_flatten_evaluation_row(
                cfg, world_name, step, train_metrics, held_out_metrics,
                history_entry["representation"], pair_split, point_split,
            ))

            print(
                f"eval step={step:6d} "
                f"train_MAE={train_metrics['mae']:.4f} "
                f"held_out_MAE={held_out_metrics['mae']:.4f} "
                f"held_out_R²={held_out_metrics['r2']:.4f} "
                f"held_out_Spearman="
                f"{held_out_metrics['spearman']:.4f}"
            )

        # Print progress every 250 steps and on the first step.
        if step % 250 == 0 or step == 1:
            # Begin the printed message with step and total loss.
            parts = [
                f"step={step:5d}",
                f"loss={loss.item():.5f}",
            ]

            # Add each active task loss.
            for task_name, task_loss in losses.items():
                parts.append(
                    f"{task_name}_loss="
                    f"{task_loss.item():.5f}"
                )

            # Join all message components with spaces.
            print(" ".join(parts))

        if (
            (
                (cfg.checkpoint_every is not None and step % cfg.checkpoint_every == 0)
                or (evaluation_steps is not None and step in evaluation_steps)
            )
            and step != cfg.steps
        ):
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                world=world,
                distance_train=distance_train,
                distance_eval=distance_eval,
                distance_history=distance_history,
                step=step,
                periodic=True,
                pair_split=pair_split,
                point_split=point_split,
            )

    distance_metrics = (
        distance_history[-1]["held_out"]
        if "distance" in cfg.tasks
        else None
    )

    if distance_metrics is not None:
        print("\nHeld-out distance evaluation")
        print(
            f"examples={distance_metrics['n_examples']}"
        )
        print(
            f"MAE={distance_metrics['mae']:.4f}"
        )
        print(
            f"RMSE={distance_metrics['rmse']:.4f}"
        )
        print(
            f"R²={distance_metrics['r2']:.4f}"
        )
        print(
            f"Spearman={distance_metrics['spearman']:.4f}"
        )

    run_name = checkpoint_task_name(cfg)
    learning_curve_path = os.path.join(
        cfg.results_dir, "learning_curves", f"{run_name}.csv",
    )
    _write_rows(learning_curve_path, learning_curve_rows)

    recovery_results = None
    recovery_rows = []
    if len(point_split.held_out_points) and cfg.recovery_anchor_counts:
        recovery_results = {}
        for anchor_count in cfg.recovery_anchor_counts:
            result = recover_held_out_points(
                model=model,
                full_world=full_world,
                retained_point_ids=point_split.retained_points,
                held_out_point_ids=point_split.held_out_points,
                anchor_count=anchor_count,
                steps=cfg.recovery_steps,
                learning_rate=cfg.recovery_learning_rate,
                seed=cfg.recovery_seed,
                device=device,
            )
            recovery_results[str(anchor_count)] = result
            for row in result["rows"]:
                recovery_rows.append({
                    "world": world_name,
                    "supervision_budget": cfg.train_examples_per_task,
                    "checkpoint": cfg.steps,
                    "optimizer_updates": cfg.steps,
                    "model_seed": cfg.seed,
                    "world_seed": world_seed,
                    "data_order_seed": cfg.data_order_seed if cfg.data_order_seed is not None else cfg.seed,
                    "pair_split_seed": pair_seed,
                    "held_out_point_seed": cfg.held_out_point_seed,
                    "held_out_point_count": len(point_split.held_out_points),
                    "evaluation_protocol": "held_out_points",
                    "pair_split_digest": pair_split.digest,
                    "held_out_pair_digest": pair_split.held_out_digest,
                    "point_split_digest": point_split.digest,
                    **row,
                })
        recovery_path = os.path.join(
            cfg.results_dir, "recovery", f"{run_name}.csv",
        )
        _write_rows(recovery_path, recovery_rows)
        metadata_path = os.path.join(
            cfg.results_dir, "recovery", f"{run_name}.json",
        )
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(recovery_results, handle, indent=2, allow_nan=True)

    checkpoint_path = save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        world=world,
        distance_train=distance_train,
        distance_eval=distance_eval,
        distance_history=distance_history,
        step=cfg.steps,
        pair_split=pair_split,
        point_split=point_split,
        recovery=recovery_results,
    )
    return {
        "checkpoint": checkpoint_path,
        "learning_curve": learning_curve_path,
        "recovery_rows": recovery_rows,
    }


# Run main only when this file is executed directly.
#
# It will not run automatically if this file is imported elsewhere.
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a world model with a controlled number of unique "
            "pairwise-distance relations."
        )
    )

    parser.add_argument(
        "--distance-pairs",
        type=int,
        default=TrainConfig.train_examples_per_task,
        help="Number of unique unordered distance pairs used for training.",
    )
    parser.add_argument(
        "--pair-seed",
        type=int,
        default=TrainConfig.distance_pair_seed,
        help="Seed for the shared shuffled ordering of distance pairs.",
    )
    parser.add_argument(
        "--pair-split-seed",
        type=int,
        default=None,
        help="Dedicated split seed; overrides the backward-compatible --pair-seed.",
    )
    parser.add_argument(
        "--world-seed",
        type=int,
        default=None,
        help="World sampling seed. Fix this while varying --seed.",
    )
    parser.add_argument(
        "--data-order-seed",
        type=int,
        default=None,
        help="Dedicated shuffled minibatch-order seed; fix across comparable runs.",
    )
    parser.add_argument(
        "--eval-pairs",
        type=int,
        default=TrainConfig.val_examples_per_task,
        help=(
            "Number of pairs reserved as a fixed held-out prefix before "
            "the nested training-pair prefixes."
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=TrainConfig.eval_every,
        help="Run train and held-out distance evaluation every N steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TrainConfig.seed,
        help="Model/world random seed.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=TrainConfig.steps,
        help="Number of optimizer updates.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=TrainConfig.checkpoint_every,
        help=(
            "Save an intermediate checkpoint every N optimizer steps. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--evaluation-checkpoints",
        type=int,
        nargs="+",
        default=None,
        help="Exact optimizer updates to evaluate and checkpoint in one run.",
    )
    parser.add_argument("--held-out-points", type=int, default=0)
    parser.add_argument("--held-out-point-fraction", type=float, default=None)
    parser.add_argument("--held-out-point-seed", type=int, default=0)
    parser.add_argument("--recovery-anchor-counts", type=int, nargs="+", default=())
    parser.add_argument("--recovery-steps", type=int, default=TrainConfig.recovery_steps)
    parser.add_argument("--recovery-learning-rate", type=float, default=TrainConfig.recovery_learning_rate)
    parser.add_argument("--recovery-seed", type=int, default=TrainConfig.recovery_seed)
    parser.add_argument("--results-dir", default=TrainConfig.results_dir)
    parser.add_argument("--width", type=int, default=TrainConfig.width)
    parser.add_argument("--height", type=int, default=TrainConfig.height)
    parser.add_argument("--manifold-points", type=int, default=TrainConfig.manifold_points)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument(
        "--world-type",
        choices=("grid", "manifold"),
        default=TrainConfig.world_type,
    )
    parser.add_argument(
        "--manifold",
        choices=("mobius", "octahedron", "torus"),
        default=TrainConfig.manifold,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        TrainConfig(
            train_examples_per_task=args.distance_pairs,
            distance_pair_seed=args.pair_seed,
            pair_split_seed=args.pair_split_seed,
            world_seed=args.world_seed,
            data_order_seed=args.data_order_seed,
            val_examples_per_task=args.eval_pairs,
            eval_every=args.eval_every,
            seed=args.seed,
            steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            evaluation_checkpoints=(
                tuple(args.evaluation_checkpoints)
                if args.evaluation_checkpoints is not None else None
            ),
            held_out_points=args.held_out_points,
            held_out_point_fraction=args.held_out_point_fraction,
            held_out_point_seed=args.held_out_point_seed,
            recovery_anchor_counts=tuple(args.recovery_anchor_counts),
            recovery_steps=args.recovery_steps,
            recovery_learning_rate=args.recovery_learning_rate,
            recovery_seed=args.recovery_seed,
            results_dir=args.results_dir,
            world_type=args.world_type,
            manifold=args.manifold,
            width=args.width,
            height=args.height,
            manifold_points=args.manifold_points,
            batch_size=args.batch_size,
        )
    )
