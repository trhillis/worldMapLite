import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# dataclass makes it convenient to define a training-configuration object.
from dataclasses import dataclass
import argparse

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
from src.worlds import make_grid, make_manifold_world
from manifolds.mobius import FlatMobiusStrip
from manifolds.polyhedra import octahedron

# Import task-specific dataset generators and utilities.
from src.datasets import (
    make_disjoint_distance_splits,
    distance_scale,
    make_nearest_examples,
    build_nearest_and_negative_cache,
)

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

    # Number of fixed held-out distance relations used for evaluation.
    val_examples_per_task: int = 5_000

    # Frequency of held-out evaluation during training.
    eval_every: int = 1_000

    # Number of examples processed by each task per training step.
    batch_size: int = 256

    # Number of optimizer updates.
    steps: int = 5000

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
    """Evaluate normalized predictions and raw distance errors."""

    if not examples:
        raise ValueError(
            "Distance evaluation requires at least one example"
        )

    loader = DataLoader(
        PairDataset(examples),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = []
    targets = []
    was_training = model.training

    model.eval()

    try:
        with torch.inference_mode():
            for point_i, point_j, target in loader:
                prediction = model.forward_distance(
                    point_i.to(device),
                    point_j.to(device),
                )
                predictions.append(
                    prediction.detach().cpu()
                )
                targets.append(target)
    finally:
        model.train(was_training)

    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    errors = prediction - target
    normalized_mae = float(
        np.abs(errors).mean()
    )
    normalized_rmse = float(
        np.sqrt(np.square(errors).mean())
    )
    residual_sum = float(
        np.square(errors).sum()
    )
    total_sum = float(
        np.square(target - target.mean()).sum()
    )

    r2 = (
        float(1.0 - residual_sum / total_sum)
        if total_sum > 0.0
        else float("nan")
    )
    spearman = (
        float(spearmanr(target, prediction).statistic)
        if (
            len(target) > 1
            and np.ptp(target) > 0.0
            and np.ptp(prediction) > 0.0
        )
        else float("nan")
    )

    return {
        "n_examples": len(target),
        "normalized_mae": normalized_mae,
        "normalized_rmse": normalized_rmse,
        "mae": normalized_mae * scale,
        "rmse": normalized_rmse * scale,
        "r2": r2,
        "spearman": spearman,
    }


def main(cfg=None):
    # Create a configuration object with the values defined above.
    if cfg is None:
        cfg = TrainConfig()

    if cfg.eval_every <= 0:
        raise ValueError(
            "eval_every must be a positive integer"
        )

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

    # Create the underlying 20-by-20 grid.
    if cfg.world_type == "grid":
        world = make_grid(
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

        else:
            raise ValueError(
                f"Unknown manifold: {cfg.manifold}"
            )

        world = make_manifold_world(
            manifold,
            n=cfg.manifold_points,
            seed=cfg.seed,
            diameter=np.pi,
        )

    else:
        raise ValueError(
            f"Unknown world type: {cfg.world_type}"
        )
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

    if "distance" in cfg.tasks:
        print(
            "Generating fixed held-out and nested training "
            "distance pairs..."
        )

        distance_train, distance_eval = (
            make_disjoint_distance_splits(
                world,
                n_train=cfg.train_examples_per_task,
                n_eval=cfg.val_examples_per_task,
                seed=cfg.distance_pair_seed,
            )
        )

        print(
            f"Using {len(distance_train):,} training pairs and "
            f"{len(distance_eval):,} fixed held-out pairs "
            f"(pair seed {cfg.distance_pair_seed})"
        )

        # Convert the examples into shuffled minibatches.
        distance_train_loader = DataLoader(
            PairDataset(distance_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
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

        if (
            "distance" in cfg.tasks
            and (
                step == 1
                or step % cfg.eval_every == 0
                or step == cfg.steps
            )
        ):
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
            }
            distance_history.append(
                history_entry
            )

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

    # Create the model directory when it does not already exist.
    os.makedirs(
        "models",
        exist_ok=True,
    )

    # Create a filename based on the active task names.
    #
    # Examples:
    #   ("distance",) -> distance
    #   ("nearest",) -> nearest
    #   ("distance", "nearest") -> distance_nearest
    task_name = "_".join(cfg.tasks)

    if cfg.world_type == "grid":
        task_name = (
            f"grid_{task_name}_pairs"
            f"{cfg.train_examples_per_task}_pairseed"
            f"{cfg.distance_pair_seed}_seed{cfg.seed}"
        )
    elif cfg.world_type == "manifold":
        task_name = (
            f"{cfg.manifold}_{task_name}_pairs"
            f"{cfg.train_examples_per_task}_pairseed"
            f"{cfg.distance_pair_seed}_seed{cfg.seed}"
        )

    # Construct the complete output path.
    save_path = (
        f"models/{task_name}_model.pt"
    )

    # Save enough information to reconstruct the trained model later.
    torch.save(
        {
            # Learned weights and biases.
            "model_state_dict": (
                model.state_dict()
            ),

            # Training and architecture settings.
            "config": vars(cfg),

            # Information describing the grid.
            "world_meta": world.meta,

            # Exact relations are saved so the experiment is auditable and
            # held-out evaluation can exclude training pairs later.
            "distance_train_pairs": (
                torch.tensor(
                    [
                        example["indices"]
                        for example in distance_train
                    ],
                    dtype=torch.long,
                )
                if "distance" in cfg.tasks
                else None
            ),

            "distance_eval_pairs": (
                torch.tensor(
                    [
                        example["indices"]
                        for example in distance_eval
                    ],
                    dtype=torch.long,
                )
                if "distance" in cfg.tasks
                else None
            ),

            "evaluation": {
                "distance_final": distance_metrics,
                "distance_history": distance_history,
            },

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
        },
        save_path,
    )

    print(f"Saved model to {save_path}")


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
        "--world-type",
        choices=("grid", "manifold"),
        default=TrainConfig.world_type,
    )
    parser.add_argument(
        "--manifold",
        choices=("mobius", "octahedron"),
        default=TrainConfig.manifold,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        TrainConfig(
            train_examples_per_task=args.distance_pairs,
            distance_pair_seed=args.pair_seed,
            val_examples_per_task=args.eval_pairs,
            eval_every=args.eval_every,
            seed=args.seed,
            steps=args.steps,
            world_type=args.world_type,
            manifold=args.manifold,
        )
    )
