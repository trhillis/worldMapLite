"""
Train a multi-task transformer on a manifolds.Manifold surface.

Usage:
    conda activate worldMap
    python -m src.train_manifold [config_path]

Example:
    conda activate worldMap
    python -m src.train_manifold configs/octahedron_default.yaml

The set of trained tasks is config-driven (training.tasks, default
["distance"] - see TASK_SPECS below for what's available). See
configs/octahedron_same_triangle.yaml (same_triangle alone) and
configs/octahedron_distance_same_triangle.yaml (distance + same_triangle
together) for multi-task examples.

A run can also warm-start from an existing checkpoint (training.
init_checkpoint) and freeze whole submodules by attribute name
(training.freeze, e.g. ["emb", "transformer"]), to fine-tune only a new
task's token/head on top of a pretrained, frozen representation - see
configs/octahedron_same_triangle_frozen_ft.yaml. training.
finetune_pool_points additionally restricts which points that new task's
training pairs may touch (test_pairs stays drawn from every point), to
probe how few points' worth of supervision the new head needs before it
generalizes to the whole point set - see
configs/same_triangle_points_sweep/.
"""

# argparse reads the config-path command-line argument.
import argparse

# os is used to create the output directory.
import os

# shutil copies the resolved config next to the saved checkpoint.
import shutil

# NumPy is used to cast world coordinates for the periodic progress snapshots.
import numpy as np

# PyTorch provides tensors, devices, optimization, and training.
import torch

# Functional loss functions.
import torch.nn.functional as F

# DataLoader creates shuffled minibatches.
from torch.utils.data import DataLoader

# YAML config loading.
import yaml

# Manifold registry: turns a config string like "octahedron" into a
# manifolds.Manifold instance.
from manifolds import get_manifold

# Build a fixed-entity World out of a manifold's sampled surface points.
from src.worlds import make_manifold_world

# Cheap, transformer-forward-free representation-quality metrics, reused
# here to take periodic snapshots of the embedding table during training
# (see analysis/analysis_manifold.py, which uses the same metrics on the
# final checkpoint).
from analysis.representations import (
    compute_embedding_metrics,
    pairwise_distances,
    upper_triangle_values,
)

# Task-specific dataset generators and generic training plumbing.
from src.datasets import (
    make_distance_examples,
    make_distance_examples_from_pairs,
    make_same_triangle_examples,
    make_same_triangle_examples_from_pairs,
    split_pairs,
    split_points,
    make_holdout_point_pairs,
    points_on_faces,
    PairDataset,
    set_seed,
    infinite_loader,
)

# Shared evaluation helpers - live outside this module so
# src/holdout_probe.py can reuse them without importing this module in turn.
from src.eval_utils import (
    evaluate_distance_examples,
    evaluate_same_triangle_examples,
)

# Recovers held-out points' embeddings from a few probe distances and
# checks whether they generalize - the point-level counterpart to the
# pair-level held-out check below.
from src.holdout_probe import recover_and_evaluate_holdout_points

# The shared-embedding transformer, reused unchanged from the grid-world
# pipeline: a plain nn.Module, no HuggingFace Trainer/PreTrainedModel wrapping.
from src.multitask_model import MultiTaskWorldModel


# Task name -> everything main() needs to train and evaluate that task.
#
#   make_examples             sample n labeled examples (optionally
#                              restricted to allowed_pairs)
#   make_examples_from_pairs  build exactly one example per given pair
#   evaluate                  run the task head over a fixed example set
#                              and summarize loss + task-specific metrics
#   forward                   (model, i, j) -> raw prediction/logit
#   loss                      (prediction, target) -> scalar
#   weight_key                training config key for this task's loss
#                              weight (training_cfg.get(weight_key, 1.0))
TASK_SPECS = {
    "distance": {
        "make_examples": make_distance_examples,
        "make_examples_from_pairs": make_distance_examples_from_pairs,
        "evaluate": evaluate_distance_examples,
        "forward": lambda model, i, j: model.forward_distance(i, j),
        "loss": F.smooth_l1_loss,
        "weight_key": "distance_weight",
    },
    "same_triangle": {
        "make_examples": make_same_triangle_examples,
        "make_examples_from_pairs": make_same_triangle_examples_from_pairs,
        "evaluate": evaluate_same_triangle_examples,
        "forward": lambda model, i, j: model.forward_same_triangle(i, j),
        "loss": F.binary_cross_entropy_with_logits,
        "weight_key": "same_triangle_weight",
    },
}


def load_config(config_path: str) -> dict:
    """
    Load a nested YAML config with `manifold` / `model` / `training` sections.
    """

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    # Load the full nested config (manifold / model / training settings).
    config = load_config(config_path)

    manifold_cfg = config["manifold"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    # Which tasks to train, in order. Defaults to distance-only, so every
    # config predating this feature behaves exactly as before.
    tasks = training_cfg.get("tasks", ["distance"])

    # Make the run reproducible.
    set_seed(training_cfg["seed"])

    # Use the GPU when CUDA is available.
    # Otherwise use the CPU.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Training tasks: {tasks}")

    # Build the manifold and sample its fixed set of entity points.
    #
    # This is the manifold equivalent of Park et al.'s 5,075 cities: a
    # reproducible, seeded point set whose indices become embedding-table
    # indices. Pairwise geodesic distances are precomputed once and cached
    # to disk by make_manifold_world (see src/worlds.py).
    manifold = get_manifold(manifold_cfg["name"])

    print(
        f"Sampling {manifold_cfg['n_points']} points on "
        f"'{manifold_cfg['name']}' (this precomputes the full pairwise "
        "geodesic distance matrix on first run; cached afterward)..."
    )

    world = make_manifold_world(
        manifold,
        n=manifold_cfg["n_points"],
        seed=manifold_cfg["seed"],
    )

    # Point-level holdout generalization test (opt-in, off by default):
    # entire points, not just pairs, can be withheld from every training
    # pair, then have their embeddings recovered afterward from a
    # handful of probe distances (src/holdout_probe.py). This is a
    # stronger check than the pair-level split below - it tests whether
    # the model's learned geometry would place a genuinely new point
    # correctly, not just whether it generalizes to unseen pairs among
    # points it already has fully-trained embeddings for. Recovery is
    # always distance-only (see src/holdout_probe.py); when other tasks
    # are also active, the recovered point's embedding is additionally
    # evaluated on them below, to check whether distance-only recovery
    # generalizes to a task it was never directly probed on.
    n_holdout_points = training_cfg.get("n_holdout_points", 0)
    n_probes = training_cfg.get("n_probes", 8)
    probe_steps = training_cfg.get("probe_steps", 200)
    probe_lr = training_cfg.get("probe_lr", 0.05)
    # Defaults to the main run's own weight_decay, so probing really is
    # "fine-tune the new point the same way training fine-tunes
    # everything else," just scoped to one (or a few) rows.
    probe_weight_decay = training_cfg.get(
        "probe_weight_decay", training_cfg["weight_decay"],
    )
    # Optional list of polyhedron face indices (octahedron/icosahedron
    # only - see src/datasets.py's points_on_faces): when given, holdout
    # points are restricted to those faces instead of drawn uniformly at
    # random over every point, so recovered points can be visually
    # checked against their true local neighborhood.
    holdout_faces = training_cfg.get("holdout_faces")

    holdout_points = None

    if n_holdout_points > 0:
        candidate_points = (
            points_on_faces(world, holdout_faces)
            if holdout_faces else None
        )

        _, holdout_points = split_points(
            num_points=len(world.names),
            n_holdout=n_holdout_points,
            seed=training_cfg["seed"],
            candidate_points=candidate_points,
        )

    # Reserve a fraction of all possible pairs as a held-out generalization
    # check, before any training pairs are sampled. train_examples is then
    # drawn only from the remaining pool, so it stays exactly the size the
    # config asks for - only the pool it's drawn from shrinks. Holdout
    # points (if any) are excluded here too, so the pair-level check never
    # touches a point that is being held out entirely. This split is
    # shared, unchanged, across every active task, so pair-level held-out
    # results are directly comparable between tasks.
    test_fraction = training_cfg.get("test_fraction", 0.1)

    train_pairs, test_pairs = split_pairs(
        num_points=len(world.names),
        test_fraction=test_fraction,
        seed=training_cfg["seed"],
        exclude_points=holdout_points,
    )

    # Optional: restrict which points training pairs may be drawn from,
    # independent of the pair-level test_fraction split above - meant for
    # probing how few points' worth of a new task's supervision (e.g. a
    # frozen-embedding same_triangle head fine-tune) a model needs before
    # it generalizes to the *entire* point set. test_pairs is left
    # untouched (still drawn from all len(world.names) points), so it
    # stays a fixed, sweep-comparable check of whole-point-set accuracy
    # no matter how small the training pool gets. The pool is a prefix of
    # one fixed seeded permutation of every point, so a smaller pool
    # (e.g. 10 points) is always a subset of a larger one (e.g. 20
    # points) at the same seed, making pool-size comparisons additive
    # rather than independently-noisy draws.
    finetune_pool_points = training_cfg.get("finetune_pool_points")

    if finetune_pool_points is not None:
        point_order = np.random.default_rng(
            training_cfg["seed"]
        ).permutation(len(world.names))
        pool_points = set(point_order[:finetune_pool_points].tolist())

        train_pairs = np.array(
            [
                (i, j) for i, j in train_pairs
                if i in pool_points and j in pool_points
            ],
            dtype=np.int64,
        )

        print(
            f"Restricting training pairs to a {finetune_pool_points}-point "
            f"pool: {len(train_pairs)} eligible pairs "
            f"(test_pairs unchanged: {len(test_pairs)} pairs drawn from "
            f"all {len(world.names)} points)."
        )

    # Each holdout point's probe pairs (used to recover its embedding
    # after training) and eval pairs (used to check how well the
    # recovered embedding generalizes) - built now, before training,
    # purely to fail fast on a bad n_probes/n_holdout_points combination
    # rather than after a long training run.
    holdout_point_pairs = None

    if n_holdout_points > 0:
        holdout_point_pairs = make_holdout_point_pairs(
            num_points=len(world.names),
            holdout_points=holdout_points,
            n_probes=n_probes,
            seed=training_cfg["seed"],
        )

    # Per-task training iterators and held-out/in-distribution example
    # sets, built once and reused both by the periodic snapshots below
    # and the final evaluation after training.
    train_iterators = {}
    held_out_examples = {}
    in_distribution_examples = {}

    for task in tasks:
        spec = TASK_SPECS[task]

        print(f"Generating {task} training examples...")

        # Generate labeled point pairs, restricted to the train pool so
        # test_pairs are never seen during training.
        train_examples = spec["make_examples"](
            world,
            n=training_cfg["train_examples"],
            seed=training_cfg["seed"],
            allowed_pairs=train_pairs,
        )

        train_loader = DataLoader(
            PairDataset(train_examples),
            batch_size=training_cfg["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        train_iterators[task] = infinite_loader(train_loader)

        # Every reserved test pair, evaluated exactly once, plus a
        # same-size fresh independent draw (not the literal training
        # examples) from the train pool, so the only difference between
        # the two sets is pool membership, not "was this exact example
        # seen during training."
        held_out_examples[task] = spec["make_examples_from_pairs"](
            world, test_pairs,
        )
        in_distribution_examples[task] = spec["make_examples"](
            world,
            n=len(test_pairs),
            seed=training_cfg["seed"] + 1,
            allowed_pairs=train_pairs,
        )

    # Create the transformer: shared entity embedding table, one task
    # token + output head per active task, one shared TransformerEncoder.
    model = MultiTaskWorldModel(
        num_points=len(world.names),
        **model_cfg,
    ).to(device)

    # Optionally warm-start from an existing checkpoint (e.g. a
    # distance-trained model, to fine-tune a different task's token/head
    # on top of its embedding) instead of the random init above. The
    # checkpoint must come from a model with identical model_cfg and the
    # same (manifold_name, n_points, seed) - MultiTaskWorldModel always
    # instantiates every task's token/head regardless of which tasks were
    # trained, so state_dict shapes always match across task combinations
    # at fixed model_cfg.
    init_checkpoint = training_cfg.get("init_checkpoint")

    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location=device)
        checkpoint_world_meta = checkpoint["world_meta"]

        if (
            checkpoint_world_meta["manifold_name"] != manifold_cfg["name"]
            or checkpoint_world_meta["n"] != manifold_cfg["n_points"]
            or checkpoint_world_meta["seed"] != manifold_cfg["seed"]
        ):
            raise ValueError(
                f"init_checkpoint {init_checkpoint!r} was trained on "
                f"{checkpoint_world_meta}, which does not match this "
                f"run's manifold {manifold_cfg!r} - embedding-table rows "
                "would refer to different points."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized model from checkpoint: {init_checkpoint}")

    # Optionally freeze whole submodules (by attribute name, e.g. "emb",
    # "transformer") so only the remaining parameters - typically a new
    # task's token/head - receive gradient updates. Meant to pair with
    # init_checkpoint above: freeze the pretrained submodules, fine-tune
    # only what's task-specific on top of them.
    frozen = training_cfg.get("freeze", [])

    for name in frozen:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if frozen:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"Frozen submodules: {frozen} "
            f"({n_trainable}/{n_total} parameters trainable)"
        )

    # AdamW updates model parameters using calculated gradients. Only
    # parameters with requires_grad=True are included, so frozen
    # submodules (if any) never get an optimizer state or update.
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    # How often (in steps) to snapshot embedding-quality metrics during
    # training. 0 (or absent) disables snapshotting entirely.
    progress_interval = training_cfg.get("progress_interval", 0)

    # True ambient coordinates and their pairwise-distance upper triangle
    # don't depend on the model, so they're computed once and reused by
    # every snapshot instead of once per snapshot.
    #
    # Holdout points (if any) sit at random, never-trained init for the
    # entire duration of training - recovery only happens once, after
    # the last step - so they're excluded from every progress snapshot
    # below (unlike analysis/analysis_manifold.py, which runs on the
    # final, post-recovery checkpoint and can meaningfully include them).
    metrics_mask = None

    if holdout_points is not None:
        metrics_mask = np.ones(len(world.names), dtype=bool)
        metrics_mask[holdout_points] = False

    true_coordinates = world.coordinates.astype(np.float64)
    true_coordinates_for_metrics = (
        true_coordinates if metrics_mask is None
        else true_coordinates[metrics_mask]
    )
    true_distance_upper = upper_triangle_values(
        pairwise_distances(true_coordinates_for_metrics)
    )

    # One row per snapshot: step, total loss, embedding-quality metrics,
    # and every active task's held-out/in-distribution evaluation
    # metrics (prefixed "{task}_held_out_{metric}" /
    # "{task}_in_distribution_{metric}"). Saved into the checkpoint below.
    progress_rows = []

    # Perform the requested number of optimizer updates.
    for step in range(1, training_cfg["steps"] + 1):
        # Enable training behavior.
        model.train()

        # Sum every active task's weighted loss into one scalar for
        # .backward(), mirroring legacy/train_multitask.py.
        total_loss = torch.zeros((), device=device)
        step_losses = {}

        for task in tasks:
            spec = TASK_SPECS[task]

            # Read one minibatch for this task.
            ti, tj, ty = next(train_iterators[task])

            # Move point indices and labels onto the selected device.
            ti = ti.to(device)
            tj = tj.to(device)
            ty = ty.to(device)

            prediction = spec["forward"](model, ti, tj)
            task_loss = spec["loss"](prediction, ty)
            step_losses[task] = task_loss

            weight = training_cfg.get(spec["weight_key"], 1.0)
            total_loss = total_loss + weight * task_loss

        # Delete gradients left over from the previous step.
        optimizer.zero_grad(set_to_none=True)

        # Compute gradients for every trained parameter.
        total_loss.backward()

        # Update the model parameters.
        optimizer.step()

        # Print progress every 250 steps and on the first step.
        if step % 250 == 0 or step == 1:
            parts = [f"step={step:5d}", f"loss={total_loss.item():.5f}"]

            for task, task_loss in step_losses.items():
                parts.append(f"{task}_loss={task_loss.item():.5f}")

            print(" ".join(parts))

        # Snapshot embedding-quality metrics every progress_interval
        # steps (plus the first step, as a before-training baseline).
        # The embedding-table metrics only read model.emb.weight, no
        # transformer forward pass, so those are cheap even at this
        # cadence; the per-task held-out/in-distribution eval below does
        # need a forward pass over ~2*len(test_pairs) pairs per task, but
        # that's still cheap for this tiny model, and it's what lets
        # training_curves.png show whether the held-out generalization
        # gap opens up (or stays near zero) over the course of training,
        # not just at the end.
        if progress_interval and (step == 1 or step % progress_interval == 0):
            model.eval()

            with torch.no_grad():
                embeddings_for_metrics = model.emb.weight.detach().cpu().numpy()

                if metrics_mask is not None:
                    embeddings_for_metrics = embeddings_for_metrics[metrics_mask]

                metrics = compute_embedding_metrics(
                    embeddings_for_metrics,
                    true_coordinates_for_metrics,
                    true_distance_upper=true_distance_upper,
                    seed=training_cfg["seed"],
                )

            metrics["step"] = step
            metrics["loss"] = total_loss.item()

            for task in tasks:
                spec = TASK_SPECS[task]

                held_out_progress = spec["evaluate"](
                    model, held_out_examples[task], device,
                    training_cfg["batch_size"],
                )
                in_distribution_progress = spec["evaluate"](
                    model, in_distribution_examples[task], device,
                    training_cfg["batch_size"],
                )

                for key, value in held_out_progress.items():
                    if key == "n_pairs":
                        continue
                    metrics[f"{task}_held_out_{key}"] = value

                for key, value in in_distribution_progress.items():
                    if key == "n_pairs":
                        continue
                    metrics[f"{task}_in_distribution_{key}"] = value

            model.train()

            progress_rows.append(metrics)

    # Final held-out generalization check, on the same example sets used
    # by the periodic snapshots above (see their construction earlier),
    # once per active task.
    held_out_eval = {}

    for task in tasks:
        spec = TASK_SPECS[task]

        held_out_result = spec["evaluate"](
            model, held_out_examples[task], device, training_cfg["batch_size"],
        )
        in_distribution_result = spec["evaluate"](
            model, in_distribution_examples[task], device,
            training_cfg["batch_size"],
        )

        held_out_eval[task] = {
            "held_out": held_out_result,
            "in_distribution": in_distribution_result,
        }

        print(
            f"{task} held-out pairs:        n={held_out_result['n_pairs']:6d} "
            + " ".join(
                f"{key}={value:.5f}"
                for key, value in held_out_result.items()
                if key != "n_pairs"
            )
        )
        print(
            f"{task} in-distribution pairs: n={in_distribution_result['n_pairs']:6d} "
            + " ".join(
                f"{key}={value:.5f}"
                for key, value in in_distribution_result.items()
                if key != "n_pairs"
            )
        )

    # Point-level holdout generalization check: recover each holdout
    # point's embedding from a few probe distances, then see how well it
    # predicts every active task's outcome for the rest of the points.
    # Run last, right before saving, since it permanently overwrites the
    # holdout rows in model.emb.weight with their recovered values (see
    # src/holdout_probe.py's model-mutation contract).
    holdout_point_eval = None

    if n_holdout_points > 0:
        holdout_point_eval = recover_and_evaluate_holdout_points(
            model=model,
            world=world,
            holdout_point_pairs=holdout_point_pairs,
            probe_steps=probe_steps,
            probe_lr=probe_lr,
            probe_weight_decay=probe_weight_decay,
            batch_size=training_cfg["batch_size"],
            device=device,
            eval_tasks=tasks,
        )

        print(
            f"holdout points: n={holdout_point_eval['n_holdout_points']} "
            f"n_probes={holdout_point_eval['n_probes']}"
        )

        for task in tasks:
            aggregate = holdout_point_eval["aggregate"][task]
            print(
                f"  {task}: aggregate="
                + " ".join(
                    f"{key}={value:.5f}"
                    for key, value in aggregate.items()
                    if key != "n_pairs"
                )
            )

    # Create the model directory when it does not already exist.
    os.makedirs("models", exist_ok=True)

    manifold_name = manifold_cfg["name"]
    task_name = "_".join(tasks)
    save_path = f"models/{manifold_name}_{task_name}_model.pt"

    # Save enough information to reconstruct the trained model later,
    # without re-saving the (potentially large) precomputed distance
    # matrix - analysis re-derives it cheaply from the on-disk cache via
    # make_manifold_world with the same (manifold_name, n, seed).
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "world_meta": {
                "type": "manifold",
                "manifold_name": manifold_name,
                "n": manifold_cfg["n_points"],
                "seed": manifold_cfg["seed"],
            },
            # Embedding-quality metrics collected every progress_interval
            # steps; empty when progress_interval is 0/absent.
            "progress": progress_rows,
            # Per-task held-out vs. in-distribution raw task loss/metrics,
            # computed once after training - see TASK_SPECS[task]["evaluate"].
            "held_out_eval": {
                "test_fraction": test_fraction,
                "per_task": held_out_eval,
            },
            # Which point indices were held out entirely (None when the
            # feature is off) - analysis/analysis_manifold.py needs this
            # to know which embedding rows are recovered-from-probes
            # rather than directly trained.
            "holdout_points": (
                holdout_points.tolist() if holdout_points is not None
                else None
            ),
            # Point-level holdout generalization results (None when the
            # feature is off) - see recover_and_evaluate_holdout_points.
            "holdout_point_eval": holdout_point_eval,
        },
        save_path,
    )

    print(f"Saved model to {save_path}")

    # Keep a plain-text record of exactly what config produced this
    # checkpoint, alongside it.
    config_copy_path = f"models/{manifold_name}_{task_name}_config.yaml"
    shutil.copyfile(config_path, config_copy_path)
    print(f"Saved config copy to {config_copy_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a multi-task transformer on a manifolds.Manifold surface."
    )

    parser.add_argument(
        "config_path",
        nargs="?",
        default="configs/octahedron_default.yaml",
        help="Path to a YAML config (see configs/octahedron_default.yaml).",
    )

    args = parser.parse_args()

    main(args.config_path)
