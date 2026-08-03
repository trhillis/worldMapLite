"""
Does the octahedron (or any manifolds.Manifold surface) emerge in the
transformer's hidden representations?

Loads a checkpoint produced by src/train_manifold.py, rebuilds the same
fixed-entity manifold world (hits the on-disk geodesic-distance cache, so
this is fast, not a recompute), and probes whether the model's learned
per-entity embeddings recover the manifold's true ambient surface geometry -
via PCA/linear-probe comparison against `manifold.embed(chart_points)`,
distance correlation, nearest-neighbor recall, and intrinsic dimension - the
same toolkit analysis/representations.py already provides for the grid
pipeline, since none of it is world-specific.

Every figure is saved as a PNG (no interactive windows - the Agg backend is
forced below, so nothing pops up and nothing needs closing), alongside the
exact config and printed metrics, under
experiments/manifold_learning/<manifold_name>/<experiment_number>/ - the
archived record of that experiment, since models/*_distance_model.pt gets
overwritten by the next training run.

Usage:
    conda activate worldMap
    python -m analysis.analysis_manifold EXPERIMENT_NUMBER [--checkpoint PATH]

Example:
    conda activate worldMap
    python -m analysis.analysis_manifold 001
    python -m analysis.analysis_manifold 002 --checkpoint models/octahedron_distance_model.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Headless backend: figures are only ever saved to disk here, never shown -
# this must be set before pyplot is imported anywhere, including inside
# analysis.representations below, so it comes before that import too.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

from manifolds import get_manifold
from src.worlds import make_manifold_world
from src.multitask_model import MultiTaskWorldModel
from src.datasets import points_on_faces

from analysis.representations import (
    linear_cka,
    sample_unique_pairs,
    pairwise_distances,
    upper_triangle_values,
    linear_coordinate_probe,
    cross_validated_coordinate_probe,
    nearest_neighbor_recall,
    estimate_intrinsic_dimension,
    compute_embedding_metrics,
    compute_transformer_probe_embeddings,
    get_transformer_activations,
    plot_cka_matrix,
    plot_intrinsic_dimensions,
    plot_training_progress,
)

# Fallback task list for checkpoints saved before training.tasks existed.
DEFAULT_TASKS = ["distance"]

# Number of unique entity pairs used for transformer
# representation analysis.
PAIR_SAMPLE_SIZE = 5000

# Number of fixed reference points used for the transformer-probe entity
# representation (see the "Transformer-probe entity representations"
# section of main()) - one per polyhedron face, so references are spread
# across the manifold rather than clustered near a single arbitrary point.
N_PROBE_REFERENCES = 8

SEED = 0


def _format_metrics(metrics: dict) -> str:
    """
    Format an evaluate_*_examples() result (e.g. {"n_pairs": .., "loss":
    .., "spearman": ..} or {"n_pairs": .., "loss": .., "accuracy": ..,
    "auc": ..}) as "key=value ..." for logging, skipping n_pairs (logged
    separately by every caller) since different tasks report different
    metric names.
    """

    return " ".join(
        f"{key}={value:.5f}"
        for key, value in metrics.items()
        if key != "n_pairs"
    )


def scatter_3d_or_2d(
    ax_or_fig, points, colors, title, save_path=None, highlight_mask=None,
):
    """
    Scatter `points` (n, 2) or (n, 3) on a new figure, 3D if possible.

    `highlight_mask` (optional, boolean, shape (n,)): when given, points
    where it's True are plotted as distinct red-star markers on top of
    the ordinary viridis dots, with a legend entry - used to mark
    recovered holdout points (see src/holdout_probe.py) so recovery
    quality can be inspected visually against the bulk point cloud.

    If `save_path` is given, the figure is also written there (parent
    directories are created as needed) before it is shown.
    """

    is_3d = points.shape[1] >= 3

    if highlight_mask is None:
        highlight_mask = np.zeros(len(points), dtype=bool)

    base_points = points[~highlight_mask]
    base_colors = colors[~highlight_mask]
    highlight_points = points[highlight_mask]

    if is_3d:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            base_points[:, 0], base_points[:, 1], base_points[:, 2],
            c=base_colors, cmap="viridis", s=15,
        )
        if highlight_mask.any():
            ax.scatter(
                highlight_points[:, 0], highlight_points[:, 1],
                highlight_points[:, 2],
                marker="*", s=140, c="red", edgecolors="black",
                label="holdout (recovered)",
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            base_points[:, 0], base_points[:, 1],
            c=base_colors, cmap="viridis", s=15,
        )
        if highlight_mask.any():
            ax.scatter(
                highlight_points[:, 0], highlight_points[:, 1],
                marker="*", s=140, c="red", edgecolors="black",
                label="holdout (recovered)",
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    if highlight_mask.any():
        ax.legend()

    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="chart_points[:, 0]")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def main(experiment_number: str, checkpoint_path: str):
    # Every printed line is also archived to metrics.txt in the experiment
    # folder, so the console output doesn't have to be copy-pasted by hand.
    report_lines = []

    def log(message: str = ""):
        print(message)
        report_lines.append(message)

    log(f"Checkpoint: {checkpoint_path}")

    # --------------------------------------------------
    # Load checkpoint and reconstruct model + world
    # --------------------------------------------------

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    config = checkpoint["config"]
    manifold_cfg = config["manifold"]
    model_cfg = config["model"]

    # Which tasks this checkpoint was trained on - drives every per-task
    # section below (pair-symmetry check, held-out/holdout-point
    # generalization, CKA/intrinsic-dim plots).
    tasks = config["training"].get("tasks", DEFAULT_TASKS)

    manifold = get_manifold(manifold_cfg["name"])

    # Same (manifold_name, n, seed) as training, so this hits the on-disk
    # geodesic-distance cache instead of recomputing it.
    world = make_manifold_world(
        manifold,
        n=manifold_cfg["n_points"],
        seed=manifold_cfg["seed"],
    )

    model = MultiTaskWorldModel(
        num_points=len(world.names),
        **model_cfg,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    figures_dir = (
        Path("experiments/manifold_learning") / manifold.name / experiment_number
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Archive the exact config that produced this checkpoint next to the
    # figures it produces, so the experiment folder is self-contained.
    with open(figures_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    # Archive the periodic embedding-quality snapshots taken during
    # training, if any (older checkpoints, or progress_interval=0, have
    # none - that's fine, just nothing to plot).
    progress_rows = checkpoint.get("progress", [])

    if progress_rows:
        pd.DataFrame(progress_rows).to_csv(
            figures_dir / "progress.csv", index=False
        )

        plot_training_progress(
            progress_rows,
            title=f"{manifold.name}: representation-quality metrics over training",
            save_path=figures_dir / "training_curves.png",
        )
    else:
        log(
            "\nNo training-progress snapshots in this checkpoint "
            "(trained before this feature, or progress_interval=0) - "
            "skipping progress.csv / training_curves.png."
        )

    log(f"Experiment: {experiment_number}")
    log(f"Manifold: {manifold.name} (ambient_dim={manifold.ambient_dim})")
    log(f"Entities: {len(world.names)}")
    log(f"Transformer layers: {len(model.transformer.layers)}")

    # --------------------------------------------------
    # Entity embedding analysis
    # --------------------------------------------------

    num_points = len(world.names)

    # True ambient surface embedding - the thing we're asking whether the
    # model's hidden representation "matches".
    true_coordinates = world.coordinates.astype(np.float64)

    # Color points by the first chart coordinate: the face index for the
    # octahedron/icosahedron (a small integer, one per face), or a
    # continuous chart coordinate for the torus/Moebius strip.
    chart_points = world.meta["chart_points"]
    point_colors = chart_points[:, 0]

    embeddings = model.emb.weight.detach().cpu().numpy()

    # Points held out entirely from training (see src/train_manifold.py's
    # holdout-point feature) already carry their *recovered* values by
    # this point - recover_and_evaluate_holdout_points committed them
    # into model_state_dict before the checkpoint was saved - so they are
    # not noise the way they are during training's periodic snapshots.
    # Rather than exclude them, every "bulk quality" metric below is
    # fit/measured on the non-holdout points only (a clean read on the
    # embedding table's general quality, unaffected by how well recovery
    # happened to work), and the recovered points are then transformed
    # into that same basis and plotted, visually marked, as a direct
    # check on recovery itself.
    holdout_points = checkpoint.get("holdout_points")

    non_holdout_mask = np.ones(num_points, dtype=bool)
    if holdout_points:
        non_holdout_mask[holdout_points] = False
        log(
            f"\nExcluding {len(holdout_points)} holdout points from bulk "
            "embedding-quality metrics (plotted separately, marked)"
        )

    # Used both by the transformer-probe PCA plots below and by the
    # learned-embedding PCA plots further down, so recovered holdout
    # points can be marked distinctly wherever they're plotted.
    holdout_mask = ~non_holdout_mask

    bulk_embeddings = embeddings[non_holdout_mask]
    bulk_true_coordinates = true_coordinates[non_holdout_mask]

    n_components = min(3, manifold.ambient_dim)

    embedding_pca = PCA(n_components=n_components).fit(bulk_embeddings)
    embedding_pca_points = embedding_pca.transform(embeddings)

    true_pca = PCA(n_components=n_components).fit(bulk_true_coordinates)
    true_pca_points = true_pca.transform(true_coordinates)

    _, coordinate_r2 = linear_coordinate_probe(
        bulk_embeddings, bulk_true_coordinates,
    )

    # Same linear probe, refit here (rather than via linear_coordinate_probe,
    # which only predicts on the data it was fit on) so recovered holdout
    # points can be transformed into the same reconstruction basis below.
    linear_probe = LinearRegression().fit(
        bulk_embeddings, bulk_true_coordinates,
    )
    linear_prediction = linear_probe.predict(embeddings)

    cv_r2_mean, cv_r2_std = cross_validated_coordinate_probe(
        bulk_embeddings, bulk_true_coordinates, n_splits=5, seed=SEED,
    )

    latent_distance_matrix = pairwise_distances(bulk_embeddings)
    true_distance_matrix = pairwise_distances(bulk_true_coordinates)

    distance_correlation = spearmanr(
        upper_triangle_values(latent_distance_matrix),
        upper_triangle_values(true_distance_matrix),
    ).statistic

    neighbor_recall = nearest_neighbor_recall(
        bulk_embeddings, bulk_true_coordinates,
    )

    embedding_intrinsic_dimension = estimate_intrinsic_dimension(
        bulk_embeddings
    )

    log("\nEntity embedding metrics (non-holdout points)")
    log(
        "PCA explained variance "
        f"({n_components} components): "
        f"{embedding_pca.explained_variance_ratio_.sum():.4f}"
    )
    log(f"In-sample linear coordinate probe R²: {coordinate_r2:.4f}")
    log(
        "Cross-validated coordinate R²: "
        f"{cv_r2_mean:.4f} ± {cv_r2_std:.4f}"
    )
    log(f"Geodesic-distance Spearman correlation: {distance_correlation:.4f}")
    log(f"Nearest-neighbor recall: {neighbor_recall:.4f}")
    log(f"Embedding intrinsic dimension: {embedding_intrinsic_dimension:.4f}")

    # --------------------------------------------------
    # Transformer-probe entity representations
    # --------------------------------------------------
    #
    # EXPERIMENTS.md entry 102: the metrics above (raw model.emb.weight)
    # are blind to whatever the transformer itself contributes and can
    # understate a representation's true task-relevant content. Re-run the
    # same probes on the transformer's output instead: every entity is
    # paired with a fixed reference entity through the trained 3-token
    # sequence, and the transformed task token is read out as that
    # entity's representation - see compute_transformer_probe_embeddings.
    #
    # 103's "next steps" flagged that a single arbitrary reference point
    # makes these numbers sensitive to which point happens to be chosen.
    # To address that, N_PROBE_REFERENCES reference points are used
    # instead of one - one per polyhedron face where possible (via
    # points_on_faces), so references are spread across the manifold's
    # distinct faces rather than clustered near one arbitrary point - and
    # the resulting per-entity representations are averaged. Falls back to
    # a single reference point (the first non-holdout point) for manifolds
    # without polyhedron faces (e.g. FlatTorus, FlatMobiusStrip).
    if hasattr(manifold, "n_faces"):
        reference_faces = np.unique(
            np.linspace(
                0, manifold.n_faces - 1,
                num=min(N_PROBE_REFERENCES, manifold.n_faces),
            ).round().astype(np.int64)
        )

        probe_point_indices = []

        for face in reference_faces:
            face_points = points_on_faces(world, [int(face)])
            candidates = face_points[non_holdout_mask[face_points]]

            if len(candidates) == 0:
                raise ValueError(
                    f"No non-holdout points on face {face} to use as a "
                    "transformer-probe reference point."
                )

            probe_point_indices.append(int(candidates[0]))
    else:
        probe_point_indices = [int(np.flatnonzero(non_holdout_mask)[0])]

    log(
        f"\nTransformer-probe reference points ({len(probe_point_indices)} "
        f"total): {probe_point_indices}"
    )

    def log_probe_metrics(header, metrics):
        log(f"\n{header}")
        log(
            f"PCA explained variance ({n_components} components): "
            f"{metrics['pca_explained_variance']:.4f}"
        )
        log(f"In-sample linear coordinate probe R²: {metrics['linear_r2']:.4f}")
        log(
            "Cross-validated coordinate R²: "
            f"{metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}"
        )
        log(
            "Geodesic-distance Spearman correlation: "
            f"{metrics['distance_spearman']:.4f}"
        )
        log(f"Nearest-neighbor recall: {metrics['nn_recall']:.4f}")
        log(f"Embedding intrinsic dimension: {metrics['intrinsic_dim']:.4f}")

    def plot_probe_pca(embeddings_for_pca, title, save_path):
        # Same PCA-scatter treatment as the raw embedding table below,
        # fit on non-holdout points and applied to every point (so
        # recovered holdout points can be marked distinctly).
        pca = PCA(n_components=n_components).fit(
            embeddings_for_pca[non_holdout_mask]
        )
        pca_points = pca.transform(embeddings_for_pca)

        scatter_3d_or_2d(
            None, pca_points, point_colors, title,
            save_path=save_path, highlight_mask=holdout_mask,
        )

    for task in tasks:
        # One transformer-probe representation per reference point.
        per_reference_embeddings = [
            compute_transformer_probe_embeddings(
                model=model, task=task, probe_point_index=probe_point_index,
            )
            for probe_point_index in probe_point_indices
        ]

        # Headline result: the per-entity representations averaged across
        # every reference point, which is what compute_embedding_metrics
        # is run on above the per-reference breakdown below.
        averaged_embeddings = np.mean(per_reference_embeddings, axis=0)

        averaged_metrics = compute_embedding_metrics(
            averaged_embeddings[non_holdout_mask],
            bulk_true_coordinates,
            true_distance_upper=upper_triangle_values(true_distance_matrix),
            seed=SEED,
        )

        log_probe_metrics(
            f"Transformer-probe entity metrics ({task}, non-holdout points, "
            f"averaged over {len(probe_point_indices)} reference points "
            f"{probe_point_indices})",
            averaged_metrics,
        )

        plot_probe_pca(
            averaged_embeddings,
            f"PCA of transformer-probe entity representations "
            f"({manifold.name}, {task}, averaged over "
            f"{len(probe_point_indices)} references)",
            figures_dir / f"transformer_probe_pca_{task}.png",
        )

        # Per-reference-point breakdown, so the averaged result above can
        # be checked against - and each individual reference point's own
        # metrics/plot inspected - rather than only ever seeing the
        # average.
        for probe_point_index, transformer_probe_embeddings in zip(
            probe_point_indices, per_reference_embeddings,
        ):
            transformer_probe_metrics = compute_embedding_metrics(
                transformer_probe_embeddings[non_holdout_mask],
                bulk_true_coordinates,
                true_distance_upper=upper_triangle_values(true_distance_matrix),
                seed=SEED,
            )

            log_probe_metrics(
                f"Transformer-probe entity metrics ({task}, non-holdout "
                f"points, fixed reference point index={probe_point_index})",
                transformer_probe_metrics,
            )

            plot_probe_pca(
                transformer_probe_embeddings,
                f"PCA of transformer-probe entity representations "
                f"({manifold.name}, {task}, reference={probe_point_index})",
                figures_dir / f"transformer_probe_pca_{task}_ref{probe_point_index}.png",
            )

    # Side-by-side comparison: learned-embedding PCA vs. true-manifold PCA.
    # This is the direct visual answer to "does the hidden representation
    # match the manifold's surface." Recovered holdout points (if any) are
    # transformed into the same bases and marked distinctly, so recovery
    # can be checked visually against the bulk point cloud.
    scatter_3d_or_2d(
        None, embedding_pca_points, point_colors,
        f"PCA of learned entity embeddings ({manifold.name})",
        save_path=figures_dir / "embedding_pca.png",
        highlight_mask=holdout_mask,
    )

    scatter_3d_or_2d(
        None, true_pca_points, point_colors,
        f"PCA of true ambient {manifold.name} surface",
        save_path=figures_dir / "true_manifold_pca.png",
        highlight_mask=holdout_mask,
    )

    # Linear-probe reconstruction, plotted in the same ambient space.
    reconstruction_pca_points = PCA(
        n_components=n_components
    ).fit(linear_prediction[non_holdout_mask]).transform(linear_prediction)

    scatter_3d_or_2d(
        None, reconstruction_pca_points, point_colors,
        f"Linear-probe reconstruction of {manifold.name} coordinates "
        f"(R²={coordinate_r2:.3f})",
        save_path=figures_dir / "linear_probe_reconstruction.png",
        highlight_mask=holdout_mask,
    )

    # --------------------------------------------------
    # Pair and transformer representation analysis
    # --------------------------------------------------

    pairs = sample_unique_pairs(
        num_points=num_points, n_pairs=PAIR_SAMPLE_SIZE, seed=SEED,
    )

    point_i = torch.tensor(pairs[:, 0], dtype=torch.long)
    point_j = torch.tensor(pairs[:, 1], dtype=torch.long)

    transformer_task_layers = [
        f"transformer_{layer_index}_task"
        for layer_index in range(1, len(model.transformer.layers) + 1)
    ]

    transformer_sequence_layers = [
        f"transformer_{layer_index}_sequence"
        for layer_index in range(1, len(model.transformer.layers) + 1)
    ]

    task_token_layer_names = [*transformer_task_layers, "pair", "h1", "h2"]

    # Every plot/log in this section depends on which task token drives
    # the transformer forward pass, so it's repeated once per active task
    # (entity-token representations are task-dependent too, since the
    # shared self-attention mixes the task token into them) - filenames
    # are suffixed with the task name so multi-task runs don't clobber
    # each other's figures.
    for task in tasks:
        activations = get_transformer_activations(
            model=model, point_i=point_i, point_j=point_j, task=task,
        )

        plot_cka_matrix(
            activations=activations,
            layer_names=task_token_layer_names,
            title=f"{manifold.name} ({task}): task-token and head CKA",
            save_path=figures_dir / f"cka_task_tokens_{task}.png",
        )

        plot_intrinsic_dimensions(
            activations=activations,
            layer_names=task_token_layer_names,
            title=f"{manifold.name} ({task}): task-token and head intrinsic "
                  "dimensions (across layers - look for the 'hunchback' "
                  "profile)",
            save_path=figures_dir / f"intrinsic_dim_task_tokens_{task}.png",
        )

        if transformer_sequence_layers:
            plot_cka_matrix(
                activations=activations,
                layer_names=transformer_sequence_layers,
                title=f"{manifold.name} ({task}): full-sequence transformer CKA",
                save_path=figures_dir / f"cka_sequence_layers_{task}.png",
            )

            plot_intrinsic_dimensions(
                activations=activations,
                layer_names=transformer_sequence_layers,
                title=f"{manifold.name} ({task}): full-sequence intrinsic "
                      "dimensions",
                save_path=figures_dir / f"intrinsic_dim_sequence_layers_{task}.png",
            )

        log(f"\n{manifold.name} ({task}): entity-token CKA")
        for layer_index in range(1, len(model.transformer.layers) + 1):
            similarity = linear_cka(
                activations[f"transformer_{layer_index}_i"],
                activations[f"transformer_{layer_index}_j"],
            )
            log(f"layer {layer_index}: {similarity:.4f}")

    # --------------------------------------------------
    # Pair-symmetry sanity check
    # --------------------------------------------------

    log(f"\n{'=' * 60}")
    log("PAIR-SYMMETRY CHECK")
    log(f"{'=' * 60}")

    with torch.inference_mode():
        for task in tasks:
            forward_prediction = model(task, point_i, point_j)
            reverse_prediction = model(task, point_j, point_i)

            absolute_difference = torch.abs(
                forward_prediction - reverse_prediction
            )

            log(
                f"{task}: mean difference={absolute_difference.mean().item():.10f}, "
                f"maximum difference={absolute_difference.max().item():.10f}"
            )

    # --------------------------------------------------
    # Held-out pair generalization
    # --------------------------------------------------

    log(f"\n{'=' * 60}")
    log("HELD-OUT PAIR GENERALIZATION")
    log(f"{'=' * 60}")

    held_out_eval = checkpoint.get("held_out_eval")

    if held_out_eval is None:
        log(
            "No held-out evaluation in this checkpoint (trained before "
            "this feature) - skipping."
        )
    else:
        log(f"test_fraction: {held_out_eval['test_fraction']}")

        for task in tasks:
            task_eval = held_out_eval["per_task"][task]
            held_out = task_eval["held_out"]
            in_distribution = task_eval["in_distribution"]

            log(f"\n[{task}]")
            log(
                f"held-out:        n={held_out['n_pairs']:6d} "
                f"{_format_metrics(held_out)}"
            )
            log(
                f"in-distribution: n={in_distribution['n_pairs']:6d} "
                f"{_format_metrics(in_distribution)}"
            )
            log(
                "generalization gap (held-out loss - in-distribution loss): "
                f"{held_out['loss'] - in_distribution['loss']:+.5f}"
            )

    # --------------------------------------------------
    # Holdout point generalization
    # --------------------------------------------------

    log(f"\n{'=' * 60}")
    log("HOLDOUT POINT GENERALIZATION")
    log(f"{'=' * 60}")

    holdout_point_eval = checkpoint.get("holdout_point_eval")

    if holdout_point_eval is None:
        log(
            "No holdout-point evaluation in this checkpoint (trained "
            "before this feature, or n_holdout_points=0) - skipping."
        )
    else:
        log(f"n_holdout_points: {holdout_point_eval['n_holdout_points']}")
        log(f"n_probes: {holdout_point_eval['n_probes']}")
        log(f"probe_steps: {holdout_point_eval['probe_steps']}")
        log(
            "Recovery is always distance-only (see src/holdout_probe.py); "
            "results below show how well the distance-only-recovered "
            "embedding predicts each active task."
        )

        for task in holdout_point_eval["eval_tasks"]:
            aggregate = holdout_point_eval["aggregate"][task]
            holdout_to_holdout = holdout_point_eval["holdout_to_holdout"][task]

            log(f"\n[{task}]")
            log(
                "holdout-to-trained (aggregate): "
                f"n={aggregate['n_pairs']:6d} {_format_metrics(aggregate)}"
            )

            if holdout_to_holdout is None:
                log(
                    "holdout-to-holdout: skipped (fewer than 2 holdout points)"
                )
            else:
                log(
                    "holdout-to-holdout:            "
                    f"n={holdout_to_holdout['n_pairs']:6d} "
                    f"{_format_metrics(holdout_to_holdout)}"
                )

    with open(figures_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    log(f"\nSaved all experiment outputs to {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a manifold-distance transformer checkpoint."
    )

    parser.add_argument(
        "experiment_number",
        type=int,
        help="Three-digit experiment number (e.g. 1 or 001). Figures, the "
             "exact config, and printed metrics are archived under "
             "experiments/manifold_learning/<manifold_name>/<NNN>/.",
    )

    parser.add_argument(
        "--checkpoint",
        default="models/octahedron_distance_model.pt",
        help="Path to a checkpoint saved by src/train_manifold.py. Defaults "
             "to a plain distance-only run's fixed path; multi-task runs "
             "save to models/{manifold}_{'_'.join(tasks)}_model.pt instead "
             "(e.g. models/octahedron_distance_same_triangle_model.pt), so "
             "pass --checkpoint explicitly for those.",
    )

    args = parser.parse_args()

    if not 0 <= args.experiment_number <= 999:
        parser.error("experiment_number must be between 0 and 999")

    main(f"{args.experiment_number:03d}", args.checkpoint)
