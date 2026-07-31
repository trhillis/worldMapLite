import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from scipy.stats import spearmanr

from legacy.worlds import make_grid
from src.multitask_model import MultiTaskWorldModel

from analysis.representations import (
    linear_cka,
    sample_unique_pairs,
    pairwise_distances,
    upper_triangle_values,
    linear_coordinate_probe,
    cross_validated_coordinate_probe,
    nearest_neighbor_recall,
    estimate_intrinsic_dimension,
    get_transformer_activations,
    plot_cka_matrix,
    plot_intrinsic_dimensions,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT_PATH = "models/distance_model.pt"

# Number of unique entity pairs used for transformer
# representation analysis.
PAIR_SAMPLE_SIZE = 5000

SEED = 0

print(f"Checkpoint: {CHECKPOINT_PATH}")


# --------------------------------------------------
# Load checkpoint and reconstruct model
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
)

cfg = checkpoint["config"]

tasks = cfg.get(
    "tasks",
    ("distance", "nearest"),
)

if isinstance(tasks, str):
    tasks = (tasks,)
else:
    tasks = tuple(tasks)

world = make_grid(
    cfg["width"],
    cfg["height"],
)

model = MultiTaskWorldModel(
    num_points=len(world.names),
    emb_dim=cfg["emb_dim"],
    hidden_dim=cfg["hidden_dim"],

    # These values use transformer defaults when older checkpoint
    # configurations do not contain them.
    num_heads=cfg.get(
        "num_heads",
        4,
    ),

    num_layers=cfg.get(
        "num_layers",
        2,
    ),

    dropout=cfg.get(
        "dropout",
        0.0,
    ),

    normalize_embeddings=cfg.get(
        "normalize_embeddings",
        False,
    ),
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(f"Tasks: {tasks}")
print(
    "Transformer layers: "
    f"{len(model.transformer.layers)}"
)


# --------------------------------------------------
# Entity embedding analysis
# --------------------------------------------------

num_points = len(
    world.names
)

true_coordinates = (
    world.coordinates
    .astype(np.float64)
)

embeddings = (
    model.emb.weight
    .detach()
    .cpu()
    .numpy()
)

pca = PCA(
    n_components=2
)

embedding_pca = pca.fit_transform(
    embeddings
)

linear_prediction, coordinate_r2 = (
    linear_coordinate_probe(
        embeddings,
        true_coordinates,
    )
)

cv_r2_mean, cv_r2_std = (
    cross_validated_coordinate_probe(
        embeddings,
        true_coordinates,
        n_splits=5,
        seed=SEED,
    )
)

latent_distance_matrix = (
    pairwise_distances(
        embeddings
    )
)

# The grid distance task uses Euclidean distance.
true_distance_matrix = (
    pairwise_distances(
        true_coordinates
    )
)

distance_correlation = spearmanr(
    upper_triangle_values(
        latent_distance_matrix
    ),
    upper_triangle_values(
        true_distance_matrix
    ),
).statistic

neighbor_recall = (
    nearest_neighbor_recall(
        embeddings,
        true_coordinates,
    )
)

embedding_intrinsic_dimension = (
    estimate_intrinsic_dimension(
        embeddings
    )
)

print("\nEntity embedding metrics")

print(
    "PCA explained variance: "
    f"{pca.explained_variance_ratio_.sum():.4f}"
)

print(
    "In-sample linear coordinate probe R²: "
    f"{coordinate_r2:.4f}"
)

print(
    "Cross-validated coordinate R²: "
    f"{cv_r2_mean:.4f} "
    f"± {cv_r2_std:.4f}"
)

print(
    "Euclidean-distance Spearman correlation: "
    f"{distance_correlation:.4f}"
)

print(
    "Nearest-neighbor recall: "
    f"{neighbor_recall:.4f}"
)

print(
    "Embedding intrinsic dimension: "
    f"{embedding_intrinsic_dimension:.4f}"
)


# PCA plot
plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    embedding_pca[:, 0],
    embedding_pca[:, 1],
    s=15,
)

plt.title(
    "PCA of shared entity embeddings"
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()


# Linear-probe reconstruction
plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    linear_prediction[:, 0],
    linear_prediction[:, 1],
    s=15,
)

plt.title(
    "Coordinates reconstructed by linear probe "
    f"(R²={coordinate_r2:.3f})"
)

plt.xlabel("Predicted x")
plt.ylabel("Predicted y")
plt.axis("equal")
plt.tight_layout()
plt.show()


# True coordinates
plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    true_coordinates[:, 0],
    true_coordinates[:, 1],
    s=15,
)

plt.title(
    "True grid coordinates"
)

plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Pair and transformer representation analysis
# --------------------------------------------------

pairs = sample_unique_pairs(
    num_points=num_points,
    n_pairs=PAIR_SAMPLE_SIZE,
    seed=SEED,
)

point_i = torch.tensor(
    pairs[:, 0],
    dtype=torch.long,
)

point_j = torch.tensor(
    pairs[:, 1],
    dtype=torch.long,
)

task_activations = {}

for task in tasks:
    task_activations[task] = (
        get_transformer_activations(
            model=model,
            point_i=point_i,
            point_j=point_j,
            task=task,
        )
    )


# Create representation names dynamically so the analysis also
# works when num_layers changes.
transformer_task_layers = [
    f"transformer_{layer_index}_task"
    for layer_index in range(
        1,
        len(model.transformer.layers) + 1,
    )
]

transformer_sequence_layers = [
    f"transformer_{layer_index}_sequence"
    for layer_index in range(
        1,
        len(model.transformer.layers) + 1,
    )
]


for task_name, activations in (
    task_activations.items()
):
    print(
        f"\n{'=' * 60}"
    )

    print(
        f"{task_name.upper()} TASK ANALYSIS"
    )

    print(
        f"{'=' * 60}"
    )

    # The input task token is identical for every sampled pair.
    # It is therefore excluded from CKA and intrinsic-dimension
    # analysis because centering makes it a zero representation.
    task_token_layer_names = [
        *transformer_task_layers,
        "pair",
        "h1",
        "h2",
    ]

    plot_cka_matrix(
        activations=activations,
        layer_names=task_token_layer_names,
        title=(
            f"{task_name.capitalize()} "
            "task-token and head CKA"
        ),
    )

    plot_intrinsic_dimensions(
        activations=activations,
        layer_names=task_token_layer_names,
        title=(
            f"{task_name.capitalize()} "
            "task-token and head intrinsic dimensions"
        ),
    )

    # Analyze the complete three-token sequence at each
    # transformer layer.
    if transformer_sequence_layers:
        plot_cka_matrix(
            activations=activations,
            layer_names=transformer_sequence_layers,
            title=(
                f"{task_name.capitalize()} "
                "full-sequence transformer CKA"
            ),
        )

        plot_intrinsic_dimensions(
            activations=activations,
            layer_names=transformer_sequence_layers,
            title=(
                f"{task_name.capitalize()} "
                "full-sequence intrinsic dimensions"
            ),
        )

    # Compare entity-i and entity-j token representations at
    # each transformer layer.
    print(
        f"\n{task_name.capitalize()} "
        "entity-token CKA"
    )

    for layer_index in range(
        1,
        len(model.transformer.layers) + 1,
    ):
        token_i_name = (
            f"transformer_{layer_index}_i"
        )

        token_j_name = (
            f"transformer_{layer_index}_j"
        )

        similarity = linear_cka(
            activations[token_i_name],
            activations[token_j_name],
        )

        print(
            f"layer {layer_index}: "
            f"{similarity:.4f}"
        )


# --------------------------------------------------
# Cross-task representation similarity
# --------------------------------------------------

if (
    "distance" in task_activations
    and "nearest" in task_activations
):
    print(
        f"\n{'=' * 60}"
    )

    print(
        "CROSS-TASK REPRESENTATION SIMILARITY"
    )

    print(
        f"{'=' * 60}"
    )

    cross_task_layer_names = [
        *transformer_task_layers,
        "pair",
        "h1",
        "h2",
    ]

    cross_task_similarities = []

    for layer_name in (
        cross_task_layer_names
    ):
        similarity = linear_cka(
            task_activations[
                "distance"
            ][layer_name],
            task_activations[
                "nearest"
            ][layer_name],
        )

        cross_task_similarities.append(
            similarity
        )

        print(
            f"{layer_name}: "
            f"{similarity:.4f}"
        )

    plt.figure(
        figsize=(
            max(
                7,
                len(cross_task_layer_names)
                * 0.9,
            ),
            5,
        )
    )

    plt.plot(
        cross_task_layer_names,
        cross_task_similarities,
        marker="o",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.title(
        "Distance versus nearest cross-task CKA"
    )

    plt.xlabel("Representation")
    plt.ylabel("Linear CKA")

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Optional symmetry verification
# --------------------------------------------------

print(
    f"\n{'=' * 60}"
)

print(
    "PAIR-SYMMETRY CHECK"
)

print(
    f"{'=' * 60}"
)

with torch.inference_mode():
    for task in tasks:
        forward_prediction = model(
            task,
            point_i,
            point_j,
        )

        reverse_prediction = model(
            task,
            point_j,
            point_i,
        )

        absolute_difference = torch.abs(
            forward_prediction
            - reverse_prediction
        )

        print(
            f"{task}: "
            f"mean difference="
            f"{absolute_difference.mean().item():.10f}, "
            f"maximum difference="
            f"{absolute_difference.max().item():.10f}"
        )
