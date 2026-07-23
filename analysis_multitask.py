import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from scipy.stats import spearmanr
from skdim.id import TwoNN

from worlds import make_grid, make_manifold_world

from manifolds.flat_torus import FlatTorus
from manifolds.mobius import FlatMobiusStrip

from manifolds.polyhedra import octahedron

from multitask_model import MultiTaskWorldModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT_PATH = "models/distance_model.pt"

# Number of unique entity pairs used for transformer
# representation analysis.
PAIR_SAMPLE_SIZE = 5000

SEED = 0

print(f"Checkpoint: {CHECKPOINT_PATH}")


def build_world_from_config(cfg):
    """
    Reconstruct the same world used during training.
    """

    world_type = cfg.get(
        "world_type",
        "grid",
    )

    if world_type == "grid":
        return make_grid(
            cfg["width"],
            cfg["height"],
        )

    if world_type != "manifold":
        raise ValueError(
            f"Unsupported world type: {world_type}"
        )

    manifold_name = cfg.get("manifold")

    if manifold_name in {
        "mobius",
        "flat_mobius",
    }:
        manifold = FlatMobiusStrip(
            length=cfg.get(
                "mobius_length",
                2.0 * np.pi,
            ),
            width=cfg.get(
                "mobius_width",
                1.0,
            ),
        )

        diameter = np.pi

    elif manifold_name in {
        "torus",
        "flat_torus",
    }:
        manifold = FlatTorus()

        diameter = (
            np.pi
            * np.sqrt(2.0)
        )

    elif manifold_name in {
        "octahedron",
        "regular_octahedron",
    }:
        manifold = Octahedron()

        diameter = np.sqrt(3.0)

    else:
        raise ValueError(
            f"Unknown manifold: {manifold_name}"
        )

    return make_manifold_world(
        manifold=manifold,
        n=cfg["manifold_points"],
        seed=cfg["seed"],
        diameter=diameter,
    )


def true_world_distance_matrix(world):
    world_type = world.meta["type"]

    if world_type == "grid":
        return pairwise_distances(
            world.coordinates
        )

    if world_type == "manifold":
        if world.manifold is None:
            raise ValueError(
                "Manifold world has no manifold object"
            )

        return np.asarray(
            world.manifold.distance_matrix(
                world.coordinates
            ),
            dtype=np.float64,
        )

    raise ValueError(
        "True distance matrix is not implemented "
        f"for world type {world_type}"
    )


def get_probe_targets(world):
    """
    Choose coordinates used only for linear probing and plotting.

    Grid:
        use intrinsic 2D grid coordinates

    Manifold:
        use ambient visualization coordinates

    These probe targets are not used for geodesic-distance evaluation.
    """

    world_type = world.meta["type"]

    if world_type == "grid":
        return (
            np.asarray(
                world.coordinates,
                dtype=np.float64,
            ),
            "grid coordinates",
        )

    if world_type == "manifold":
        if world.ambient_coordinates is None:
            raise ValueError(
                "Manifold world has no ambient coordinates"
            )

        return (
            np.asarray(
                world.ambient_coordinates,
                dtype=np.float64,
            ),
            "ambient coordinates",
        )

    raise ValueError(
        "Probe targets are not implemented for "
        f"world type {world_type}"
    )


def project_to_2d(values):
    """
    Convert coordinate targets into two dimensions for plotting.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "Plot values must have shape "
            "[num_points, dimensions]"
        )

    if values.shape[1] == 2:
        return values

    return PCA(
        n_components=2
    ).fit_transform(values)

# --------------------------------------------------
# General analysis utilities
# --------------------------------------------------

def linear_cka(x, y):
    """
    Compute linear centered-kernel alignment between two
    representation matrices.

    Expected shapes:
        x: [num_examples, x_dimension]
        y: [num_examples, y_dimension]
    """

    if isinstance(x, torch.Tensor):
        x = (
            x.detach()
            .cpu()
            .numpy()
        )

    if isinstance(y, torch.Tensor):
        y = (
            y.detach()
            .cpu()
            .numpy()
        )

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(
            f"x must have two dimensions, got shape {x.shape}"
        )

    if y.ndim != 2:
        raise ValueError(
            f"y must have two dimensions, got shape {y.shape}"
        )

    if len(x) != len(y):
        raise ValueError(
            "x and y must contain the same number of examples"
        )

    # Center every representation dimension.
    x = x - x.mean(
        axis=0,
        keepdims=True,
    )

    y = y - y.mean(
        axis=0,
        keepdims=True,
    )

    numerator = (
        np.linalg.norm(
            x.T @ y,
            ord="fro",
        )
        ** 2
    )

    denominator = (
        np.linalg.norm(
            x.T @ x,
            ord="fro",
        )
        * np.linalg.norm(
            y.T @ y,
            ord="fro",
        )
    )

    # A constant representation has zero centered norm.
    if denominator <= 1e-12:
        return float("nan")

    return float(
        numerator / denominator
    )


def sample_unique_pairs(
    num_points,
    n_pairs,
    seed=0,
):
    """
    Sample unique unordered pairs of point indices.

    Every pair satisfies:
        i < j
    """

    rng = np.random.default_rng(seed)

    all_pairs = np.array(
        [
            (i, j)
            for i in range(num_points)
            for j in range(i + 1, num_points)
        ],
        dtype=np.int64,
    )

    n_pairs = min(
        n_pairs,
        len(all_pairs),
    )

    selected_indices = rng.choice(
        len(all_pairs),
        size=n_pairs,
        replace=False,
    )

    return all_pairs[selected_indices]


def pairwise_distances(x):
    """
    Compute the full Euclidean distance matrix.

    Input:
        [num_points, dimensions]

    Output:
        [num_points, num_points]
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    differences = (
        x[:, None, :]
        - x[None, :, :]
    )

    return np.linalg.norm(
        differences,
        axis=-1,
    )


def upper_triangle_values(matrix):
    """
    Return all entries above the diagonal.
    """

    indices = np.triu_indices_from(
        matrix,
        k=1,
    )

    return matrix[indices]


def linear_coordinate_probe(
    embeddings,
    coordinates,
):
    """
    Fit a linear model from learned embeddings to true coordinates.

    This score is measured on the same data used to train the probe.
    Use the cross-validated probe below for a stronger estimate.
    """

    probe = LinearRegression()

    probe.fit(
        embeddings,
        coordinates,
    )

    prediction = probe.predict(
        embeddings
    )

    score = r2_score(
        coordinates,
        prediction,
        multioutput="variance_weighted",
    )

    return prediction, float(score)


def cross_validated_coordinate_probe(
    embeddings,
    coordinates,
    n_splits=5,
    seed=0,
):
    """
    Evaluate coordinate recovery with cross-validated Ridge regression.
    """

    embeddings = np.asarray(embeddings)
    coordinates = np.asarray(coordinates)

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    scores = []

    for train_indices, test_indices in splitter.split(
        embeddings
    ):
        probe = Ridge(
            alpha=1.0
        )

        probe.fit(
            embeddings[train_indices],
            coordinates[train_indices],
        )

        prediction = probe.predict(
            embeddings[test_indices]
        )

        score = r2_score(
            coordinates[test_indices],
            prediction,
            multioutput="variance_weighted",
        )

        scores.append(score)

    return (
        float(np.mean(scores)),
        float(np.std(scores)),
    )


def nearest_neighbor_recall_from_matrix(
    embeddings,
    true_distances,
):
    latent_distances = pairwise_distances(
        embeddings
    )

    true_distances = np.array(
        true_distances,
        dtype=np.float64,
        copy=True,
    )

    np.fill_diagonal(
        latent_distances,
        np.inf,
    )

    np.fill_diagonal(
        true_distances,
        np.inf,
    )

    recovered = []

    for point_index in range(
        len(embeddings)
    ):
        minimum_true_distance = (
            true_distances[
                point_index
            ].min()
        )

        true_neighbors = set(
            np.flatnonzero(
                np.isclose(
                    true_distances[
                        point_index
                    ],
                    minimum_true_distance,
                    atol=1e-8,
                    rtol=0.0,
                )
            )
        )

        predicted_neighbor = int(
            np.argmin(
                latent_distances[
                    point_index
                ]
            )
        )

        recovered.append(
            predicted_neighbor
            in true_neighbors
        )

    return float(
        np.mean(recovered)
    )


def estimate_intrinsic_dimension(x):
    """
    Estimate intrinsic dimension using the TwoNN estimator.

    Returns NaN when the representation is too small, constant,
    duplicate-heavy, or otherwise unsuitable for TwoNN.
    """

    if isinstance(x, torch.Tensor):
        x = (
            x.detach()
            .cpu()
            .numpy()
        )

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.ndim != 2:
        raise ValueError(
            "Intrinsic-dimension input must have shape "
            "[num_examples, dimensions]"
        )

    if len(x) < 10:
        return float("nan")

    # Remove nonfinite rows.
    finite_mask = np.all(
        np.isfinite(x),
        axis=1,
    )

    x = x[finite_mask]

    if len(x) < 10:
        return float("nan")

    # TwoNN can fail when all observations are identical.
    if np.allclose(
        x,
        x[0],
    ):
        return float("nan")

    try:
        estimator = TwoNN()
        estimator.fit(x)

        return float(
            estimator.dimension_
        )

    except Exception as error:
        print(
            "Intrinsic-dimension estimation failed: "
            f"{error}"
        )

        return float("nan")


# --------------------------------------------------
# Transformer activation extraction
# --------------------------------------------------

def get_task_components(
    model,
    task,
):
    """
    Return the task token and output head for one task.
    """

    if task == "distance":
        return (
            model.distance_token,
            model.distance_head,
        )

    if task == "nearest":
        return (
            model.nearest_token,
            model.nearest_head,
        )

    raise ValueError(
        f"Unknown task: {task}"
    )


def get_transformer_activations(
    model,
    point_i,
    point_j,
    task,
):
    """
    Run one batch through the transformer and save intermediate
    token representations and output-head activations.

    Each transformer sequence is:

        [task token, entity i, entity j]
    """

    task_token, head = get_task_components(
        model,
        task,
    )

    with torch.inference_mode():
        # Look up entity embeddings.
        embedding_i = model.encode(
            point_i
        )

        embedding_j = model.encode(
            point_j
        )

        batch_size = point_i.shape[0]

        # Expand the learned task token across the batch.
        expanded_task_token = task_token.expand(
            batch_size,
            -1,
            -1,
        )

        # Add sequence dimensions to entity embeddings.
        embedding_i_token = (
            embedding_i.unsqueeze(1)
        )

        embedding_j_token = (
            embedding_j.unsqueeze(1)
        )

        # Construct:
        #   [batch_size, 3, emb_dim]
        tokens = torch.cat(
            [
                expanded_task_token,
                embedding_i_token,
                embedding_j_token,
            ],
            dim=1,
        )

        activations = {
            # Complete initial sequence.
            "input_tokens": tokens,

            # Individual initial tokens.
            "input_task": tokens[:, 0, :],
            "input_i": tokens[:, 1, :],
            "input_j": tokens[:, 2, :],
        }

        hidden = tokens

        # Run the sequence through each transformer encoder layer
        # individually so intermediate representations can be saved.
        for layer_index, layer in enumerate(
            model.transformer.layers,
            start=1,
        ):
            hidden = layer(hidden)

            activations[
                f"transformer_{layer_index}_tokens"
            ] = hidden

            activations[
                f"transformer_{layer_index}_task"
            ] = hidden[:, 0, :]

            activations[
                f"transformer_{layer_index}_i"
            ] = hidden[:, 1, :]

            activations[
                f"transformer_{layer_index}_j"
            ] = hidden[:, 2, :]

            # A flattened representation of the complete three-token
            # sequence can be used in CKA if desired.
            activations[
                f"transformer_{layer_index}_sequence"
            ] = hidden.flatten(
                start_dim=1
            )

        # Apply the encoder's optional final normalization.
        if model.transformer.norm is not None:
            hidden = model.transformer.norm(
                hidden
            )

        activations[
            "transformer_final_tokens"
        ] = hidden

        activations[
            "transformer_final_sequence"
        ] = hidden.flatten(
            start_dim=1
        )

        # The transformed task token is the pair representation used
        # by the output head.
        pair = hidden[:, 0, :]

        activations["pair"] = pair

        # TransformerHead structure:
        #
        #   net[0] = LayerNorm
        #   net[1] = Linear
        #   net[2] = ReLU
        #   net[3] = Linear
        #   net[4] = ReLU
        #   net[5] = Linear
        net = head.net

        head_normalized = net[0](
            pair
        )

        h1_pre = net[1](
            head_normalized
        )

        h1 = net[2](
            h1_pre
        )

        h2_pre = net[3](
            h1
        )

        h2 = net[4](
            h2_pre
        )

        output = net[5](
            h2
        ).squeeze(-1)

        activations[
            "head_normalized"
        ] = head_normalized

        activations[
            "h1_pre"
        ] = h1_pre

        activations[
            "h1"
        ] = h1

        activations[
            "h2_pre"
        ] = h2_pre

        activations[
            "h2"
        ] = h2

        activations[
            "output"
        ] = output

    return activations


# --------------------------------------------------
# Plotting utilities
# --------------------------------------------------

def plot_cka_matrix(
    activations,
    layer_names,
    title,
):
    """
    Calculate and display a CKA matrix.
    """

    cka_matrix = np.zeros(
        (
            len(layer_names),
            len(layer_names),
        ),
        dtype=np.float64,
    )

    for row, name_a in enumerate(
        layer_names
    ):
        for column, name_b in enumerate(
            layer_names
        ):
            cka_matrix[row, column] = (
                linear_cka(
                    activations[name_a],
                    activations[name_b],
                )
            )

    print(f"\n{title}")
    print(cka_matrix)

    plt.figure(
        figsize=(
            max(7, len(layer_names) * 0.9),
            max(6, len(layer_names) * 0.8),
        )
    )

    image = plt.imshow(
        cka_matrix,
        cmap="viridis",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )

    plt.xticks(
        range(len(layer_names)),
        layer_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(layer_names)),
        layer_names,
    )

    plt.colorbar(
        image,
        label="Linear CKA",
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()

    return cka_matrix


def plot_intrinsic_dimensions(
    activations,
    layer_names,
    title,
):
    """
    Estimate and plot intrinsic dimension for selected layers.
    """

    dimensions = []

    for layer_name in layer_names:
        dimension = (
            estimate_intrinsic_dimension(
                activations[layer_name]
            )
        )

        dimensions.append(
            dimension
        )

    print(f"\n{title}")

    for layer_name, dimension in zip(
        layer_names,
        dimensions,
    ):
        print(
            f"  {layer_name}: "
            f"{dimension:.4f}"
        )

    plt.figure(
        figsize=(
            max(7, len(layer_names) * 0.9),
            5,
        )
    )

    plt.plot(
        layer_names,
        dimensions,
        marker="o",
    )

    plt.title(title)
    plt.xlabel("Representation")
    plt.ylabel(
        "Estimated intrinsic dimension"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()
    plt.show()

    return dimensions


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

world = build_world_from_config(
    cfg
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

probe_targets, probe_target_name = (
    get_probe_targets(world)
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
        probe_targets,
    )
)

cv_r2_mean, cv_r2_std = (
    cross_validated_coordinate_probe(
        embeddings,
        probe_targets,
        n_splits=5,
        seed=SEED,
    )
)

true_target_plot = project_to_2d(
    probe_targets
)

predicted_target_plot = project_to_2d(
    linear_prediction
)

latent_distance_matrix = (
    pairwise_distances(
        embeddings
    )
)

# The grid distance task uses Euclidean distance.
true_distance_matrix = (
    true_world_distance_matrix(
        world
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
    nearest_neighbor_recall_from_matrix(
        embeddings,
        true_distance_matrix,
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
    "Geodesic-distance Spearman correlation: "
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
    predicted_target_plot[:, 0],
    predicted_target_plot[:, 1],
    s=15,
)

plt.title(
    f"Reconstructed {probe_target_name} "
    f"(R²={coordinate_r2:.3f})"
)

plt.xlabel("Projection 1")
plt.ylabel("Projection 2")
plt.axis("equal")
plt.tight_layout()
plt.show()


# True coordinates
plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    true_target_plot[:, 0],
    true_target_plot[:, 1],
    s=15,
)

plt.title(
    f"True {probe_target_name}"
)

plt.xlabel("Projection 1")
plt.ylabel("Projection 2")
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