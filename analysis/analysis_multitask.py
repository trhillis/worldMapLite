import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
import torch
import matplotlib.pyplot as plt

import json
import hashlib
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from scipy.stats import spearmanr
from skdim.id import TwoNN

from src.worlds import make_grid, make_manifold_world, subset_world

from manifolds.flat_torus import FlatTorus
from manifolds.mobius import FlatMobiusStrip

from manifolds.polyhedra import octahedron

from src.multitask_model import MultiTaskWorldModel

from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT_DIR = Path("models")
OUTPUT_DIR = Path("analysis_results")
PLOT_DIR = OUTPUT_DIR / "plots"
CACHE_DIR = OUTPUT_DIR / "distance_cache"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Adjust this pattern to match your exact checkpoint filenames.
checkpoint_paths = sorted(
    CHECKPOINT_DIR.glob("*_distance_*_model.pt")
)

PAIR_SAMPLE_SIZE = 5000
DISTANCE_SAMPLE_SIZE = 10_000

# Use the same fixed evaluation pairs for every model.
EVALUATION_SEED = 10_000

print(
    f"Found {len(checkpoint_paths)} checkpoints"
)

for path in checkpoint_paths:
    print(f"  {path}")


def get_world_name(cfg):
    """
    Return one consistent world label for tables and paths.
    """

    if cfg.get("world_type", "grid") == "grid":
        return "grid"

    return cfg["manifold"]

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
        manifold = octahedron()

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


def true_world_distance_matrix(
    world,
    world_name,
    seed,
):
    """
    Load or compute the complete true-distance matrix.
    """

    point_ids = world.meta.get("original_point_ids", range(len(world.names)))
    point_digest = hashlib.sha256(
        np.asarray(list(point_ids), dtype=np.int64).tobytes()
    ).hexdigest()[:10]
    cache_path = (
        CACHE_DIR
        / f"{world_name}_seed{seed}_n{len(world.names)}_{point_digest}.npy"
    )

    if cache_path.exists():
        return np.load(
            cache_path
        )

    world_type = world.meta["type"]

    if world_type == "grid":
        matrix = pairwise_distances(
            world.coordinates
        )

    elif world_type == "manifold":
        if world.manifold is None:
            raise ValueError(
                "Manifold world has no manifold object"
            )

        matrix = np.asarray(
            world.manifold.distance_matrix(
                world.coordinates
            ),
            dtype=np.float64,
        )

    else:
        raise ValueError(
            "True distance matrix is not implemented "
            f"for world type {world_type}"
        )

    np.save(
        cache_path,
        matrix,
    )

    return matrix

def get_distance_scale(world):
    """
    Return the normalization scale used during training.
    """

    if "diameter" in world.meta:
        return float(
            world.meta["diameter"]
        )

    world_type = world.meta["type"]

    if world_type == "grid":
        width = world.meta["width"]
        height = world.meta["height"]

        return float(
            np.sqrt(
                (width - 1) ** 2
                + (height - 1) ** 2
            )
        )

    raise ValueError(
        "No distance scale is available for "
        f"{world.meta}"
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

def sampled_distance_correlation(
    world,
    embeddings,
    n_pairs=10_000,
    seed=0,
):
    """
    Estimate Spearman correlation between latent distances and
    true world distances using randomly sampled unique pairs.

    This avoids constructing the full N x N geodesic-distance
    matrix, which is expensive for polyhedral manifolds.
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float64,
    )

    pairs = sample_unique_pairs(
        num_points=len(embeddings),
        n_pairs=n_pairs,
        seed=seed,
    )

    point_i = pairs[:, 0]
    point_j = pairs[:, 1]

    # Euclidean distances in learned embedding space.
    latent_distances = np.linalg.norm(
        embeddings[point_i]
        - embeddings[point_j],
        axis=1,
    )

    coordinates = np.asarray(
        world.coordinates
    )

    if world.meta["type"] == "grid":
        true_distances = np.linalg.norm(
            coordinates[point_i]
            - coordinates[point_j],
            axis=1,
        )

    elif world.meta["type"] == "manifold":
        if world.manifold is None:
            raise ValueError(
                "Manifold world has no manifold object"
            )

        # Calculate only the sampled manifold distances.
        true_distances = np.asarray(
            world.manifold.distance(
                coordinates[point_i],
                coordinates[point_j],
            ),
            dtype=np.float64,
        )

    else:
        raise ValueError(
            "Distance correlation is not implemented for "
            f"world type {world.meta['type']}"
        )

    correlation = spearmanr(
        latent_distances,
        true_distances,
    ).statistic

    return float(correlation)

def sampled_prediction_metrics(
    model,
    world,
    n_pairs=10_000,
    seed=0,
):
    """
    Evaluate the transformer's predicted distances on sampled pairs.
    """

    pairs = sample_unique_pairs(
        num_points=len(world.names),
        n_pairs=n_pairs,
        seed=seed,
    )

    point_i_numpy = pairs[:, 0]
    point_j_numpy = pairs[:, 1]

    point_i = torch.tensor(
        point_i_numpy,
        dtype=torch.long,
    )

    point_j = torch.tensor(
        point_j_numpy,
        dtype=torch.long,
    )

    coordinates = np.asarray(
        world.coordinates
    )

    if world.meta["type"] == "grid":
        true_raw = np.linalg.norm(
            coordinates[point_i_numpy]
            - coordinates[point_j_numpy],
            axis=1,
        )

    elif world.meta["type"] == "manifold":
        true_raw = np.asarray(
            world.manifold.distance(
                coordinates[point_i_numpy],
                coordinates[point_j_numpy],
            ),
            dtype=np.float64,
        )

    else:
        raise ValueError(
            "Prediction evaluation is not implemented for "
            f"{world.meta['type']}"
        )

    scale = get_distance_scale(world)

    true_normalized = (
        true_raw / scale
    )

    with torch.inference_mode():
        predicted_normalized = (
            model.forward_distance(
                point_i,
                point_j,
            )
            .detach()
            .cpu()
            .numpy()
        )

    errors = (
        predicted_normalized
        - true_normalized
    )

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    prediction_spearman = float(
        spearmanr(
            predicted_normalized,
            true_normalized,
        ).statistic
    )

    return {
        "prediction_mae": mae,
        "prediction_rmse": rmse,
        "prediction_spearman": (
            prediction_spearman
        ),
    }

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
    output_path,
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
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return cka_matrix


def plot_intrinsic_dimensions(
    activations,
    layer_names,
    title,
    output_path,
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
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return dimensions


def analyze_checkpoint(
    checkpoint_path,
):
    """
    Analyze one trained model and return one results row.
    """

    print(
        f"\nAnalyzing {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        # These checkpoints are produced locally by train_multitask.py and
        # contain configuration metadata plus the saved training-pair array,
        # not only model weights. PyTorch 2.6 changed this default to True.
        weights_only=False,
    )

    cfg = checkpoint["config"]

    tasks = cfg.get(
        "tasks",
        ("distance",),
    )

    if isinstance(tasks, str):
        tasks = (tasks,)
    else:
        tasks = tuple(tasks)

    world_name = get_world_name(
        cfg
    )

    seed = int(
        cfg["seed"]
    )

    relation_budget = int(
        cfg.get(
            "train_examples_per_task",
            0,
        )
    )

    distance_pair_seed = cfg.get(
        "pair_split_seed"
    )
    if distance_pair_seed is None:
        distance_pair_seed = cfg.get("distance_pair_seed")

    training_step = int(
        checkpoint.get(
            "training_step",
            cfg.get("steps", 0),
        )
    )

    run_name = (
        f"{world_name}_pairs{relation_budget}_pairseed"
        f"{distance_pair_seed}_seed{seed}"
    )
    if cfg.get("held_out_points", 0) or cfg.get("held_out_point_fraction"):
        amount = (
            f"frac{cfg['held_out_point_fraction']:g}"
            if cfg.get("held_out_point_fraction") is not None
            else str(cfg.get("held_out_points", 0))
        )
        run_name += f"_pointholdout{amount}_pointseed{cfg.get('held_out_point_seed', 0)}"

    if (
        f"_step{training_step}_model"
        in Path(checkpoint_path).name
    ):
        run_name = (
            f"{run_name}_step{training_step}"
        )

    run_plot_dir = (
        PLOT_DIR / run_name
    )

    run_plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    world = build_world_from_config(
        cfg
    )

    point_split = checkpoint.get("point_split")
    if point_split and point_split.get("retained_points") is not None:
        world = subset_world(world, point_split["retained_points"])

    # Prefer exact saved coordinates when checkpoints contain them.
    if "world_coordinates" in checkpoint:
        world.coordinates = np.asarray(
            checkpoint[
                "world_coordinates"
            ]
        )

    if (
        "ambient_coordinates"
        in checkpoint
        and checkpoint[
            "ambient_coordinates"
        ] is not None
    ):
        world.ambient_coordinates = (
            np.asarray(
                checkpoint[
                    "ambient_coordinates"
                ]
            )
        )

    model = MultiTaskWorldModel(
        num_points=len(world.names),
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
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
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    num_points = len(
        world.names
    )

    embeddings = (
        model.emb.weight
        .detach()
        .cpu()
        .numpy()
    )

    probe_targets, probe_target_name = (
        get_probe_targets(
            world
        )
    )

    # ----------------------------------------------
    # Entity metrics
    # ----------------------------------------------

    pca = PCA(
        n_components=2
    )

    embedding_pca = pca.fit_transform(
        embeddings
    )

    is_octahedron = world_name in {
        "octahedron",
        "regular_octahedron",
    }

    if is_octahedron:
        embedding_pca_3d = PCA(
            n_components=3
        ).fit_transform(embeddings)

        face_index = world.coordinates[:, 0].astype(int)
        n_faces = world.manifold.n_faces

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
            seed=EVALUATION_SEED,
        )
    )

    true_target_plot = project_to_2d(
        probe_targets
    )

    predicted_target_plot = project_to_2d(
        linear_prediction
    )

    embedding_distance_spearman = (
        sampled_distance_correlation(
            world=world,
            embeddings=embeddings,
            n_pairs=DISTANCE_SAMPLE_SIZE,
            seed=EVALUATION_SEED,
        )
    )

    history = checkpoint.get("evaluation", {}).get("distance_history", [])
    exact_entry = next(
        (entry for entry in reversed(history) if int(entry["step"]) == training_step),
        history[-1] if history else None,
    )
    if exact_entry is not None:
        held = exact_entry["held_out"]
        trained = exact_entry["train"]
        prediction_results = {
            "prediction_mae": held["normalized_mae"],
            "prediction_rmse": held["normalized_rmse"],
            "prediction_pearson": held.get("pearson", float("nan")),
            "prediction_spearman": held["spearman"],
            **{f"held_out_pair_{key}": value for key, value in held.items()},
            **{f"training_pair_{key}": value for key, value in trained.items()},
        }
    else:
        prediction_results = sampled_prediction_metrics(
            model=model,
            world=world,
            n_pairs=DISTANCE_SAMPLE_SIZE,
            seed=EVALUATION_SEED,
        )

    true_distance_matrix = (
        true_world_distance_matrix(
            world=world,
            world_name=world_name,
            seed=seed,
        )
    )

    neighbor_recall = (
        nearest_neighbor_recall_from_matrix(
            embeddings,
            true_distance_matrix,
        )
    )

    embedding_id = (
        estimate_intrinsic_dimension(
            embeddings
        )
    )

    # ----------------------------------------------
    # Save entity plots
    # ----------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        embedding_pca[:, 0],
        embedding_pca[:, 1],
        s=15,
    )

    plt.title(
        f"{run_name}: embedding PCA"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    plt.savefig(
        run_plot_dir
        / "embedding_pca.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        predicted_target_plot[:, 0],
        predicted_target_plot[:, 1],
        s=15,
    )

    plt.title(
        f"{run_name}: reconstructed "
        f"{probe_target_name}"
    )

    plt.xlabel("Projection 1")
    plt.ylabel("Projection 2")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        run_plot_dir
        / "probe_reconstruction.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        true_target_plot[:, 0],
        true_target_plot[:, 1],
        s=15,
    )

    plt.title(
        f"{run_name}: true "
        f"{probe_target_name}"
    )

    plt.xlabel("Projection 1")
    plt.ylabel("Projection 2")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        run_plot_dir
        / "true_targets.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    if is_octahedron:
        fig = plt.figure(
            figsize=(8, 6)
        )
        ax = fig.add_subplot(
            projection="3d"
        )

        scatter = ax.scatter(
            embedding_pca_3d[:, 0],
            embedding_pca_3d[:, 1],
            embedding_pca_3d[:, 2],
            c=face_index,
            cmap="tab10",
            vmin=-0.5,
            vmax=n_faces - 0.5,
            s=15,
        )

        ax.set_title(
            f"{run_name}: embedding PCA (3D)"
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        fig.colorbar(
            scatter,
            ax=ax,
            ticks=range(n_faces),
            label="octahedron face",
        )

        fig.savefig(
            run_plot_dir
            / "embedding_pca_3d.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        fig = plt.figure(
            figsize=(8, 6)
        )
        ax = fig.add_subplot(
            projection="3d"
        )

        scatter = ax.scatter(
            linear_prediction[:, 0],
            linear_prediction[:, 1],
            linear_prediction[:, 2],
            c=face_index,
            cmap="tab10",
            vmin=-0.5,
            vmax=n_faces - 0.5,
            s=15,
        )

        ax.set_title(
            f"{run_name}: reconstructed "
            f"{probe_target_name} (3D)"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        fig.colorbar(
            scatter,
            ax=ax,
            ticks=range(n_faces),
            label="octahedron face",
        )

        fig.savefig(
            run_plot_dir
            / "probe_reconstruction_3d.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        fig = plt.figure(
            figsize=(8, 6)
        )
        ax = fig.add_subplot(
            projection="3d"
        )

        scatter = ax.scatter(
            probe_targets[:, 0],
            probe_targets[:, 1],
            probe_targets[:, 2],
            c=face_index,
            cmap="tab10",
            vmin=-0.5,
            vmax=n_faces - 0.5,
            s=15,
        )

        ax.set_title(
            f"{run_name}: true "
            f"{probe_target_name} (3D)"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        fig.colorbar(
            scatter,
            ax=ax,
            ticks=range(n_faces),
            label="octahedron face",
        )

        fig.savefig(
            run_plot_dir
            / "true_targets_3d.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ----------------------------------------------
    # Transformer representations
    # ----------------------------------------------

    pairs = sample_unique_pairs(
        num_points=num_points,
        n_pairs=PAIR_SAMPLE_SIZE,
        seed=EVALUATION_SEED,
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

    transformer_task_layers = [
        f"transformer_{index}_task"
        for index in range(
            1,
            len(
                model.transformer.layers
            ) + 1,
        )
    ]

    transformer_sequence_layers = [
        f"transformer_{index}_sequence"
        for index in range(
            1,
            len(
                model.transformer.layers
            ) + 1,
        )
    ]

    result = {
        "checkpoint": str(
            checkpoint_path
        ),
        "world": world_name,
        "seed": seed,
        "relation_budget": relation_budget,
        "distance_pair_seed": distance_pair_seed,
        "pair_split_seed": cfg.get("pair_split_seed", distance_pair_seed),
        "world_seed": cfg.get("world_seed") if cfg.get("world_seed") is not None else seed,
        "data_order_seed": cfg.get("data_order_seed") if cfg.get("data_order_seed") is not None else seed,
        "held_out_point_seed": cfg.get("held_out_point_seed"),
        "pair_split_digest": (
            checkpoint.get("pair_split") or {}
        ).get("split_digest"),
        "held_out_pair_digest": (
            checkpoint.get("pair_split") or {}
        ).get("held_out_pair_digest"),
        "point_split_digest": (
            checkpoint.get("point_split") or {}
        ).get("point_split_digest"),
        "training_step": training_step,
        "num_points": num_points,
        "tasks": "_".join(tasks),
        "pca_explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "probe_r2": coordinate_r2,
        "cv_probe_r2_mean": cv_r2_mean,
        "cv_probe_r2_std": cv_r2_std,
        "embedding_distance_spearman": (
            embedding_distance_spearman
        ),
        "nearest_recall": neighbor_recall,
        "embedding_intrinsic_dimension": (
            embedding_id
        ),
        **prediction_results,
    }

    for task, activations in (
        task_activations.items()
    ):
        task_layer_names = [
            *transformer_task_layers,
            "pair",
            "h1",
            "h2",
        ]

        cka_matrix = plot_cka_matrix(
            activations=activations,
            layer_names=task_layer_names,
            title=(
                f"{run_name}: {task} "
                "task-token/head CKA"
            ),
            output_path=(
                run_plot_dir
                / f"{task}_task_cka.png"
            ),
        )

        dimensions = (
            plot_intrinsic_dimensions(
                activations=activations,
                layer_names=task_layer_names,
                title=(
                    f"{run_name}: {task} "
                    "representation dimensions"
                ),
                output_path=(
                    run_plot_dir
                    / f"{task}_dimensions.png"
                ),
            )
        )

        for layer_name, dimension in zip(
            task_layer_names,
            dimensions,
        ):
            result[
                f"{task}_{layer_name}_id"
            ] = dimension

        if len(transformer_task_layers) >= 2:
            result[
                f"{task}_task_layer1_layer2_cka"
            ] = float(
                cka_matrix[0, 1]
            )

        if (
            len(
                transformer_sequence_layers
            )
            >= 2
        ):
            sequence_cka = plot_cka_matrix(
                activations=activations,
                layer_names=(
                    transformer_sequence_layers
                ),
                title=(
                    f"{run_name}: {task} "
                    "sequence CKA"
                ),
                output_path=(
                    run_plot_dir
                    / f"{task}_sequence_cka.png"
                ),
            )

            result[
                f"{task}_sequence_layer1_layer2_cka"
            ] = float(
                sequence_cka[0, 1]
            )

        for layer_index in range(
            1,
            len(
                model.transformer.layers
            ) + 1,
        ):
            token_similarity = linear_cka(
                activations[
                    f"transformer_"
                    f"{layer_index}_i"
                ],
                activations[
                    f"transformer_"
                    f"{layer_index}_j"
                ],
            )

            result[
                f"{task}_entity_token_"
                f"cka_layer{layer_index}"
            ] = token_similarity

    # ----------------------------------------------
    # Symmetry
    # ----------------------------------------------

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

            difference = torch.abs(
                forward_prediction
                - reverse_prediction
            )

            result[
                f"{task}_symmetry_mean"
            ] = float(
                difference.mean().item()
            )

            result[
                f"{task}_symmetry_max"
            ] = float(
                difference.max().item()
            )

    with open(
        run_plot_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    return result

def main():
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found in "
            f"{CHECKPOINT_DIR.resolve()}"
        )

    result_rows = []

    for checkpoint_path in (
        checkpoint_paths
    ):
        try:
            result = analyze_checkpoint(
                checkpoint_path
            )

            result_rows.append(
                result
            )

        except Exception as error:
            print(
                f"FAILED: {checkpoint_path}"
            )

            print(
                f"  {type(error).__name__}: "
                f"{error}"
            )

    if not result_rows:
        raise RuntimeError(
            "No checkpoints were analyzed successfully"
        )

    results = pd.DataFrame(
        result_rows
    )

    results = results.sort_values(
        [
            "world",
            "relation_budget",
            "seed",
            "training_step",
        ]
    )

    results_path = (
        OUTPUT_DIR / "all_runs.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    numeric_columns = [
        column
        for column in results.columns
        if (
            column
            not in {
                "checkpoint",
                "world",
                "tasks",
                "relation_budget",
                "distance_pair_seed",
                "training_step",
            }
            and pd.api.types.is_numeric_dtype(
                results[column]
            )
        )
    ]

    summary = (
        results
        .groupby(
            [
                "world",
                "relation_budget",
                "training_step",
            ]
        )[
            numeric_columns
        ]
        .agg(
            ["mean", "std"]
        )
    )

    summary_path = (
        OUTPUT_DIR
        / "world_summary.csv"
    )

    summary.to_csv(
        summary_path
    )

    print(
        f"\nSaved individual results to "
        f"{results_path}"
    )

    print(
        f"Saved world summary to "
        f"{summary_path}"
    )

    print("\nWorld summary:")
    print(summary)


if __name__ == "__main__":
    main()
