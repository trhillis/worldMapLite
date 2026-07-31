"""
Reusable representation-analysis utilities: CKA, intrinsic dimension,
coordinate probing, and MultiTaskWorldModel activation extraction.

None of these functions assume anything about the world that produced the
entity indices (grid, sphere, or manifold) - they operate on arbitrary
[num_examples, dimensions] arrays and on MultiTaskWorldModel's architecture
(via duck-typed access to model.emb / model.transformer / a task token and
head). This is what lets both the retired grid pipeline (legacy/) and the
manifold pipeline (analysis/analysis_manifold.py) share them without
duplication.
"""

from pathlib import Path

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


def nearest_neighbor_recall(
    embeddings,
    coordinates,
):
    """
    Measure whether the nearest entity in learned embedding space
    is one of the true nearest entities in coordinate space.

    Both distances are Euclidean.
    """

    latent_distances = pairwise_distances(
        embeddings
    )

    true_distances = pairwise_distances(
        coordinates
    )

    # Prevent each point from selecting itself.
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
            true_distances[point_index].min()
        )

        true_neighbors = set(
            np.flatnonzero(
                np.isclose(
                    true_distances[point_index],
                    minimum_true_distance,
                    atol=1e-8,
                    rtol=0.0,
                )
            )
        )

        predicted_neighbor = int(
            np.argmin(
                latent_distances[point_index]
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


def compute_embedding_metrics(
    embeddings,
    true_coordinates,
    true_distance_upper=None,
    seed=0,
    n_splits=5,
):
    """
    Bundle the cheap, transformer-forward-free representation-quality
    metrics into one call: PCA explained variance, linear and
    cross-validated coordinate-probe R², geodesic-distance Spearman
    correlation, nearest-neighbor recall, and embedding intrinsic
    dimension. All operate directly on the embedding table and the true
    ambient coordinates, so this is cheap enough to call repeatedly
    (e.g. a periodic snapshot during training), unlike anything that
    needs `get_transformer_activations`.

    `true_distance_upper` (the upper-triangle of the true pairwise
    distance matrix) can be precomputed once by a caller that calls this
    repeatedly against the same `true_coordinates`, since it does not
    depend on `embeddings`.
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    true_coordinates = np.asarray(true_coordinates, dtype=np.float64)

    n_components = min(3, true_coordinates.shape[1])

    pca_explained_variance = float(
        PCA(n_components=n_components)
        .fit(embeddings)
        .explained_variance_ratio_
        .sum()
    )

    _, linear_r2 = linear_coordinate_probe(
        embeddings,
        true_coordinates,
    )

    cv_r2_mean, cv_r2_std = cross_validated_coordinate_probe(
        embeddings,
        true_coordinates,
        n_splits=n_splits,
        seed=seed,
    )

    if true_distance_upper is None:
        true_distance_upper = upper_triangle_values(
            pairwise_distances(true_coordinates)
        )

    latent_distance_upper = upper_triangle_values(
        pairwise_distances(embeddings)
    )

    distance_spearman = float(
        spearmanr(
            latent_distance_upper,
            true_distance_upper,
        ).statistic
    )

    nn_recall = nearest_neighbor_recall(
        embeddings,
        true_coordinates,
    )

    intrinsic_dim = estimate_intrinsic_dimension(
        embeddings
    )

    return {
        "pca_explained_variance": pca_explained_variance,
        "linear_r2": linear_r2,
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "distance_spearman": distance_spearman,
        "nn_recall": nn_recall,
        "intrinsic_dim": intrinsic_dim,
    }


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

    if task == "same_triangle":
        return (
            model.same_triangle_token,
            model.same_triangle_head,
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
    save_path=None,
):
    """
    Calculate and display a CKA matrix.

    If `save_path` is given, the figure is also written there (parent
    directories are created as needed) before it is shown.
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

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    return cka_matrix


def plot_intrinsic_dimensions(
    activations,
    layer_names,
    title,
    save_path=None,
):
    """
    Estimate and plot intrinsic dimension for selected layers.

    If `save_path` is given, the figure is also written there (parent
    directories are created as needed) before it is shown.
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

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    return dimensions


def plot_training_progress(
    progress_rows,
    title,
    save_path=None,
):
    """
    Plot loss and the metrics from `compute_embedding_metrics`, each
    against training step, from a list of per-snapshot dicts collected
    during training. src/train_manifold.py additionally tags every active
    task's held-out/in-distribution evaluation metrics with a
    "{task}_held_out_{metric}" / "{task}_in_distribution_{metric}" key
    (e.g. distance_held_out_spearman, same_triangle_held_out_accuracy) -
    discovered here dynamically (not hardcoded to "distance"), so one
    extra panel is added per (task, metric) pair actually present,
    overlaying held-out vs. in-distribution over training so the
    generalization gap (or lack of one) is visible across steps, not just
    at the final checkpoint. Snapshots without any such keys (e.g. from
    before this feature existed) simply get no extra panels.

    If `save_path` is given, the figure is also written there (parent
    directories are created as needed) before it is shown.
    """

    steps = [row["step"] for row in progress_rows]

    panels = [
        ("loss", "Training loss", True),
        ("pca_explained_variance", "PCA explained variance", False),
        ("linear_r2", "Linear coordinate probe R²", False),
        ("cv_r2_mean", "Cross-validated coordinate R²", False),
        ("distance_spearman", "Geodesic-distance Spearman corr.", False),
        ("nn_recall", "Nearest-neighbor recall", False),
        ("intrinsic_dim", "Embedding intrinsic dimension", False),
    ]

    # Discover which (task, metric) held-out/in-distribution pairs this
    # run actually recorded, e.g. {"distance": ["loss", "spearman"],
    # "same_triangle": ["loss", "accuracy", "auc"]}.
    task_metrics = {}
    for key in progress_rows[0]:
        if "_held_out_" not in key:
            continue
        task, metric = key.split("_held_out_", 1)
        task_metrics.setdefault(task, []).append(metric)

    extra_panels = [
        (task, metric)
        for task in sorted(task_metrics)
        for metric in sorted(task_metrics[task])
    ]

    num_panels = len(panels) + len(extra_panels)
    num_columns = 4
    num_rows = -(-num_panels // num_columns)  # ceil division

    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(4.5 * num_columns, 4 * num_rows),
    )

    axes = axes.flatten()

    for axis, (key, label, log_scale) in zip(axes, panels):
        values = [row[key] for row in progress_rows]

        axis.plot(
            steps,
            values,
            marker="o",
        )

        # Show the fold std as a shaded band around the cross-validated
        # mean, since it's the one metric with an uncertainty estimate.
        if key == "cv_r2_mean":
            stds = [row["cv_r2_std"] for row in progress_rows]

            axis.fill_between(
                steps,
                [value - std for value, std in zip(values, stds)],
                [value + std for value, std in zip(values, stds)],
                alpha=0.2,
            )

        if log_scale:
            axis.set_yscale("log")

        axis.set_title(label)
        axis.set_xlabel("step")

    next_axis_index = len(panels)

    for offset, (task, metric) in enumerate(extra_panels):
        axis = axes[next_axis_index + offset]

        axis.plot(
            steps,
            [row[f"{task}_held_out_{metric}"] for row in progress_rows],
            marker="o",
            label="held-out",
        )
        axis.plot(
            steps,
            [row[f"{task}_in_distribution_{metric}"] for row in progress_rows],
            marker="o",
            label="in-distribution",
        )

        if metric == "loss":
            axis.set_yscale("log")

        axis.set_title(f"{task}: held-out vs. in-distribution {metric}")
        axis.set_xlabel("step")
        axis.legend()

    next_axis_index += len(extra_panels)

    # Turn off any unused trailing slots in the panel grid.
    for axis in axes[next_axis_index:]:
        axis.axis("off")

    figure.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
