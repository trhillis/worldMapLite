import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from skdim.id import TwoNN

from worlds import make_grid
from multitask_model import MultiTaskWorldModel

from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

CHECKPOINT_PATH = "models/distance_nearest_model.pt"
PAIR_SAMPLE_SIZE = 5000
SEED = 0
print(CHECKPOINT_PATH)

def linear_cka(x, y):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    numerator = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = (
        np.linalg.norm(x.T @ x, ord="fro")
        * np.linalg.norm(y.T @ y, ord="fro")
    )

    return float(numerator / (denominator + 1e-12))


def sample_unique_pairs(num_points, n_pairs, seed=0):
    rng = np.random.default_rng(seed)

    all_pairs = np.array([
        (i, j)
        for i in range(num_points)
        for j in range(i + 1, num_points)
    ])

    n_pairs = min(n_pairs, len(all_pairs))

    selected = rng.choice(
        len(all_pairs),
        size=n_pairs,
        replace=False,
    )

    return all_pairs[selected]


def get_head_activations(model, point_i, point_j, task):
    with torch.no_grad():
        pair = model.pair_representation(point_i, point_j)

        if task == "distance":
            net = model.distance_head.net
        elif task == "nearest":
            net = model.nearest_head.net
        else:
            raise ValueError(f"Unknown task: {task}")

        h1 = net[1](net[0](pair))
        h2 = net[3](net[2](h1))
        output = net[4](h2)

    return {
        "pair": pair,
        "h1": h1,
        "h2": h2,
        "output": output,
    }


def pairwise_distances(x):
    difference = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(difference, axis=-1)


def upper_triangle_values(matrix):
    indices = np.triu_indices_from(matrix, k=1)
    return matrix[indices]


def linear_coordinate_probe(embeddings, coordinates):
    probe = LinearRegression()
    probe.fit(embeddings, coordinates)

    prediction = probe.predict(embeddings)

    return prediction, r2_score(
        coordinates,
        prediction,
        multioutput="variance_weighted",
    )


def nearest_neighbor_recall(embeddings, coordinates):
    latent_distances = pairwise_distances(embeddings)

    # Manhattan distance is the grid's natural distance.
    true_distances = np.abs(
        coordinates[:, None, :] - coordinates[None, :, :]
    ).sum(axis=-1)

    np.fill_diagonal(latent_distances, np.inf)
    np.fill_diagonal(true_distances, np.inf)

    recovered = []

    for i in range(len(embeddings)):
        true_min = true_distances[i].min()
        true_neighbors = set(
            np.flatnonzero(true_distances[i] == true_min)
        )

        predicted_neighbor = int(
            np.argmin(latent_distances[i])
        )

        recovered.append(
            predicted_neighbor in true_neighbors
        )

    return float(np.mean(recovered))


def estimate_intrinsic_dimension(x):
    x = np.asarray(x)

    # TwoNN requires enough distinct samples and can fail on
    # degenerate or duplicate-heavy representations.
    if len(x) < 10:
        return float("nan")

    return float(TwoNN().fit(x).dimension_)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
)

cfg = checkpoint["config"]

tasks = cfg.get("tasks", ("distance", "nearest"))

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
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

num_points = len(world.names)
true_coordinates = world.coordinates.astype(np.float64)

embeddings = (
    model.emb.weight
    .detach()
    .cpu()
    .numpy()
)

# --------------------------------------------------
# Entity embedding analysis
# --------------------------------------------------

pca = PCA(n_components=2)
embedding_pca = pca.fit_transform(embeddings)

linear_prediction, coordinate_r2 = (
    linear_coordinate_probe(
        embeddings,
        true_coordinates,
    )
)

latent_distance_matrix = pairwise_distances(embeddings)

true_manhattan_matrix = np.abs(
    true_coordinates[:, None, :]
    - true_coordinates[None, :, :]
).sum(axis=-1)

distance_correlation = spearmanr(
    upper_triangle_values(latent_distance_matrix),
    upper_triangle_values(true_manhattan_matrix),
).statistic

neighbor_recall = nearest_neighbor_recall(
    embeddings,
    true_coordinates,
)

embedding_id = estimate_intrinsic_dimension(
    embeddings
)

print("Entity embedding metrics")
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
print(f"Linear coordinate probe R²: {coordinate_r2:.4f}")
print(f"Distance Spearman correlation: {distance_correlation:.4f}")
print(f"Nearest-neighbor recall: {neighbor_recall:.4f}")
print(f"Embedding intrinsic dimension: {embedding_id:.4f}")

# PCA plot
plt.figure(figsize=(8, 6))
plt.scatter(
    embedding_pca[:, 0],
    embedding_pca[:, 1],
    s=15,
)

plt.title("PCA of shared entity embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# Linear probe reconstruction
plt.figure(figsize=(8, 6))
plt.scatter(
    linear_prediction[:, 0],
    linear_prediction[:, 1],
    s=15,
)

plt.title(
    f"Coordinates reconstructed by linear probe "
    f"(R²={coordinate_r2:.3f})"
)
plt.xlabel("Predicted x")
plt.ylabel("Predicted y")
plt.axis("equal")
plt.tight_layout()
plt.show()

# True coordinates
plt.figure(figsize=(8, 6))
plt.scatter(
    true_coordinates[:, 0],
    true_coordinates[:, 1],
    s=15,
)

plt.title("True grid coordinates")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Pair representation analysis
# --------------------------------------------------

pairs = sample_unique_pairs(
    num_points,
    PAIR_SAMPLE_SIZE,
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

task_layers = {}

for task in tasks:
    task_layers[task] = get_head_activations(
        model,
        point_i,
        point_j,
        task=task,
    )

for task_name, layers in task_layers.items():
    layer_names = ["pair", "h1", "h2"]
    cka_matrix = np.zeros(
        (len(layer_names), len(layer_names))
    )

    for row, name_a in enumerate(layer_names):
        for column, name_b in enumerate(layer_names):
            cka_matrix[row, column] = linear_cka(
                layers[name_a],
                layers[name_b],
            )

    print(f"\n{task_name.capitalize()} head CKA")
    print(cka_matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(
        cka_matrix,
        cmap="viridis",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    plt.xticks(
        range(len(layer_names)),
        layer_names,
    )
    plt.yticks(
        range(len(layer_names)),
        layer_names,
    )
    plt.colorbar(label="Linear CKA")
    plt.title(f"{task_name.capitalize()} head layer CKA")
    plt.tight_layout()
    plt.show()

    intrinsic_dimensions = [
        estimate_intrinsic_dimension(
            layers[name].detach().cpu().numpy()
        )
        for name in layer_names
    ]

    print(
        f"{task_name.capitalize()} pair-layer "
        f"intrinsic dimensions:"
    )

    for name, dimension in zip(
        layer_names,
        intrinsic_dimensions,
    ):
        print(f"  {name}: {dimension:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(
        layer_names,
        intrinsic_dimensions,
        marker="o",
    )
    plt.title(
        f"{task_name.capitalize()} head "
        f"intrinsic dimension"
    )
    plt.xlabel("Layer")
    plt.ylabel("Estimated intrinsic dimension")
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Cross-task representation similarity
# --------------------------------------------------

if "distance" in task_layers and "nearest" in task_layers:
    print("\nCross-task CKA")

    for layer_name in ["h1", "h2"]:
        similarity = linear_cka(
            task_layers["distance"][layer_name],
            task_layers["nearest"][layer_name],
        )

        print(f"{layer_name}: {similarity:.4f}")


def cross_validated_coordinate_probe(
    embeddings,
    coordinates,
    n_splits=5,
    seed=0,
):
    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    scores = []

    for train_indices, test_indices in splitter.split(embeddings):
        probe = Ridge(alpha=1.0)

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

    return float(np.mean(scores)), float(np.std(scores))

cv_r2_mean, cv_r2_std = cross_validated_coordinate_probe(
    embeddings,
    true_coordinates,
)

print(
    f"Cross-validated coordinate R²: "
    f"{cv_r2_mean:.4f} ± {cv_r2_std:.4f}"
)