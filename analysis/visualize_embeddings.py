"""Create PCA or UMAP plots from learned point embeddings in a .pt checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def load_checkpoint(path: Path) -> Any:
    """Load a PyTorch checkpoint onto CPU."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def find_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Find the model state dictionary inside common checkpoint formats."""
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint is not a dictionary. "
            "Pass a checkpoint containing a model state dictionary."
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "model_state",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            if all(torch.is_tensor(item) for item in value.values()):
                return value

    # Some .pt files are state dictionaries directly.
    if checkpoint and all(torch.is_tensor(item) for item in checkpoint.values()):
        return checkpoint

    raise KeyError(
        "Could not find a model state dictionary. "
        "Run with --list-keys to inspect the checkpoint."
    )


def list_checkpoint_contents(checkpoint: Any) -> None:
    """Print checkpoint and state-dictionary keys."""
    if not isinstance(checkpoint, dict):
        print(f"Checkpoint type: {type(checkpoint).__name__}")
        return

    print("Top-level checkpoint keys:")
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            print(f"  {key}: tensor {tuple(value.shape)}")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: ndarray {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} entries")
        else:
            print(f"  {key}: {type(value).__name__}")

    try:
        state_dict = find_state_dict(checkpoint)
    except (KeyError, TypeError):
        return

    print("\nState-dictionary tensors:")
    for key, value in state_dict.items():
        print(f"  {key}: {tuple(value.shape)}")


def choose_embedding_key(
    state_dict: dict[str, torch.Tensor],
    requested_key: str | None,
) -> str:
    """Find the likely learned point-embedding weight."""
    if requested_key is not None:
        if requested_key not in state_dict:
            raise KeyError(
                f"Embedding key {requested_key!r} was not found.\n"
                f"Available keys include:\n" +
                "\n".join(f"  {key}" for key in state_dict)
            )
        return requested_key

    candidates: list[tuple[str, torch.Tensor]] = []

    for key, tensor in state_dict.items():
        if tensor.ndim != 2:
            continue

        lowered = key.lower()

        if (
            "point" in lowered
            and "embed" in lowered
            and lowered.endswith("weight")
        ):
            candidates.append((key, tensor))

    if not candidates:
        for key, tensor in state_dict.items():
            lowered = key.lower()
            if tensor.ndim == 2 and "embed" in lowered and lowered.endswith("weight"):
                candidates.append((key, tensor))

    if not candidates:
        raise KeyError(
            "Could not automatically find a point-embedding matrix. "
            "Run with --list-keys, then pass its key using --embedding-key."
        )

    # Prefer matrices with the largest number of token rows.
    candidates.sort(key=lambda item: item[1].shape[0], reverse=True)

    if len(candidates) > 1:
        print("Possible embedding matrices:")
        for key, tensor in candidates:
            print(f"  {key}: {tuple(tensor.shape)}")
        print(f"Using: {candidates[0][0]}")

    return candidates[0][0]


def find_array(
    checkpoint: Any,
    requested_key: str | None,
    candidates: tuple[str, ...],
) -> np.ndarray | None:
    """Find an optional coordinate or label array in the checkpoint."""
    if not isinstance(checkpoint, dict):
        return None

    keys = (requested_key,) if requested_key else candidates

    for key in keys:
        if key is None or key not in checkpoint:
            continue

        value = checkpoint[key]

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()

        if isinstance(value, np.ndarray):
            return value

        if isinstance(value, list):
            return np.asarray(value)

    return None


def reduce_embeddings(
    embeddings: np.ndarray,
    method: str,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[np.ndarray, str]:
    """Reduce embeddings to two dimensions."""
    if method == "pca":
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)

        explained = reducer.explained_variance_ratio_.sum()
        subtitle = f"PCA explained variance: {explained:.1%}"
        return reduced, subtitle

    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP is not installed. Install it with:\n"
                "pip install umap-learn"
            ) from exc

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=random_state,
        )
        reduced = reducer.fit_transform(embeddings)
        subtitle = (
            f"UMAP: n_neighbors={n_neighbors}, "
            f"min_dist={min_dist}, seed={random_state}"
        )
        return reduced, subtitle

    raise ValueError(f"Unknown method: {method}")


def make_color_values(
    true_coordinates: np.ndarray | None,
    labels: np.ndarray | None,
    color_dimension: int,
    count: int,
) -> tuple[np.ndarray, str]:
    """Choose values used to color points."""
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)

        if len(labels) != count:
            raise ValueError(
                f"Label count {len(labels)} does not match "
                f"embedding count {count}."
            )

        return labels, "True label"

    if true_coordinates is not None:
        true_coordinates = np.asarray(true_coordinates)

        if true_coordinates.ndim == 1:
            true_coordinates = true_coordinates[:, None]

        if true_coordinates.shape[0] != count:
            raise ValueError(
                f"Coordinate count {true_coordinates.shape[0]} does not "
                f"match embedding count {count}."
            )

        if color_dimension >= true_coordinates.shape[1]:
            raise ValueError(
                f"--color-dimension {color_dimension} is invalid for "
                f"coordinates with shape {true_coordinates.shape}."
            )

        return (
            true_coordinates[:, color_dimension],
            f"True coordinate {color_dimension}",
        )

    # A stable fallback that at least preserves token identity.
    return np.arange(count), "Point index"


def plot_projection(
    reduced: np.ndarray,
    colors: np.ndarray,
    color_label: str,
    title: str,
    subtitle: str,
    output: Path,
    edges: list[tuple[int, int]] | None = None,
) -> None:
    """Save a two-dimensional embedding projection."""
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))

    # Draw true graph connections underneath the points.
    if edges:
        for source, target in edges:
            axis.plot(
                [reduced[source, 0], reduced[target, 0]],
                [reduced[source, 1], reduced[target, 1]],
                linewidth=0.25,
                alpha=0.05,
                color="black",
                zorder=1,
            )

    scatter = axis.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=colors,
        s=24,
        alpha=0.9,
        zorder=2,
    )

    axis.set_title(f"{title}\n{subtitle}")
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.grid(alpha=0.2)

    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label(color_label)

    figure.tight_layout()
    figure.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved {output}")

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--method",
        choices=("pca", "umap"),
        default="pca",
    )
    parser.add_argument(
        "--embedding-key",
        help="Exact state-dictionary key for the point-embedding weight.",
    )
    parser.add_argument(
        "--coordinates-key",
        help="Top-level checkpoint key containing true coordinates.",
    )
    parser.add_argument(
        "--labels-key",
        help="Top-level checkpoint key containing categorical labels.",
    )
    parser.add_argument(
        "--color-dimension",
        type=int,
        default=0,
        help="True-coordinate dimension used to color points.",
    )

    parser.add_argument(
    "--graph-type",
    choices=("none", "grid", "torus-knn"),
    default="none",
    )
    parser.add_argument(
        "--graph-k",
        type=int,
        default=4,
        help="Number of true-geodesic neighbours for k-NN overlays.",
    )

    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--title")
    parser.add_argument("--output")
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="Print checkpoint contents and exit.",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)

    if args.list_keys:
        list_checkpoint_contents(checkpoint)
        return

    state_dict = find_state_dict(checkpoint)
    embedding_key = choose_embedding_key(
        state_dict,
        args.embedding_key,
    )

    embeddings = (
        state_dict[embedding_key]
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    print(f"Embedding key: {embedding_key}")
    print(f"Embedding shape: {embeddings.shape}")

    true_coordinates = find_array(
        checkpoint,
        args.coordinates_key,
        (
            "chart_points",
            "coordinates",
            "true_coordinates",
            "world_points",
            "points",
        ),
    )

    labels = find_array(
        checkpoint,
        args.labels_key,
        (
            "labels",
            "face_labels",
            "triangle_labels",
        ),
    )

    reduced, subtitle = reduce_embeddings(
        embeddings=embeddings,
        method=args.method,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    colors, color_label = make_color_values(
        true_coordinates=true_coordinates,
        labels=labels,
        color_dimension=args.color_dimension,
        count=len(embeddings),
    )

    output = (
        Path(args.output)
        if args.output
        else Path("analysis_results/embedding_visualizations")
        / f"{checkpoint_path.stem}_{args.method}.png"
    )

    title = args.title or (
        f"{checkpoint_path.stem}: learned embeddings ({args.method.upper()})"
    )

    edges: list[tuple[int, int]] | None = None

    if args.graph_type != "none":
        if true_coordinates is None:
            raise ValueError(
                "--graph-type requires true coordinates in the checkpoint."
            )

        if args.graph_type == "grid":
            edges = build_grid_edges(true_coordinates)

        elif args.graph_type == "torus-knn":
            edges = build_torus_knn_edges(
                true_coordinates,
                k=args.graph_k,
            )

        print(f"Drawing {len(edges)} true-geometry edges")

    plot_projection(
        reduced=reduced,
        colors=colors,
        color_label=color_label,
        title=title,
        subtitle=subtitle,
        output=output,
        edges=edges,
    )

def build_grid_edges(coordinates: np.ndarray) -> list[tuple[int, int]]:
    """Connect horizontal and vertical neighbours in a regular grid."""
    coordinates = np.asarray(coordinates)

    lookup = {
        (float(x), float(y)): index
        for index, (x, y) in enumerate(coordinates)
    }

    unique_x = np.sort(np.unique(coordinates[:, 0]))
    unique_y = np.sort(np.unique(coordinates[:, 1]))

    x_step = np.min(np.diff(unique_x)) if len(unique_x) > 1 else 1.0
    y_step = np.min(np.diff(unique_y)) if len(unique_y) > 1 else 1.0

    edges: list[tuple[int, int]] = []

    for index, (x, y) in enumerate(coordinates):
        right = lookup.get((float(x + x_step), float(y)))
        above = lookup.get((float(x), float(y + y_step)))

        if right is not None:
            edges.append((index, right))

        if above is not None:
            edges.append((index, above))

    return edges

def build_torus_knn_edges(
    coordinates: np.ndarray,
    k: int = 4,
) -> list[tuple[int, int]]:
    """Build a true-geodesic k-nearest-neighbour graph on a flat torus."""
    coordinates = np.asarray(coordinates, dtype=float)
    tau = 2.0 * np.pi

    difference = coordinates[:, None, :] - coordinates[None, :, :]
    difference -= tau * np.round(difference / tau)

    distances = np.linalg.norm(difference, axis=2)
    np.fill_diagonal(distances, np.inf)

    neighbours = np.argsort(distances, axis=1)[:, :k]

    edges: set[tuple[int, int]] = set()

    for source, targets in enumerate(neighbours):
        for target in targets:
            edge = tuple(sorted((source, int(target))))
            edges.add(edge)

    return sorted(edges)

if __name__ == "__main__":
    main()