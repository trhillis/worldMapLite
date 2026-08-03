"""Side-effect-free task and representation evaluation helpers."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from skdim.id import TwoNN

from src.tasks import distance
from torch.utils.data import DataLoader, TensorDataset


def safe_correlation(kind, targets, predictions) -> float:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if len(targets) < 2 or np.ptp(targets) == 0 or np.ptp(predictions) == 0:
        return float("nan")
    function = pearsonr if kind == "pearson" else spearmanr
    result = function(targets, predictions)
    return float(result.statistic)


def evaluate_distance_examples(model, examples, scale, device, batch_size):
    """Evaluate exact examples without changing mode, weights, RNG, or ordering."""

    if not examples:
        raise ValueError("Distance evaluation requires at least one example")
    dataset = TensorDataset(
        torch.tensor([example["indices"][0] for example in examples], dtype=torch.long),
        torch.tensor([example["indices"][1] for example in examples], dtype=torch.long),
        torch.tensor([example["answer"] for example in examples], dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions, targets = [], []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for point_i, point_j, target in loader:
                predictions.append(model.forward_distance(
                    point_i.to(device), point_j.to(device),
                ).cpu())
                targets.append(target)
    finally:
        model.train(was_training)

    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    errors = prediction - target
    normalized_mae = float(np.abs(errors).mean())
    normalized_rmse = float(np.sqrt(np.square(errors).mean()))
    residual_sum = float(np.square(errors).sum())
    total_sum = float(np.square(target - target.mean()).sum())
    return {
        "n_examples": len(target),
        "normalized_mae": normalized_mae,
        "normalized_rmse": normalized_rmse,
        "mae": normalized_mae * scale,
        "rmse": normalized_rmse * scale,
        "r2": float(1.0 - residual_sum / total_sum) if total_sum > 0 else float("nan"),
        "pearson": safe_correlation("pearson", target, prediction),
        "spearman": safe_correlation("spearman", target, prediction),
    }


def true_distance_matrix(world):
    coordinates = np.asarray(world.coordinates)
    if world.meta["type"] == "grid":
        return np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    if world.meta["type"] == "manifold" and world.manifold is not None:
        return np.asarray(world.manifold.distance_matrix(coordinates), dtype=np.float64)
    return None


def representation_metrics(model, world, seed=0, max_pairs=10_000, world_distance_matrix=None) -> dict:
    """Compute the representation metrics already used by main's analysis.

    `world_distance_matrix` lets a caller that evaluates the same world at
    several checkpoints (e.g. learning curves) compute the true geodesic
    distance matrix once and reuse it, instead of paying its full O(n^2)
    cost - expensive for manifolds like the octahedron, whose per-pair
    geodesic is an unfolding search rather than a closed-form formula - on
    every evaluation.
    """

    embeddings = model.emb.weight.detach().cpu().numpy().astype(np.float64)
    coordinates = (
        np.asarray(world.ambient_coordinates, dtype=np.float64)
        if world.ambient_coordinates is not None
        else np.asarray(world.coordinates, dtype=np.float64)
    )
    result = {}

    if len(embeddings) >= 10 and coordinates.ndim == 2:
        n_splits = min(5, len(embeddings))
        scores = []
        for train, test in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(embeddings):
            predictor = Ridge(alpha=1.0).fit(embeddings[train], coordinates[train])
            scores.append(r2_score(
                coordinates[test], predictor.predict(embeddings[test]),
                multioutput="variance_weighted",
            ))
        result["cv_probe_r2_mean"] = float(np.mean(scores))
        result["coordinate_probe_status"] = "supported"
    else:
        result["cv_probe_r2_mean"] = float("nan")
        result["coordinate_probe_status"] = "unsupported: fewer than ten coordinate-bearing points"

    matrix = (
        world_distance_matrix if world_distance_matrix is not None
        else true_distance_matrix(world)
    )
    if matrix is None:
        result.update({
            "embedding_distance_spearman": float("nan"),
            "nearest_recall": float("nan"),
            "distance_geometry_status": f"unsupported: {world.meta['type']} world distance matrix",
        })
    else:
        point_i, point_j = np.triu_indices(len(embeddings), k=1)
        if len(point_i) > max_pairs:
            chosen = np.random.default_rng(seed).choice(len(point_i), max_pairs, replace=False)
            point_i, point_j = point_i[chosen], point_j[chosen]
        latent = np.linalg.norm(embeddings[point_i] - embeddings[point_j], axis=1)
        result["embedding_distance_spearman"] = safe_correlation(
            "spearman", matrix[point_i, point_j], latent,
        )
        latent_matrix = np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=-1)
        true_copy = matrix.copy()
        np.fill_diagonal(latent_matrix, np.inf)
        np.fill_diagonal(true_copy, np.inf)
        recalls = []
        for row in range(len(embeddings)):
            true_neighbours = np.flatnonzero(np.isclose(
                true_copy[row], true_copy[row].min(), atol=1e-8, rtol=0.0,
            ))
            recalls.append(int(np.argmin(latent_matrix[row])) in set(true_neighbours))
        result["nearest_recall"] = float(np.mean(recalls))
        result["distance_geometry_status"] = "supported"

    try:
        result["embedding_intrinsic_dimension"] = (
            float(TwoNN().fit(embeddings).dimension_) if len(embeddings) >= 10 else float("nan")
        )
        result["intrinsic_dimension_status"] = (
            "supported" if len(embeddings) >= 10 else "unsupported: fewer than ten points"
        )
    except Exception as error:
        result["embedding_intrinsic_dimension"] = float("nan")
        result["intrinsic_dimension_status"] = f"unsupported: {type(error).__name__}"
    return result


def true_distances(world, pairs) -> np.ndarray:
    return np.asarray([
        distance(world, int(i), int(j)) for i, j in np.asarray(pairs)
    ], dtype=np.float64)
