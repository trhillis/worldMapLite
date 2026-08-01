"""Recover unseen entity embeddings against a frozen trained world model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from src.datasets import distance_scale
from src.evaluation import safe_correlation, true_distances
from src.splits import make_recovery_observation_splits


def recover_held_out_points(
    model,
    full_world,
    retained_point_ids,
    held_out_point_ids,
    anchor_count,
    steps,
    learning_rate,
    seed,
    device,
):
    """Optimize only new vectors, then evaluate on unused observations."""

    observations = make_recovery_observation_splits(
        len(full_world.names), held_out_point_ids, anchor_count, seed,
    )
    retained_point_ids = np.asarray(retained_point_ids, dtype=np.int64)
    original_to_local = {int(original): local for local, original in enumerate(retained_point_ids)}
    base_weights = model.emb.weight.detach().clone()
    base_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    requires_grad = {name: value.requires_grad for name, value in model.named_parameters()}
    was_training = model.training
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    scale = distance_scale(full_world)
    recovered = {}
    rows = []

    try:
        for held_out_id in sorted(observations):
            generator = torch.Generator(device=device).manual_seed(int(seed) * 1_000_003 + held_out_id)
            recovered_embedding = nn.Parameter(torch.empty(
                model.emb.embedding_dim, device=device,
            ).normal_(std=0.02, generator=generator))
            initial_embedding = recovered_embedding.detach().clone()
            optimizer = torch.optim.Adam([recovered_embedding], lr=learning_rate)
            anchor_pairs = observations[held_out_id]["anchor_pairs"]
            anchor_local = torch.tensor(
                [original_to_local[int(point)] for point in anchor_pairs[:, 1]],
                dtype=torch.long, device=device,
            )
            targets = torch.tensor(
                true_distances(full_world, anchor_pairs) / scale,
                dtype=torch.float32, device=device,
            )
            for _ in range(steps):
                repeated = recovered_embedding.unsqueeze(0).expand(len(anchor_local), -1)
                prediction = model.forward_distance_from_embeddings(
                    repeated, model.encode(anchor_local),
                )
                loss = F.smooth_l1_loss(prediction, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            recovered[held_out_id] = recovered_embedding.detach().clone()

            eval_pairs = observations[held_out_id]["evaluation_pairs"]
            eval_local = torch.tensor(
                [original_to_local[int(point)] for point in eval_pairs[:, 1]],
                dtype=torch.long, device=device,
            )
            with torch.inference_mode():
                predictions = model.forward_distance_from_embeddings(
                    recovered[held_out_id].unsqueeze(0).expand(len(eval_local), -1),
                    model.encode(eval_local),
                ).cpu().numpy()
            targets_np = true_distances(full_world, eval_pairs) / scale
            errors = predictions - targets_np

            target_coordinates = (
                np.asarray(full_world.ambient_coordinates)
                if full_world.ambient_coordinates is not None
                else np.asarray(full_world.coordinates)
            )
            coordinate_status = "supported" if target_coordinates.ndim == 2 else "unsupported: no coordinate target"
            coordinate_error = float("nan")
            if coordinate_status == "supported":
                probe = Ridge(alpha=1.0).fit(
                    base_weights.cpu().numpy(), target_coordinates[retained_point_ids],
                )
                recovered_coordinate = probe.predict(recovered[held_out_id].cpu().numpy()[None])[0]
                coordinate_error = float(np.linalg.norm(
                    recovered_coordinate - target_coordinates[held_out_id],
                ))

            latent_nearest = int(torch.argmin(torch.linalg.vector_norm(
                base_weights - recovered[held_out_id], dim=1,
            )))
            true_to_retained = true_distances(full_world, np.column_stack((
                np.full(len(retained_point_ids), held_out_id), retained_point_ids,
            )))
            nearest_recovered = float(latent_nearest in np.flatnonzero(np.isclose(
                true_to_retained, true_to_retained.min(), atol=1e-8, rtol=0.0,
            )))
            rows.append({
                "held_out_point_id": held_out_id,
                "recovery_anchor_count": int(anchor_count),
                "recovery_seed": int(seed),
                "recovery_steps": int(steps),
                "recovery_learning_rate": float(learning_rate),
                "n_evaluation_observations": len(eval_pairs),
                "evaluation_normalized_mae": float(np.abs(errors).mean()),
                "evaluation_normalized_rmse": float(np.sqrt(np.square(errors).mean())),
                "evaluation_pearson": safe_correlation("pearson", targets_np, predictions),
                "evaluation_spearman": safe_correlation("spearman", targets_np, predictions),
                "coordinate_error": coordinate_error,
                "coordinate_metric_status": coordinate_status,
                "nearest_neighbour_recovery": nearest_recovered,
                "nearest_neighbour_status": "supported",
                "recovered_embedding_delta": float(torch.linalg.vector_norm(
                    recovered[held_out_id] - initial_embedding,
                ).cpu()),
            })
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad[name])
        model.train(was_training)

    for name, parameter in model.named_parameters():
        if not torch.equal(parameter.detach().cpu(), base_parameters[name].cpu()):
            raise RuntimeError(f"base-model parameter changed during recovery: {name}")

    pairwise = {"status": "unsupported: fewer than two recovered points"}
    recovered_ids = sorted(recovered)
    if len(recovered_ids) >= 2:
        pairs = np.array([
            (left, right) for position, left in enumerate(recovered_ids)
            for right in recovered_ids[position + 1:]
        ], dtype=np.int64)
        pairwise_was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                predictions = model.forward_distance_from_embeddings(
                    torch.stack([recovered[int(left)] for left in pairs[:, 0]]),
                    torch.stack([recovered[int(right)] for right in pairs[:, 1]]),
                ).cpu().numpy()
        finally:
            model.train(pairwise_was_training)
        targets = true_distances(full_world, pairs) / scale
        pairwise = {
            "status": "supported",
            "n_pairs": len(pairs),
            "normalized_mae": float(np.abs(predictions - targets).mean()),
            "pearson": safe_correlation("pearson", targets, predictions),
            "spearman": safe_correlation("spearman", targets, predictions),
        }

    return {
        "rows": rows,
        "pairwise_consistency": pairwise,
        "observation_splits": {
            str(point): {key: value.tolist() for key, value in split.items()}
            for point, split in observations.items()
        },
    }
