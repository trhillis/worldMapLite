"""Deterministic, validated experimental splits.

All arrays use original world point ids.  Training code may remap retained
points to a compact embedding table after these splits have been constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


def _canonical_pairs(pairs) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [n_pairs, 2]")
    if np.any(pairs[:, 0] >= pairs[:, 1]):
        raise ValueError("pairs must be unique unordered pairs with i < j")
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError("pairs contain duplicates")
    return pairs


def validate_pair_split(train_pairs, held_out_pairs, excluded_points=()):
    """Fail loudly if a pair split contains leakage or invalid indices."""

    train_pairs = _canonical_pairs(train_pairs)
    held_out_pairs = _canonical_pairs(held_out_pairs)
    overlap = set(map(tuple, train_pairs)) & set(map(tuple, held_out_pairs))
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(f"training/held-out pair overlap detected: {sample}")

    excluded = set(int(point) for point in excluded_points)
    if excluded:
        used = set(train_pairs.ravel()) | set(held_out_pairs.ravel())
        leaked = sorted(used & excluded)
        if leaked:
            raise ValueError(f"excluded points appear in pair split: {leaked}")


def _digest(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype=np.int64)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PairSplit:
    train_pairs: np.ndarray
    held_out_pairs: np.ndarray
    seed: int
    num_points: int
    excluded_points: np.ndarray

    def __post_init__(self):
        validate_pair_split(
            self.train_pairs,
            self.held_out_pairs,
            self.excluded_points,
        )

    @property
    def digest(self) -> str:
        return _digest(
            self.held_out_pairs,
            self.train_pairs,
            self.excluded_points,
        )

    @property
    def held_out_digest(self) -> str:
        return _digest(self.held_out_pairs, self.excluded_points)

    def metadata(self) -> dict:
        return {
            "pair_split_seed": int(self.seed),
            "num_world_points": int(self.num_points),
            "num_training_pairs": int(len(self.train_pairs)),
            "num_held_out_pairs": int(len(self.held_out_pairs)),
            "excluded_points": self.excluded_points.tolist(),
            "split_digest": self.digest,
            "held_out_pair_digest": self.held_out_digest,
        }


def make_pair_split(
    num_points: int,
    n_train: int,
    n_held_out: int,
    seed: int = 0,
    excluded_points=(),
) -> PairSplit:
    """Reserve one fixed held-out prefix, then one nested training prefix."""

    if not isinstance(n_train, (int, np.integer)) or isinstance(n_train, (bool, np.bool_)) or n_train <= 0:
        raise ValueError("n_train must be a positive integer")
    if not isinstance(n_held_out, (int, np.integer)) or isinstance(n_held_out, (bool, np.bool_)) or n_held_out <= 0:
        raise ValueError("n_held_out must be a positive integer")
    if num_points < 2:
        raise ValueError("At least two points are required to form a pair")

    excluded = np.unique(np.asarray(tuple(excluded_points), dtype=np.int64))
    if np.any((excluded < 0) | (excluded >= num_points)):
        raise ValueError("excluded point ids must be within the world")
    retained = np.setdiff1d(np.arange(num_points, dtype=np.int64), excluded)
    point_i, point_j = np.triu_indices(len(retained), k=1)
    all_pairs = np.column_stack((retained[point_i], retained[point_j]))
    requested = n_train + n_held_out
    if requested > len(all_pairs):
        raise ValueError(
            f"Requested {n_train} training pairs and {n_held_out} held-out "
            f"pairs, but only {len(all_pairs)} unique pairs are available among retained points"
        )

    ordering = np.random.default_rng(seed).permutation(len(all_pairs))
    held_out = all_pairs[ordering[:n_held_out]].astype(np.int64, copy=False)
    train = all_pairs[ordering[n_held_out:requested]].astype(np.int64, copy=False)
    return PairSplit(train, held_out, int(seed), int(num_points), excluded)


@dataclass(frozen=True)
class PointSplit:
    retained_points: np.ndarray
    held_out_points: np.ndarray
    seed: int

    @property
    def digest(self) -> str:
        return _digest(self.retained_points, self.held_out_points)

    def metadata(self) -> dict:
        return {
            "held_out_point_seed": int(self.seed),
            "retained_points": self.retained_points.tolist(),
            "held_out_points": self.held_out_points.tolist(),
            "point_split_digest": self.digest,
        }


def make_point_split(
    num_points: int,
    n_held_out: int | None = None,
    held_out_fraction: float | None = None,
    seed: int = 0,
) -> PointSplit:
    """Deterministically select entities excluded from base-model training."""

    if n_held_out is not None and held_out_fraction is not None:
        raise ValueError("set only one of n_held_out or held_out_fraction")
    if held_out_fraction is not None:
        if not 0.0 < held_out_fraction < 1.0:
            raise ValueError("held_out_fraction must be between 0 and 1")
        n_held_out = max(1, int(round(num_points * held_out_fraction)))
    if n_held_out is None:
        n_held_out = 0
    if not 0 <= n_held_out < num_points:
        raise ValueError("n_held_out must be between 0 and num_points - 1")

    ordering = np.random.default_rng(seed).permutation(num_points)
    held_out = np.sort(ordering[:n_held_out]).astype(np.int64, copy=False)
    retained = np.setdiff1d(np.arange(num_points, dtype=np.int64), held_out)
    return PointSplit(retained, held_out, int(seed))


def make_recovery_observation_splits(
    num_points: int,
    held_out_points,
    anchor_count: int,
    seed: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Build disjoint anchor/evaluation observations for each unseen point."""

    held_out = np.unique(np.asarray(held_out_points, dtype=np.int64))
    retained = np.setdiff1d(np.arange(num_points, dtype=np.int64), held_out)
    if not 0 < anchor_count < len(retained):
        raise ValueError("anchor_count must leave at least one unseen retained anchor")

    result = {}
    for point in held_out:
        ordering = np.random.default_rng((int(seed), int(point))).permutation(retained)
        anchors = ordering[:anchor_count]
        evaluation = ordering[anchor_count:]
        result[int(point)] = {
            "anchor_pairs": np.column_stack((np.full(len(anchors), point), anchors)).astype(np.int64),
            "evaluation_pairs": np.column_stack((np.full(len(evaluation), point), evaluation)).astype(np.int64),
        }
    return result
