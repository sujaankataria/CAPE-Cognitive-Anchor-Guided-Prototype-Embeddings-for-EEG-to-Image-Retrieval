"""Retrieval and classification metrics."""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def top_k_accuracy(
    ranked_indices: np.ndarray | torch.Tensor,
    target_indices: np.ndarray | torch.Tensor,
    k: int,
) -> float:
    """Fraction of queries whose target index appears in the first `k` ranks."""

    ranks = _to_numpy(ranked_indices)
    targets = _to_numpy(target_indices).reshape(-1)
    if ranks.ndim != 2:
        raise ValueError(f"ranked_indices must be 2D, got {ranks.shape}")
    if targets.shape[0] != ranks.shape[0]:
        raise ValueError(f"target length {targets.shape[0]} does not match queries {ranks.shape[0]}")
    if k <= 0:
        raise ValueError("k must be positive")

    top = ranks[:, : min(k, ranks.shape[1])]
    return float((top == targets[:, None]).any(axis=1).mean())


def mean_reciprocal_rank(
    ranked_indices: np.ndarray | torch.Tensor,
    target_indices: np.ndarray | torch.Tensor,
) -> float:
    """Mean reciprocal rank for exact target image indices."""

    ranks = _to_numpy(ranked_indices)
    targets = _to_numpy(target_indices).reshape(-1)
    if ranks.ndim != 2:
        raise ValueError(f"ranked_indices must be 2D, got {ranks.shape}")
    if targets.shape[0] != ranks.shape[0]:
        raise ValueError(f"target length {targets.shape[0]} does not match queries {ranks.shape[0]}")

    reciprocal_ranks = []
    for row, target in zip(ranks, targets, strict=True):
        matches = np.flatnonzero(row == target)
        reciprocal_ranks.append(0.0 if matches.size == 0 else 1.0 / float(matches[0] + 1))
    return float(np.mean(reciprocal_ranks))


def category_accuracy(
    ranked_indices: np.ndarray | torch.Tensor,
    target_concepts: np.ndarray | list[str],
    database_concepts: np.ndarray | list[str],
    k: int = 1,
) -> float:
    """Top-k category retrieval accuracy using concept labels."""

    ranks = _to_numpy(ranked_indices)
    targets = np.asarray(target_concepts, dtype=object).reshape(-1)
    concepts = np.asarray(database_concepts, dtype=object)
    if targets.shape[0] != ranks.shape[0]:
        raise ValueError(f"target length {targets.shape[0]} does not match queries {ranks.shape[0]}")
    if k <= 0:
        raise ValueError("k must be positive")

    top = ranks[:, : min(k, ranks.shape[1])]
    hits = []
    for row, target in zip(top, targets, strict=True):
        hits.append(bool(np.any(concepts[row] == target)))
    return float(np.mean(hits))


def retrieval_metrics(
    ranked_indices: np.ndarray | torch.Tensor,
    target_indices: np.ndarray | torch.Tensor,
    *,
    target_concepts: np.ndarray | list[str] | None = None,
    database_concepts: np.ndarray | list[str] | None = None,
) -> dict[str, float]:
    """Compute the core CAPE retrieval metrics."""

    metrics = {
        "top1": top_k_accuracy(ranked_indices, target_indices, 1),
        "top5": top_k_accuracy(ranked_indices, target_indices, 5),
        "top10": top_k_accuracy(ranked_indices, target_indices, 10),
        "mrr": mean_reciprocal_rank(ranked_indices, target_indices),
    }
    if target_concepts is not None and database_concepts is not None:
        metrics["category_top1"] = category_accuracy(ranked_indices, target_concepts, database_concepts, k=1)
        metrics["category_top5"] = category_accuracy(ranked_indices, target_concepts, database_concepts, k=5)
    return metrics
