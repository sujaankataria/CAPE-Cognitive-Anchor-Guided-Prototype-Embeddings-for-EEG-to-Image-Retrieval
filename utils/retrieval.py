"""Nearest-neighbor retrieval helpers for normalized CLIP-space embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, eps)


def cosine_similarity(query_embeddings: np.ndarray, database_embeddings: np.ndarray) -> np.ndarray:
    queries = l2_normalize(query_embeddings)
    database = l2_normalize(database_embeddings)
    if queries.ndim == 1:
        queries = queries[None, :]
    if database.ndim != 2:
        raise ValueError(f"Database embeddings must be 2D, got shape {database.shape}")
    if queries.shape[1] != database.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: queries {queries.shape[1]}, database {database.shape[1]}"
        )
    return queries @ database.T


def rank_images(
    query_embeddings: np.ndarray,
    database_embeddings: np.ndarray,
    *,
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ranked database indices and cosine scores for each query."""

    similarities = cosine_similarity(query_embeddings, database_embeddings)
    order = np.argsort(-similarities, axis=1)
    if top_k is not None:
        order = order[:, :top_k]
    scores = np.take_along_axis(similarities, order, axis=1)
    return order, scores


@dataclass(frozen=True)
class CLIPImageRetrievalDatabase:
    embeddings: np.ndarray
    image_index: np.ndarray
    concept: np.ndarray
    concept_id: np.ndarray
    concept_name: np.ndarray
    image_file: np.ndarray
    image_path: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def from_npz(cls, path: Path | str) -> "CLIPImageRetrievalDatabase":
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing embedding cache: {resolved}")

        with np.load(resolved, allow_pickle=True) as cache:
            embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
            if embeddings.ndim != 2:
                raise ValueError(f"Embeddings must be 2D, got {embeddings.shape}")
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-5):
                raise ValueError(
                    f"Embeddings are not L2 normalized; norm range {norms.min()} to {norms.max()}"
                )

            metadata_raw = cache["metadata"].item() if cache["metadata"].shape == () else str(cache["metadata"])
            metadata = json.loads(metadata_raw)

            return cls(
                embeddings=embeddings,
                image_index=cache["image_index"].copy(),
                concept=cache["concept"].copy(),
                concept_id=cache["concept_id"].copy(),
                concept_name=cache["concept_name"].copy(),
                image_file=cache["image_file"].copy(),
                image_path=cache["image_path"].copy(),
                metadata=metadata,
            )

    def search(self, query_embeddings: np.ndarray, *, top_k: int = 10) -> list[list[dict[str, Any]]]:
        indices, scores = rank_images(query_embeddings, self.embeddings, top_k=top_k)
        results: list[list[dict[str, Any]]] = []
        for query_indices, query_scores in zip(indices, scores, strict=True):
            query_results = []
            for index, score in zip(query_indices, query_scores, strict=True):
                query_results.append(
                    {
                        "ranked_index": int(index),
                        "score": float(score),
                        "image_index": int(self.image_index[index]),
                        "concept": str(self.concept[index]),
                        "concept_id": str(self.concept_id[index]),
                        "concept_name": str(self.concept_name[index]),
                        "image_file": str(self.image_file[index]),
                        "image_path": str(self.image_path[index]),
                    }
                )
            results.append(query_results)
        return results
