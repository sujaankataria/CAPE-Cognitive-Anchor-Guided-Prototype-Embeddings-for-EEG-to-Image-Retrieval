"""Build category anchors from cached CLIP image embeddings."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_EMBEDDINGS = PROJECT_ROOT / "embeddings" / "clip" / "train_image_embeddings.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "embeddings" / "anchors" / "uniform_category_anchors.npz"
EXPECTED_EMBEDDING_DIM = 512


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, eps)


def load_embedding_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CLIP embedding cache: {path}")
    with np.load(path, allow_pickle=True) as cache:
        embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            raise ValueError(f"Expected image embeddings shape (n, 512), got {embeddings.shape}")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError(f"Image embeddings are not L2 normalized; norm range {norms.min()} to {norms.max()}")

        metadata_raw = cache["metadata"].item() if cache["metadata"].shape == () else str(cache["metadata"])
        return {
            "embeddings": embeddings,
            "image_index": cache["image_index"].copy(),
            "concept": cache["concept"].copy(),
            "concept_id": cache["concept_id"].copy(),
            "concept_name": cache["concept_name"].copy(),
            "image_file": cache["image_file"].copy(),
            "image_path": cache["image_path"].copy(),
            "metadata": json.loads(metadata_raw),
        }


def build_uniform_anchors(cache: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row_index, concept in enumerate(cache["concept"]):
        groups[str(concept)].append(row_index)

    concepts = sorted(groups)
    anchor_embeddings = np.zeros((len(concepts), EXPECTED_EMBEDDING_DIM), dtype=np.float32)
    concept_id = []
    concept_name = []
    exemplar_counts = []
    exemplar_image_indices = []

    for anchor_index, concept in enumerate(concepts):
        row_indices = groups[concept]
        exemplar_vectors = cache["embeddings"][row_indices]
        anchor_embeddings[anchor_index] = l2_normalize(exemplar_vectors.mean(axis=0, keepdims=True))[0]
        concept_id.append(str(cache["concept_id"][row_indices[0]]))
        concept_name.append(str(cache["concept_name"][row_indices[0]]))
        exemplar_counts.append(len(row_indices))
        exemplar_image_indices.append(cache["image_index"][row_indices].astype(np.int64))

    norms = np.linalg.norm(anchor_embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(f"Anchor embeddings are not L2 normalized; norm range {norms.min()} to {norms.max()}")

    return {
        "anchor_embeddings": anchor_embeddings,
        "concept": np.asarray(concepts, dtype=object),
        "concept_id": np.asarray(concept_id, dtype=object),
        "concept_name": np.asarray(concept_name, dtype=object),
        "exemplar_count": np.asarray(exemplar_counts, dtype=np.int64),
        "exemplar_image_indices": np.asarray(exemplar_image_indices, dtype=object),
        "metadata": {
            "anchor_type": "uniform",
            "count": len(concepts),
            "embedding_dim": EXPECTED_EMBEDDING_DIM,
            "l2_norm_min": float(norms.min()),
            "l2_norm_max": float(norms.max()),
            "source": cache["metadata"],
        },
    }


def save_anchors(path: Path, anchors: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing anchors without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        anchor_embeddings=anchors["anchor_embeddings"],
        concept=anchors["concept"],
        concept_id=anchors["concept_id"],
        concept_name=anchors["concept_name"],
        exemplar_count=anchors["exemplar_count"],
        exemplar_image_indices=anchors["exemplar_image_indices"],
        metadata=json.dumps(anchors["metadata"], sort_keys=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-embeddings", type=Path, default=DEFAULT_TRAIN_EMBEDDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_embeddings = args.train_embeddings
    output = args.output
    if not train_embeddings.is_absolute():
        train_embeddings = PROJECT_ROOT / train_embeddings
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    cache = load_embedding_cache(train_embeddings)
    anchors = build_uniform_anchors(cache)
    save_anchors(output, anchors, overwrite=args.overwrite)

    summary = dict(anchors["metadata"])
    summary["output"] = str(output.relative_to(PROJECT_ROOT))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
