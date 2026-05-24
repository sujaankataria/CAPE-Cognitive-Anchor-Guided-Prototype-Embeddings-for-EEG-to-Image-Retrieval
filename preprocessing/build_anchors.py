"""Build category anchors from cached CLIP image embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_EMBEDDINGS = PROJECT_ROOT / "embeddings" / "clip" / "train_image_embeddings.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "embeddings" / "anchors" / "uniform_category_anchors.npz"
DEFAULT_TYPICALITY_OUTPUT = PROJECT_ROOT / "embeddings" / "anchors" / "typicality_category_anchors.npz"
EXPECTED_EMBEDDING_DIM = 512
ANCHOR_TYPES = {"uniform", "typicality"}


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


def group_embedding_rows_by_concept(cache: dict[str, Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row_index, concept in enumerate(cache["concept"]):
        groups[str(concept)].append(row_index)
    return groups


def load_typicality_weights(path: Path, valid_image_indices: np.ndarray) -> dict[int, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing typicality weights CSV: {path}")

    valid_indices = {int(index) for index in valid_image_indices}
    weights: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Typicality weights CSV is empty: {path}")
        required = {"image_index", "typicality_weight"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Typicality weights CSV missing required columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                image_index = int(row["image_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid image_index at line {line_number}: {row['image_index']!r}") from exc
            if image_index not in valid_indices:
                raise ValueError(f"Typicality weight image_index {image_index} at line {line_number} is not in cache")
            if image_index in weights:
                raise ValueError(f"Duplicate typicality weight for image_index {image_index}")

            try:
                weight = float(row["typicality_weight"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid typicality_weight at line {line_number}: {row['typicality_weight']!r}"
                ) from exc
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"typicality_weight must be finite and non-negative at line {line_number}: {weight}"
                )
            weights[image_index] = weight

    return weights


def build_category_anchors(
    cache: dict[str, Any],
    *,
    anchor_type: str = "uniform",
    typicality_weights: dict[int, float] | None = None,
    weight_source: Path | None = None,
) -> dict[str, Any]:
    if anchor_type not in ANCHOR_TYPES:
        raise ValueError(f"Unsupported anchor_type {anchor_type!r}; expected one of {sorted(ANCHOR_TYPES)}")
    if anchor_type == "typicality" and typicality_weights is None:
        raise ValueError("typicality anchor construction requires typicality_weights")
    if anchor_type == "uniform" and typicality_weights is not None:
        raise ValueError("uniform anchor construction does not use typicality_weights")

    groups = group_embedding_rows_by_concept(cache)
    concepts = sorted(groups)
    anchor_embeddings = np.zeros((len(concepts), EXPECTED_EMBEDDING_DIM), dtype=np.float32)
    concept_id = []
    concept_name = []
    exemplar_counts = []
    exemplar_image_indices = []
    all_used_weights = []

    for anchor_index, concept in enumerate(concepts):
        row_indices = groups[concept]
        exemplar_vectors = cache["embeddings"][row_indices]
        image_indices = cache["image_index"][row_indices].astype(np.int64)

        if anchor_type == "uniform":
            anchor_vector = exemplar_vectors.mean(axis=0, keepdims=True)
        else:
            weights = []
            missing_indices = []
            for image_index in image_indices:
                key = int(image_index)
                if typicality_weights is None or key not in typicality_weights:
                    missing_indices.append(key)
                else:
                    weights.append(typicality_weights[key])
            if missing_indices:
                raise ValueError(
                    f"Missing typicality weights for concept {concept!r}; "
                    f"first missing image_index values: {missing_indices[:5]}"
                )
            weight_array = np.asarray(weights, dtype=np.float32)
            weight_sum = float(weight_array.sum())
            if weight_sum <= 0.0:
                raise ValueError(f"Typicality weights sum to zero for concept {concept!r}")
            anchor_vector = (exemplar_vectors * weight_array[:, None]).sum(axis=0, keepdims=True) / weight_sum
            all_used_weights.extend(float(weight) for weight in weight_array)

        anchor_embeddings[anchor_index] = l2_normalize(anchor_vector)[0]
        concept_id.append(str(cache["concept_id"][row_indices[0]]))
        concept_name.append(str(cache["concept_name"][row_indices[0]]))
        exemplar_counts.append(len(row_indices))
        exemplar_image_indices.append(image_indices)

    norms = np.linalg.norm(anchor_embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(f"Anchor embeddings are not L2 normalized; norm range {norms.min()} to {norms.max()}")

    metadata = {
        "anchor_type": anchor_type,
        "count": len(concepts),
        "embedding_dim": EXPECTED_EMBEDDING_DIM,
        "l2_norm_min": float(norms.min()),
        "l2_norm_max": float(norms.max()),
        "source": cache["metadata"],
    }
    if anchor_type == "typicality":
        used_weights = np.asarray(all_used_weights, dtype=np.float32)
        metadata.update(
            {
                "weight_source": str(weight_source) if weight_source is not None else None,
                "weight_min": float(used_weights.min()),
                "weight_max": float(used_weights.max()),
                "missing_weight_policy": "error",
            }
        )

    return {
        "anchor_embeddings": anchor_embeddings,
        "concept": np.asarray(concepts, dtype=object),
        "concept_id": np.asarray(concept_id, dtype=object),
        "concept_name": np.asarray(concept_name, dtype=object),
        "exemplar_count": np.asarray(exemplar_counts, dtype=np.int64),
        "exemplar_image_indices": np.asarray(exemplar_image_indices, dtype=object),
        "metadata": metadata,
    }


def build_uniform_anchors(cache: dict[str, Any]) -> dict[str, Any]:
    return build_category_anchors(cache, anchor_type="uniform")


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
    parser.add_argument("--anchor-type", choices=sorted(ANCHOR_TYPES), default="uniform")
    parser.add_argument("--typicality-weights", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_embeddings = args.train_embeddings
    output = args.output
    if not train_embeddings.is_absolute():
        train_embeddings = PROJECT_ROOT / train_embeddings
    if output is None:
        output = DEFAULT_TYPICALITY_OUTPUT if args.anchor_type == "typicality" else DEFAULT_OUTPUT
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    typicality_weights_path = args.typicality_weights
    if typicality_weights_path is not None and not typicality_weights_path.is_absolute():
        typicality_weights_path = PROJECT_ROOT / typicality_weights_path

    cache = load_embedding_cache(train_embeddings)
    if args.anchor_type == "typicality":
        if typicality_weights_path is None:
            raise ValueError("--typicality-weights is required when --anchor-type typicality")
        typicality_weights = load_typicality_weights(typicality_weights_path, cache["image_index"])
        anchors = build_category_anchors(
            cache,
            anchor_type="typicality",
            typicality_weights=typicality_weights,
            weight_source=typicality_weights_path.relative_to(PROJECT_ROOT)
            if typicality_weights_path.is_relative_to(PROJECT_ROOT)
            else typicality_weights_path,
        )
    else:
        if typicality_weights_path is not None:
            raise ValueError("--typicality-weights can only be used with --anchor-type typicality")
        anchors = build_uniform_anchors(cache)
    save_anchors(output, anchors, overwrite=args.overwrite)

    summary = dict(anchors["metadata"])
    summary["output"] = str(output.relative_to(PROJECT_ROOT))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
