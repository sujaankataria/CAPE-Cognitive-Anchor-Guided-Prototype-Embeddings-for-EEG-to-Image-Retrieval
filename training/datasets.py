"""Training datasets that pair EEG trials with cached CLIP targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.eeg_trials import EEGTrialDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_IMAGE_EMBEDDINGS = PROJECT_ROOT / "embeddings" / "clip" / "train_image_embeddings.npz"
DEFAULT_TEST_IMAGE_EMBEDDINGS = PROJECT_ROOT / "embeddings" / "clip" / "test_image_embeddings.npz"
DEFAULT_ANCHORS = PROJECT_ROOT / "embeddings" / "anchors" / "uniform_category_anchors.npz"
EXPECTED_CLIP_DIM = 512


def _resolve_path(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _load_npz_metadata(cache: Any) -> dict[str, Any]:
    metadata_raw = cache["metadata"].item() if cache["metadata"].shape == () else str(cache["metadata"])
    return json.loads(metadata_raw)


def load_clip_embedding_cache(path: Path | str) -> dict[str, Any]:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing CLIP image embedding cache: {resolved}")

    with np.load(resolved, allow_pickle=True) as cache:
        embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_CLIP_DIM:
            raise ValueError(f"Expected CLIP embeddings shape (n, 512), got {embeddings.shape}")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError(f"CLIP embeddings are not L2 normalized; norm range {norms.min()} to {norms.max()}")

        image_index = np.asarray(cache["image_index"], dtype=np.int64)
        if image_index.shape[0] != embeddings.shape[0]:
            raise ValueError("image_index length does not match embedding rows")
        if not np.array_equal(image_index, np.arange(image_index.shape[0])):
            raise ValueError("Expected image_index rows to preserve metadata order 0..n-1")

        return {
            "embeddings": embeddings,
            "image_index": image_index,
            "concept": cache["concept"].copy(),
            "image_file": cache["image_file"].copy(),
            "image_path": cache["image_path"].copy(),
            "metadata": _load_npz_metadata(cache),
            "path": resolved,
        }


def load_anchor_cache(path: Path | str) -> dict[str, Any]:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing anchor cache: {resolved}")

    with np.load(resolved, allow_pickle=True) as cache:
        embeddings = np.asarray(cache["anchor_embeddings"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_CLIP_DIM:
            raise ValueError(f"Expected anchor embeddings shape (n, 512), got {embeddings.shape}")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError(f"Anchor embeddings are not L2 normalized; norm range {norms.min()} to {norms.max()}")

        concepts = np.asarray(cache["concept"], dtype=object)
        concept_to_anchor_index = {str(concept): index for index, concept in enumerate(concepts)}
        if len(concept_to_anchor_index) != len(concepts):
            raise ValueError("Anchor concepts are not unique")
        exemplar_image_indices = np.asarray(cache["exemplar_image_indices"], dtype=object)
        if exemplar_image_indices.shape[0] != embeddings.shape[0]:
            raise ValueError("Anchor exemplar metadata does not match anchor rows")

        return {
            "embeddings": embeddings,
            "concept": concepts,
            "concept_to_anchor_index": concept_to_anchor_index,
            "exemplar_image_indices": exemplar_image_indices,
            "metadata": _load_npz_metadata(cache),
            "path": resolved,
        }


def validate_anchor_image_scope(
    anchor_cache: dict[str, Any],
    *,
    allowed_image_indices: set[int],
    forbidden_image_indices: set[int],
) -> None:
    exemplar_groups = anchor_cache.get("exemplar_image_indices")
    if exemplar_groups is None:
        raise ValueError("Anchor cache does not record exemplar image indices")
    used = {
        int(image_index)
        for group in exemplar_groups
        for image_index in np.asarray(group, dtype=np.int64).reshape(-1)
    }
    unexpected = used - allowed_image_indices
    missing = allowed_image_indices - used
    forbidden = used & forbidden_image_indices
    if unexpected or missing or forbidden:
        raise ValueError(
            "Anchor cache image scope is invalid: "
            f"unexpected={len(unexpected)}, missing={len(missing)}, forbidden={len(forbidden)}"
        )


def default_image_embedding_path_for_split(split: str) -> Path:
    if split in {"train", "val"}:
        return DEFAULT_TRAIN_IMAGE_EMBEDDINGS
    if split == "test":
        return DEFAULT_TEST_IMAGE_EMBEDDINGS
    raise ValueError(f"Unsupported split: {split}")


class EEGCLIPPairDataset(Dataset):
    """Pair EEG trial rows with cached CLIP image and optional anchor targets."""

    def __init__(
        self,
        split_csv: Path | str,
        *,
        split: str,
        subject: str | Sequence[str] | None = None,
        limit: int | None = None,
        limit_per_subject: int | None = None,
        image_embeddings_path: Path | str | None = None,
        anchors_path: Path | str | None = DEFAULT_ANCHORS,
    ) -> None:
        self.split = split
        self.eeg_dataset = EEGTrialDataset(
            split_csv,
            subject=subject,
            limit=limit,
            limit_per_subject=limit_per_subject,
        )
        self.image_cache = load_clip_embedding_cache(
            image_embeddings_path if image_embeddings_path is not None else default_image_embedding_path_for_split(split)
        )
        self.anchor_cache = load_anchor_cache(anchors_path) if anchors_path is not None else None

    def __len__(self) -> int:
        return len(self.eeg_dataset)

    @property
    def anchor_embeddings(self) -> torch.Tensor | None:
        if self.anchor_cache is None:
            return None
        return torch.as_tensor(self.anchor_cache["embeddings"], dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.eeg_dataset[index]
        image_index = item["image_index"]
        image_embedding = self.image_cache["embeddings"][image_index]

        if str(self.image_cache["image_file"][image_index]) != item["image_file"]:
            raise ValueError(
                "Image embedding cache does not align with split row: "
                f"{self.image_cache['image_file'][image_index]} vs {item['image_file']}"
            )

        output = dict(item)
        output["image_embedding"] = torch.as_tensor(image_embedding, dtype=torch.float32)

        if self.anchor_cache is not None:
            concept = f"{item['concept_id']}_{item['concept_name']}"
            try:
                anchor_index = self.anchor_cache["concept_to_anchor_index"][concept]
            except KeyError as exc:
                raise KeyError(f"Concept {concept!r} not found in anchor cache") from exc
            output["anchor_index"] = int(anchor_index)

        return output
