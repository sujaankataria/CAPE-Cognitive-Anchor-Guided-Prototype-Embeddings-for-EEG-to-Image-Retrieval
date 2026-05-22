"""Extract and cache frozen CLIP image embeddings for THINGS-EEG2 images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "things_eeg2"
IMAGE_SET_ROOT = DATA_ROOT / "image_set"
METADATA_PATH = DATA_ROOT / "metadata" / "image_metadata.npy"
EMBEDDINGS_ROOT = PROJECT_ROOT / "embeddings" / "clip"
EXPECTED_EMBEDDING_DIM = 512


def load_npy_object(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value


def split_concept(concept: str) -> tuple[str, str]:
    concept_id, _, concept_name = concept.partition("_")
    if not concept_id or not concept_name:
        raise ValueError(f"Unexpected concept format: {concept!r}")
    return concept_id, concept_name


def load_metadata() -> dict[str, list[str]]:
    metadata = load_npy_object(METADATA_PATH)
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a dictionary: {METADATA_PATH}")
    return metadata


def build_image_records(split: str, limit: int | None = None) -> list[dict[str, Any]]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported image split: {split}")

    metadata = load_metadata()
    if split == "train":
        files = metadata["train_img_files"]
        concepts = metadata["train_img_concepts"]
        source_dir = "training_images"
    else:
        files = metadata["test_img_files"]
        concepts = metadata["test_img_concepts"]
        source_dir = "test_images"

    records: list[dict[str, Any]] = []
    for image_index, (image_file, concept) in enumerate(zip(files, concepts, strict=True)):
        concept_id, concept_name = split_concept(concept)
        image_path = IMAGE_SET_ROOT / source_dir / concept / image_file
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file from metadata mapping: {image_path}")
        records.append(
            {
                "image_index": image_index,
                "concept": concept,
                "concept_id": concept_id,
                "concept_name": concept_name,
                "image_file": image_file,
                "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            }
        )
        if limit is not None and len(records) >= limit:
            break

    if not records:
        raise ValueError(f"No image records found for split {split}")
    return records


class ImageRecordDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], preprocess: Any) -> None:
        self.records = records
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image_path = PROJECT_ROOT / record["image_path"]
        with Image.open(image_path) as image:
            tensor = self.preprocess(image.convert("RGB"))
        return tensor, index


def create_clip_model(model_name: str, pretrained: str, device: torch.device) -> tuple[Any, Any]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess


def encode_records(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model, preprocess = create_clip_model(model_name, pretrained, device)
    dataset = ImageRecordDataset(records, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embedding_batches: list[np.ndarray] = []
    seen_indices: list[int] = []
    with torch.inference_mode():
        for images, batch_indices in tqdm(loader, desc="Extracting CLIP embeddings"):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            embedding_batches.append(image_features.cpu().numpy().astype(np.float32))
            seen_indices.extend(int(index) for index in batch_indices.tolist())

    if seen_indices != list(range(len(records))):
        raise AssertionError("Image embedding order changed during extraction")

    embeddings = np.concatenate(embedding_batches, axis=0)
    if embeddings.shape != (len(records), EXPECTED_EMBEDDING_DIM):
        raise ValueError(
            f"Expected embeddings shape {(len(records), EXPECTED_EMBEDDING_DIM)}, got {embeddings.shape}"
        )

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(f"CLIP embeddings are not L2 normalized; norm range {norms.min()} to {norms.max()}")

    return embeddings


def output_path_for(split: str, output_dir: Path, limit: int | None) -> Path:
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    suffix = f"_first{limit}" if limit is not None else ""
    return output_dir / f"{split}_image_embeddings{suffix}.npz"


def save_embeddings(
    path: Path,
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    split: str,
    model_name: str,
    pretrained: str,
    overwrite: bool,
) -> dict[str, Any]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing embedding cache without --overwrite: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    norms = np.linalg.norm(embeddings, axis=1)
    metadata = {
        "split": split,
        "model_name": model_name,
        "pretrained": pretrained,
        "embedding_dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "l2_norm_min": float(norms.min()),
        "l2_norm_max": float(norms.max()),
        "source_metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
    }

    np.savez_compressed(
        path,
        embeddings=embeddings,
        image_index=np.asarray([record["image_index"] for record in records], dtype=np.int64),
        concept=np.asarray([record["concept"] for record in records], dtype=object),
        concept_id=np.asarray([record["concept_id"] for record in records], dtype=object),
        concept_name=np.asarray([record["concept_name"] for record in records], dtype=object),
        image_file=np.asarray([record["image_file"] for record in records], dtype=object),
        image_path=np.asarray([record["image_path"] for record in records], dtype=object),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    return metadata


def extract_split(
    split: str,
    *,
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    limit: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    records = build_image_records(split, limit=limit)
    embeddings = encode_records(
        records,
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
        device=device,
    )
    path = output_path_for(split, output_dir, limit)
    metadata = save_embeddings(
        path,
        records,
        embeddings,
        split=split,
        model_name=model_name,
        pretrained=pretrained,
        overwrite=overwrite,
    )
    metadata["output"] = str(path.relative_to(PROJECT_ROOT))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=EMBEDDINGS_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test image limit per split.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    summaries = []
    for split in args.splits:
        summary = extract_split(
            split,
            model_name=args.model_name,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))

    print("Completed CLIP image embedding extraction")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
