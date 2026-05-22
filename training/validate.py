"""Validation helpers for EEG-to-image retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import EEGCLIPModel
from training.datasets import EEGCLIPPairDataset
from utils.metrics import retrieval_metrics
from utils.retrieval import CLIPImageRetrievalDatabase, rank_images


SPLITS_ROOT = PROJECT_ROOT / "data" / "things_eeg2" / "splits"


def split_csv_path(split: str) -> Path:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return SPLITS_ROOT / f"{split}_split.csv"


@torch.inference_mode()
def collect_eeg_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    embeddings = []
    image_indices = []
    concepts = []
    losses = []

    for batch in loader:
        eeg = batch["eeg"].to(device)
        image_embedding = batch["image_embedding"].to(device)
        eeg_embedding = model(eeg)
        loss = 1.0 - torch.sum(eeg_embedding * image_embedding, dim=1).mean()
        losses.append(float(loss.item()))
        embeddings.append(eeg_embedding.detach().cpu().numpy())
        image_indices.append(batch["image_index"].detach().cpu().numpy())
        concepts.append(np.asarray([f"{cid}_{name}" for cid, name in zip(batch["concept_id"], batch["concept_name"])]))

    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(image_indices, axis=0),
        np.concatenate(concepts, axis=0),
        float(np.mean(losses)) if losses else 0.0,
    )


def evaluate_retrieval(
    model: torch.nn.Module,
    dataset: EEGCLIPPairDataset,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> dict[str, float | int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eeg_embeddings, target_indices, target_concepts, cosine_loss = collect_eeg_embeddings(model, loader, device)

    database = CLIPImageRetrievalDatabase.from_npz(dataset.image_cache["path"])
    ranked_indices, _ = rank_images(eeg_embeddings, database.embeddings, top_k=database.embeddings.shape[0])
    metrics = retrieval_metrics(
        ranked_indices,
        target_indices,
        target_concepts=target_concepts,
        database_concepts=database.concept,
    )
    return {
        "num_queries": int(eeg_embeddings.shape[0]),
        "cosine_loss": cosine_loss,
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EEGCLIPPairDataset(
        split_csv_path(args.split),
        split=args.split,
        subject=args.subject,
        limit=args.limit,
        anchors_path=None,
    )
    model = EEGCLIPModel().to(device)
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_retrieval(model, dataset, batch_size=args.batch_size, device=device)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
