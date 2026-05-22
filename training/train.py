"""Training smoke tests and future full-training entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import EEGCLIPModel
from models.losses import anchor_guided_contrastive_loss, direct_eeg_image_contrastive_loss
from training.datasets import EEGCLIPPairDataset
from training.trainer import Trainer, TrainerConfig
from utils.seed import set_seed


SPLITS_ROOT = PROJECT_ROOT / "data" / "things_eeg2" / "splits"


def split_csv_path(split: str) -> Path:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return SPLITS_ROOT / f"{split}_split.csv"


def run_smoke_step(
    *,
    split: str,
    subject: str,
    limit: int,
    batch_size: int,
    objective: str,
    learning_rate: float,
    temperature: float,
) -> dict[str, object]:
    dataset = EEGCLIPPairDataset(
        split_csv_path(split),
        split=split,
        subject=subject,
        limit=limit,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    model = EEGCLIPModel()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    eeg = batch["eeg"]
    image_embeddings = batch["image_embedding"]
    eeg_embeddings = model(eeg)

    if objective == "direct":
        loss = direct_eeg_image_contrastive_loss(
            eeg_embeddings,
            image_embeddings,
            target_ids=batch["image_index"],
            temperature=temperature,
            symmetric=True,
        )
    elif objective == "anchor":
        anchor_embeddings = dataset.anchor_embeddings
        if anchor_embeddings is None:
            raise ValueError("Anchor objective requires anchor embeddings")
        loss = anchor_guided_contrastive_loss(
            eeg_embeddings,
            anchor_embeddings,
            batch["anchor_index"],
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "split": split,
        "subject": subject,
        "objective": objective,
        "dataset_rows": len(dataset),
        "batch_size": batch_size,
        "temperature": temperature,
        "learning_rate": learning_rate,
        "eeg_shape": list(eeg.shape),
        "image_embedding_shape": list(image_embeddings.shape),
        "eeg_embedding_shape": list(eeg_embeddings.shape),
        "eeg_embedding_norm_min": float(eeg_embeddings.norm(dim=1).min().item()),
        "eeg_embedding_norm_max": float(eeg_embeddings.norm(dim=1).max().item()),
        "loss": float(loss.detach().item()),
        "first_trial": {
            "global_trial_id": int(batch["global_trial_id"][0]),
            "image_index": int(batch["image_index"][0]),
            "repetition_index": int(batch["repetition_index"][0]),
            "concept_id": str(batch["concept_id"][0]),
            "concept_name": str(batch["concept_name"][0]),
            "image_file": str(batch["image_file"][0]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-step", action="store_true", help="Run one forward/backward optimizer step.")
    parser.add_argument("--experiment-id", default="EXP-001-SANITY")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--val-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--objective", choices=["direct", "anchor"], default="direct")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.smoke_step:
        report = run_smoke_step(
            split=args.split,
            subject=args.subject,
            limit=args.limit,
            batch_size=args.batch_size,
            objective=args.objective,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EEGCLIPPairDataset(
        split_csv_path(args.split),
        split=args.split,
        subject=args.subject,
        limit=args.limit_train,
    )
    val_dataset = EEGCLIPPairDataset(
        split_csv_path(args.val_split),
        split=args.val_split,
        subject=args.subject,
        limit=args.limit_val,
    )
    trainer = Trainer(
        EEGCLIPModel(),
        train_dataset,
        val_dataset,
        TrainerConfig(
            experiment_id=args.experiment_id,
            objective=args.objective,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            num_workers=args.num_workers,
            overwrite_existing=args.overwrite_existing,
        ),
        device,
    )
    report = trainer.fit()
    report.update(
        {
            "subject": args.subject,
            "train_split": args.split,
            "val_split": args.val_split,
            "train_rows": len(train_dataset),
            "val_rows": len(val_dataset),
            "seed": args.seed,
            "device": str(device),
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
