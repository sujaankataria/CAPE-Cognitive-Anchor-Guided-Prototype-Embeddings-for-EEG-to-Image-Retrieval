"""Train and evaluate the EEG classification baseline.

The classifier is trained as a closed-set baseline over training concepts. For
retrieval, its class probabilities are mapped back into CLIP space with the
uniform category anchors, then ranked against the requested image database.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import EEGClassifier
from training.datasets import DEFAULT_ANCHORS, EEGCLIPPairDataset, load_anchor_cache
from training.validate import split_csv_path
from utils.metrics import retrieval_metrics
from utils.retrieval import CLIPImageRetrievalDatabase, rank_images
from utils.seed import set_seed


def parse_subjects(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if value.lower() == "all":
        return None
    subjects = [part.strip() for part in value.split(",") if part.strip()]
    if not subjects:
        raise ValueError("Subject list must not be empty")
    return subjects


@dataclass(frozen=True)
class ClassificationConfig:
    experiment_id: str
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    max_epochs: int = 100
    batch_size: int = 128
    patience: int = 15
    num_workers: int = 0
    seed: int = 42
    checkpoint_dir: Path = PROJECT_ROOT / "results" / "checkpoints"
    log_dir: Path = PROJECT_ROOT / "results" / "logs"
    overwrite_existing: bool = False


def _classification_targets_available(dataset: EEGCLIPPairDataset) -> bool:
    if dataset.anchor_cache is None:
        return False
    concepts = {str(concept) for concept in dataset.anchor_cache["concept"]}
    table = dataset.eeg_dataset.table
    return all(
        f"{row.concept_id}_{row.concept_name}" in concepts
        for row in table[["concept_id", "concept_name"]].itertuples(index=False)
    )


@torch.inference_mode()
def evaluate_classifier_retrieval(
    model: EEGClassifier,
    dataset: EEGCLIPPairDataset,
    anchor_embeddings: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> dict[str, float | int | None]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    anchors = torch.nn.functional.normalize(anchor_embeddings.to(device), p=2, dim=-1)
    query_embeddings = []
    target_indices = []
    target_concepts = []
    losses = []
    correct1 = 0
    correct5 = 0
    total_classified = 0
    has_class_targets = _classification_targets_available(dataset)

    for batch in loader:
        eeg = batch["eeg"].to(device)
        logits = model(eeg)
        probabilities = torch.softmax(logits, dim=-1)
        queries = torch.nn.functional.normalize(probabilities @ anchors, p=2, dim=-1)
        query_embeddings.append(queries.detach().cpu().numpy())
        target_indices.append(batch["image_index"].detach().cpu().numpy())
        target_concepts.append(
            np.asarray([f"{cid}_{name}" for cid, name in zip(batch["concept_id"], batch["concept_name"])])
        )

        if has_class_targets:
            labels = batch["anchor_index"].to(device)
            losses.append(float(torch.nn.functional.cross_entropy(logits, labels).item()))
            top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
            correct1 += int((top5[:, 0] == labels).sum().item())
            correct5 += int((top5 == labels[:, None]).any(dim=1).sum().item())
            total_classified += int(labels.shape[0])

    database = CLIPImageRetrievalDatabase.from_npz(dataset.image_cache["path"])
    eeg_queries = np.concatenate(query_embeddings, axis=0)
    ranked_indices, _ = rank_images(eeg_queries, database.embeddings, top_k=database.embeddings.shape[0])
    targets = np.concatenate(target_indices, axis=0)
    concepts = np.concatenate(target_concepts, axis=0)
    metrics = retrieval_metrics(
        ranked_indices,
        targets,
        target_concepts=concepts,
        database_concepts=database.concept,
    )

    output: dict[str, float | int | None] = {
        "num_queries": int(eeg_queries.shape[0]),
        "classification_loss": float(np.mean(losses)) if losses else None,
        "classification_top1": float(correct1 / total_classified) if total_classified else None,
        "classification_top5": float(correct5 / total_classified) if total_classified else None,
        **metrics,
    }
    return output


class ClassificationTrainer:
    """Train an EEGNet category classifier and monitor retrieval validation."""

    def __init__(
        self,
        model: EEGClassifier,
        train_dataset: EEGCLIPPairDataset,
        val_dataset: EEGCLIPPairDataset,
        anchor_embeddings: torch.Tensor,
        config: ClassificationConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.anchor_embeddings = anchor_embeddings
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.bad_epochs = 0
        self._check_outputs()

    @property
    def checkpoint_root(self) -> Path:
        return self.config.checkpoint_dir / self.config.experiment_id

    @property
    def log_path(self) -> Path:
        return self.config.log_dir / f"{self.config.experiment_id}.jsonl"

    def _check_outputs(self) -> None:
        existing = []
        if self.checkpoint_root.exists() and any(self.checkpoint_root.iterdir()):
            existing.append(str(self.checkpoint_root))
        if self.log_path.exists():
            existing.append(str(self.log_path))
        if existing and not self.config.overwrite_existing:
            raise FileExistsError(
                "Refusing to overwrite existing experiment outputs. "
                f"Use a new --experiment-id or pass --overwrite-existing. Existing: {existing}"
            )

    def train_epoch(self) -> float:
        self.model.train()
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        losses = []
        for batch in loader:
            logits = self.model(batch["eeg"].to(self.device))
            labels = batch["anchor_index"].to(self.device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().item()))
        return float(sum(losses) / max(len(losses), 1))

    def validate(self) -> dict[str, float | int | None]:
        return evaluate_classifier_retrieval(
            self.model,
            self.val_dataset,
            self.anchor_embeddings,
            batch_size=self.config.batch_size,
            device=self.device,
            num_workers=self.config.num_workers,
        )

    def save_checkpoint(self, epoch: int, val_metrics: dict[str, float | int | None]) -> Path:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_root / "best.pt"
        torch.save(
            {
                "experiment_id": self.config.experiment_id,
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": {
                    **asdict(self.config),
                    "checkpoint_dir": str(self.config.checkpoint_dir),
                    "log_dir": str(self.config.log_dir),
                },
                "val_metrics": val_metrics,
                "num_classes": int(self.anchor_embeddings.shape[0]),
            },
            path,
        )
        return path

    def log_epoch(self, record: dict[str, Any]) -> None:
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def fit(self) -> dict[str, Any]:
        history = []
        best_checkpoint = None
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            monitor = float(val_metrics["mrr"])
            improved = monitor > self.best_metric
            if improved:
                self.best_metric = monitor
                self.best_epoch = epoch
                self.bad_epochs = 0
                best_checkpoint = self.save_checkpoint(epoch, val_metrics)
            else:
                self.bad_epochs += 1

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val": val_metrics,
                "best_epoch": self.best_epoch,
                "best_mrr": self.best_metric,
                "improved": improved,
                "bad_epochs": self.bad_epochs,
            }
            self.log_epoch(record)
            history.append(record)
            if self.bad_epochs >= self.config.patience:
                break

        return {
            "experiment_id": self.config.experiment_id,
            "objective": "classification",
            "epochs_completed": len(history),
            "best_epoch": self.best_epoch,
            "best_mrr": self.best_metric,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
            "history": history,
        }


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> EEGClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = int(checkpoint.get("num_classes", 1654))
    model = EEGClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="EXP-004")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--val-split", choices=["val", "test"], default="val")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--train-subjects", default=None, help="Comma-separated train subjects, or all.")
    parser.add_argument("--val-subjects", default=None, help="Comma-separated validation subjects, or all.")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-train-per-subject", type=int, default=None)
    parser.add_argument("--limit-val-per-subject", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor_cache = load_anchor_cache(args.anchors)
    anchor_embeddings = torch.as_tensor(anchor_cache["embeddings"], dtype=torch.float32)

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
        model = _load_checkpoint_model(checkpoint_path, device)
        val_subjects = parse_subjects(args.val_subjects) if args.val_subjects is not None else [args.subject]
        dataset = EEGCLIPPairDataset(
            split_csv_path(args.val_split),
            split=args.val_split,
            subject=val_subjects,
            limit=args.limit_val,
            limit_per_subject=args.limit_val_per_subject,
            anchors_path=args.anchors if args.val_split != "test" else None,
        )
        metrics = evaluate_classifier_retrieval(
            model,
            dataset,
            anchor_embeddings,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    train_subjects = parse_subjects(args.train_subjects) if args.train_subjects is not None else [args.subject]
    val_subjects = parse_subjects(args.val_subjects) if args.val_subjects is not None else [args.subject]
    train_dataset = EEGCLIPPairDataset(
        split_csv_path(args.split),
        split=args.split,
        subject=train_subjects,
        limit=args.limit_train,
        limit_per_subject=args.limit_train_per_subject,
        anchors_path=args.anchors,
    )
    val_dataset = EEGCLIPPairDataset(
        split_csv_path(args.val_split),
        split=args.val_split,
        subject=val_subjects,
        limit=args.limit_val,
        limit_per_subject=args.limit_val_per_subject,
        anchors_path=args.anchors if args.val_split != "test" else None,
    )
    model = EEGClassifier(num_classes=anchor_embeddings.shape[0])
    trainer = ClassificationTrainer(
        model,
        train_dataset,
        val_dataset,
        anchor_embeddings,
        ClassificationConfig(
            experiment_id=args.experiment_id,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed,
            overwrite_existing=args.overwrite_existing,
        ),
        device,
    )
    result = trainer.fit()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
