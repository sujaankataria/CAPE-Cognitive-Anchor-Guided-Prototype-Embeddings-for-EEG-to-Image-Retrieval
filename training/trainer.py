"""Reusable training loop utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from models.losses import anchor_guided_contrastive_loss, direct_eeg_image_contrastive_loss
from training.datasets import EEGCLIPPairDataset
from training.validate import evaluate_retrieval


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainerConfig:
    experiment_id: str
    objective: str = "direct"
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    temperature: float = 0.07
    max_epochs: int = 100
    batch_size: int = 128
    patience: int = 15
    num_workers: int = 0
    checkpoint_dir: Path = PROJECT_ROOT / "results" / "checkpoints"
    log_dir: Path = PROJECT_ROOT / "results" / "logs"
    overwrite_existing: bool = False


class Trainer:
    """Train EEGCLIPModel variants and monitor retrieval validation."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: EEGCLIPPairDataset,
        val_dataset: EEGCLIPPairDataset,
        config: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
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
        self._check_existing_outputs()

    @property
    def checkpoint_root(self) -> Path:
        return self.config.checkpoint_dir / self.config.experiment_id

    @property
    def log_path(self) -> Path:
        return self.config.log_dir / f"{self.config.experiment_id}.jsonl"

    def _check_existing_outputs(self) -> None:
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

    def _compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        eeg = batch["eeg"].to(self.device)
        eeg_embeddings = self.model(eeg)

        if self.config.objective == "direct":
            image_embeddings = batch["image_embedding"].to(self.device)
            return direct_eeg_image_contrastive_loss(
                eeg_embeddings,
                image_embeddings,
                target_ids=batch["image_index"].to(self.device),
                temperature=self.config.temperature,
                symmetric=True,
            )

        if self.config.objective == "anchor":
            anchor_embeddings = self.train_dataset.anchor_embeddings
            if anchor_embeddings is None:
                raise ValueError("Anchor objective requires anchor embeddings")
            return anchor_guided_contrastive_loss(
                eeg_embeddings,
                anchor_embeddings.to(self.device),
                batch["anchor_index"].to(self.device),
                temperature=self.config.temperature,
            )

        raise ValueError(f"Unsupported objective: {self.config.objective}")

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
            loss = self._compute_loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().item()))
        return float(sum(losses) / max(len(losses), 1))

    def validate(self) -> dict[str, float | int]:
        return evaluate_retrieval(
            self.model,
            self.val_dataset,
            batch_size=self.config.batch_size,
            device=self.device,
            num_workers=self.config.num_workers,
        )

    def save_checkpoint(self, epoch: int, val_metrics: dict[str, float | int]) -> Path:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_root / "best.pt"
        torch.save(
            {
                "experiment_id": self.config.experiment_id,
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": {
                    **self.config.__dict__,
                    "checkpoint_dir": str(self.config.checkpoint_dir),
                    "log_dir": str(self.config.log_dir),
                },
                "val_metrics": val_metrics,
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
            "objective": self.config.objective,
            "epochs_completed": len(history),
            "best_epoch": self.best_epoch,
            "best_mrr": self.best_metric,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
            "history": history,
        }
