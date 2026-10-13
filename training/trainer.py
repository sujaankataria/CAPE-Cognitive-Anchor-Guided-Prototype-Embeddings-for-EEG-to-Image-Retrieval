"""Reusable training loop utilities."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.losses import (
    anchor_guided_contrastive_loss,
    direct_eeg_image_contrastive_loss,
    hybrid_eeg_anchor_contrastive_loss,
)
from training.datasets import EEGCLIPPairDataset
from training.prototypes import DynamicPrototypeMemory
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
    prototype_alpha: float = 0.7
    prototype_beta: float = 0.99
    lambda_anchor: float = 0.2
    lambda_prototype: float = 0.0
    prototype_warmup_epochs: int = 5
    checkpoint_dir: Path = PROJECT_ROOT / "results" / "checkpoints"
    log_dir: Path = PROJECT_ROOT / "results" / "logs"
    overwrite_existing: bool = False
    resume: bool = False
    max_session_minutes: float | None = None


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
        self.start_epoch = 1
        self.epochs_completed = 0
        self.current_epoch = 0
        self.prototype_memory = self._build_prototype_memory()
        self._check_existing_outputs()
        if config.resume:
            self._resume_latest()

    def _build_prototype_memory(self) -> DynamicPrototypeMemory | None:
        if self.config.objective == "hybrid" and self.config.lambda_prototype <= 0:
            return None
        if self.config.objective not in {"prototype", "hybrid"}:
            return None
        anchor_embeddings = self.train_dataset.anchor_embeddings
        if anchor_embeddings is None:
            raise ValueError(f"{self.config.objective} objective requires anchor embeddings")
        return DynamicPrototypeMemory(
            anchor_embeddings,
            alpha=self.config.prototype_alpha,
            beta=self.config.prototype_beta,
            device=self.device,
        )

    @property
    def checkpoint_root(self) -> Path:
        return self.config.checkpoint_dir / self.config.experiment_id

    @property
    def log_path(self) -> Path:
        return self.config.log_dir / f"{self.config.experiment_id}.jsonl"

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.checkpoint_root / "latest.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_root / "best.pt"

    def _check_existing_outputs(self) -> None:
        existing = []
        if self.checkpoint_root.exists() and any(self.checkpoint_root.iterdir()):
            existing.append(str(self.checkpoint_root))
        if self.log_path.exists():
            existing.append(str(self.log_path))
        if existing and self.config.resume:
            if not self.latest_checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume without latest checkpoint: {self.latest_checkpoint_path}"
                )
            return
        if existing and not self.config.overwrite_existing:
            raise FileExistsError(
                "Refusing to overwrite existing experiment outputs. "
                f"Use a new --experiment-id or pass --overwrite-existing. Existing: {existing}"
            )

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_state(state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _resume_latest(self) -> None:
        checkpoint = torch.load(self.latest_checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["experiment_id"] != self.config.experiment_id:
            raise ValueError("Resume checkpoint experiment ID does not match")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(self.device)
        self.best_metric = float(checkpoint["best_metric"])
        self.best_epoch = int(checkpoint["best_epoch"])
        self.bad_epochs = int(checkpoint["bad_epochs"])
        self.epochs_completed = int(checkpoint["epoch"])
        self.start_epoch = self.epochs_completed + 1
        if self.prototype_memory is not None:
            prototype_state = checkpoint.get("prototype_memory")
            if prototype_state is None:
                raise ValueError("Prototype run checkpoint is missing prototype memory")
            self.prototype_memory.load_state_dict(prototype_state)
        self._restore_rng_state(checkpoint["rng_state"])

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

        if self.config.objective == "prototype":
            if self.prototype_memory is None:
                raise ValueError("Prototype memory is not initialized")
            effective_anchors = self.prototype_memory.effective_anchors()
            return anchor_guided_contrastive_loss(
                eeg_embeddings,
                effective_anchors,
                batch["anchor_index"].to(self.device),
                temperature=self.config.temperature,
            )

        if self.config.objective == "hybrid":
            anchor_embeddings = self.train_dataset.anchor_embeddings
            if anchor_embeddings is None:
                raise ValueError("Hybrid objective requires anchor embeddings")
            prototype_anchors = None
            if (
                self.prototype_memory is not None
                and self.config.lambda_prototype > 0
                and self.current_epoch > self.config.prototype_warmup_epochs
            ):
                prototype_anchors = self.prototype_memory.effective_anchors().detach()
            return hybrid_eeg_anchor_contrastive_loss(
                eeg_embeddings,
                batch["image_embedding"].to(self.device),
                anchor_embeddings.to(self.device),
                batch["image_index"].to(self.device),
                batch["anchor_index"].to(self.device),
                temperature=self.config.temperature,
                lambda_anchor=self.config.lambda_anchor,
                prototype_anchor_embeddings=prototype_anchors,
                lambda_prototype=self.config.lambda_prototype if prototype_anchors is not None else 0.0,
                symmetric=True,
            )

        raise ValueError(f"Unsupported objective: {self.config.objective}")

    def _update_prototypes(self, batch: dict[str, Any]) -> None:
        if self.prototype_memory is None:
            return
        if self.config.objective == "hybrid" and self.current_epoch <= self.config.prototype_warmup_epochs:
            return
        self.model.eval()
        with torch.inference_mode():
            eeg_embeddings = self.model(batch["eeg"].to(self.device))
        self.prototype_memory.update(eeg_embeddings, batch["anchor_index"].to(self.device))
        self.model.train()

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
            self._update_prototypes(batch)
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
        path = self.best_checkpoint_path
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
                "prototype_memory": self.prototype_memory.state_dict() if self.prototype_memory is not None else None,
            },
            path,
        )
        return path

    def save_latest(self, epoch: int, *, completed: bool) -> Path:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "experiment_id": self.config.experiment_id,
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "bad_epochs": self.bad_epochs,
                "completed": completed,
                "config": {
                    **self.config.__dict__,
                    "checkpoint_dir": str(self.config.checkpoint_dir),
                    "log_dir": str(self.config.log_dir),
                },
                "prototype_memory": self.prototype_memory.state_dict() if self.prototype_memory is not None else None,
                "rng_state": self._rng_state(),
            },
            self.latest_checkpoint_path,
        )
        return self.latest_checkpoint_path

    def log_epoch(self, record: dict[str, Any]) -> None:
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def fit(self) -> dict[str, Any]:
        history = []
        best_checkpoint = self.best_checkpoint_path if self.best_checkpoint_path.is_file() else None
        session_started = time.monotonic()
        stopped_for_time = False
        for epoch in range(self.start_epoch, self.config.max_epochs + 1):
            self.current_epoch = epoch
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
                "initialized_prototypes": self.prototype_memory.num_initialized if self.prototype_memory is not None else None,
            }
            self.log_epoch(record)
            history.append(record)
            self.epochs_completed = epoch

            if self.bad_epochs >= self.config.patience:
                self.save_latest(epoch, completed=True)
                break
            self.save_latest(epoch, completed=False)
            if self.config.max_session_minutes is not None:
                elapsed_minutes = (time.monotonic() - session_started) / 60.0
                if elapsed_minutes >= self.config.max_session_minutes:
                    stopped_for_time = True
                    break
        else:
            if self.epochs_completed >= self.config.max_epochs:
                self.save_latest(self.epochs_completed, completed=True)

        completed = (
            self.epochs_completed >= self.config.max_epochs
            or self.bad_epochs >= self.config.patience
        )

        return {
            "experiment_id": self.config.experiment_id,
            "objective": self.config.objective,
            "epochs_completed": self.epochs_completed,
            "epochs_this_session": len(history),
            "best_epoch": self.best_epoch,
            "best_mrr": self.best_metric,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
            "latest_checkpoint": str(self.latest_checkpoint_path),
            "completed": completed,
            "stopped_for_time": stopped_for_time,
            "initialized_prototypes": self.prototype_memory.num_initialized if self.prototype_memory is not None else None,
            "history": history,
        }
