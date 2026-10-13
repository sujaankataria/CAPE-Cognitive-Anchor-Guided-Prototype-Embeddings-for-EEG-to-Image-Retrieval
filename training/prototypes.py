"""Dynamic EEG prototype memory for CAPE training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


class DynamicPrototypeMemory:
    """Maintain EMA EEG prototypes and interpolate them with semantic anchors."""

    def __init__(
        self,
        anchor_embeddings: torch.Tensor,
        *,
        alpha: float = 0.7,
        beta: float = 0.99,
        device: torch.device | None = None,
    ) -> None:
        if anchor_embeddings.ndim != 2:
            raise ValueError(f"anchor_embeddings must be 2D, got {tuple(anchor_embeddings.shape)}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")

        self.device = device if device is not None else anchor_embeddings.device
        self.anchor_embeddings = F.normalize(anchor_embeddings.to(self.device), p=2, dim=-1)
        self.prototype_embeddings = torch.zeros_like(self.anchor_embeddings, device=self.device)
        self.initialized = torch.zeros(self.anchor_embeddings.shape[0], dtype=torch.bool, device=self.device)
        self.alpha = alpha
        self.beta = beta

    @property
    def num_initialized(self) -> int:
        return int(self.initialized.sum().item())

    def effective_anchors(self) -> torch.Tensor:
        prototype_part = torch.where(
            self.initialized[:, None],
            self.prototype_embeddings,
            self.anchor_embeddings,
        )
        mixed = self.alpha * self.anchor_embeddings + (1.0 - self.alpha) * prototype_part
        return F.normalize(mixed, p=2, dim=-1)

    @torch.no_grad()
    def update(self, eeg_embeddings: torch.Tensor, anchor_indices: torch.Tensor) -> None:
        if eeg_embeddings.ndim != 2:
            raise ValueError(f"eeg_embeddings must be 2D, got {tuple(eeg_embeddings.shape)}")
        if anchor_indices.ndim != 1 or anchor_indices.shape[0] != eeg_embeddings.shape[0]:
            raise ValueError("anchor_indices must be 1D and match EEG embedding batch size")

        embeddings = F.normalize(eeg_embeddings.detach().to(self.device), p=2, dim=-1)
        indices = anchor_indices.detach().to(self.device, dtype=torch.long)
        unique_indices = torch.unique(indices)

        for anchor_index in unique_indices:
            mask = indices == anchor_index
            batch_mean = F.normalize(embeddings[mask].mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
            idx = int(anchor_index.item())
            if self.initialized[idx]:
                updated = self.beta * self.prototype_embeddings[idx] + (1.0 - self.beta) * batch_mean
                self.prototype_embeddings[idx] = F.normalize(updated, p=2, dim=0)
            else:
                self.prototype_embeddings[idx] = batch_mean
                self.initialized[idx] = True

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "anchor_embeddings": self.anchor_embeddings.detach().cpu(),
            "prototype_embeddings": self.prototype_embeddings.detach().cpu(),
            "initialized": self.initialized.detach().cpu(),
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | float]) -> None:
        anchor_embeddings = state["anchor_embeddings"]
        prototype_embeddings = state["prototype_embeddings"]
        initialized = state["initialized"]
        if not isinstance(anchor_embeddings, torch.Tensor):
            raise TypeError("Prototype anchor_embeddings must be a tensor")
        if not isinstance(prototype_embeddings, torch.Tensor):
            raise TypeError("Prototype prototype_embeddings must be a tensor")
        if not isinstance(initialized, torch.Tensor):
            raise TypeError("Prototype initialized flags must be a tensor")
        if anchor_embeddings.shape != self.anchor_embeddings.shape:
            raise ValueError("Prototype anchor shape does not match the current run")
        if prototype_embeddings.shape != self.prototype_embeddings.shape:
            raise ValueError("Prototype embedding shape does not match the current run")
        if initialized.shape != self.initialized.shape:
            raise ValueError("Prototype initialization flags do not match the current run")
        self.anchor_embeddings.copy_(anchor_embeddings.to(self.device))
        self.prototype_embeddings.copy_(prototype_embeddings.to(self.device))
        self.initialized.copy_(initialized.to(self.device, dtype=torch.bool))
        self.alpha = float(state["alpha"])
        self.beta = float(state["beta"])
