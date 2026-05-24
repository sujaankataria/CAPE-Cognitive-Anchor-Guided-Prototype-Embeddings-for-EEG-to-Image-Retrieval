"""EEGNet classifier baseline for retrieval experiments."""

from __future__ import annotations

import torch
from torch import nn

from models.eegnet import EEGNet


class EEGClassifier(nn.Module):
    """Classify EEG trials into training categories using an EEGNet encoder."""

    def __init__(
        self,
        *,
        num_classes: int,
        encoder: EEGNet | None = None,
        latent_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.encoder = encoder if encoder is not None else EEGNet(latent_dim=latent_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.encoder(eeg)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_eeg(eeg))
