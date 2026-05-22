"""Model wrappers for EEG-to-CLIP retrieval experiments."""

from __future__ import annotations

import torch
from torch import nn

from models.eegnet import EEGNet
from models.projection_head import ProjectionHead


class EEGCLIPModel(nn.Module):
    """Encode EEG trials and project them into normalized CLIP space."""

    def __init__(
        self,
        encoder: EEGNet | None = None,
        projection_head: ProjectionHead | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else EEGNet()
        self.projection_head = projection_head if projection_head is not None else ProjectionHead()

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.encoder(eeg)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        latent = self.encode_eeg(eeg)
        return self.projection_head(latent)


class CAPEModel(EEGCLIPModel):
    """CAPE-ready wrapper; anchor/prototype losses live outside the module."""
