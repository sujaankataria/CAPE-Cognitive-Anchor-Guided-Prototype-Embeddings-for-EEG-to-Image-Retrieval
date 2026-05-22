"""Projection head from EEG latent features into frozen CLIP space."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """Map EEGNet latents to normalized CLIP-space embeddings."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 512,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("Projection dimensions must be positive")

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"ProjectionHead expects shape (batch, features), got {tuple(x.shape)}")
        projected = self.net(x)
        return F.normalize(projected, p=2, dim=-1)
