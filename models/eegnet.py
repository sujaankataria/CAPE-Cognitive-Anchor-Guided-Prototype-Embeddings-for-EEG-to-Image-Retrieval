"""EEGNet-style encoder for THINGS-EEG2 retrieval embeddings.

This module follows the architecture principles from Lawhern et al.: temporal
filtering, depthwise spatial filtering across EEG channels, separable temporal
filtering, pooling, dropout, and a compact latent projection. It intentionally
omits any classification head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EEGNetConfig:
    channels: int = 17
    time_points: int = 100
    temporal_filters: int = 8
    depth_multiplier: int = 2
    separable_filters: int = 16
    temporal_kernel: int = 32
    separable_kernel: int = 16
    pool_kernel_1: int = 4
    pool_kernel_2: int = 5
    dropout: float = 0.25
    latent_dim: int = 256


class Conv2dWithMaxNorm(nn.Conv2d):
    """Conv2d with optional output-channel max-norm constraint."""

    def __init__(self, *args, max_norm: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                weight = self.weight.view(self.out_channels, -1)
                norms = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
                desired = norms.clamp(max=self.max_norm)
                self.weight.mul_((desired / norms).view(self.out_channels, 1, 1, 1))
        return super().forward(input)


class SeparableConv2d(nn.Module):
    """Depthwise temporal convolution followed by pointwise channel mixing."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class EEGNet(nn.Module):
    """EEGNet encoder returning a 256-dimensional latent vector.

    Expected input shape is `(batch, channels, time)`, matching flattened
    THINGS-EEG2 trial batches `(batch, 17, 100)`.
    """

    def __init__(
        self,
        channels: int = 17,
        time_points: int = 100,
        latent_dim: int = 256,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 16,
        temporal_kernel: int = 32,
        separable_kernel: int = 16,
        pool_kernel_1: int = 4,
        pool_kernel_2: int = 5,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if channels <= 0 or time_points <= 0:
            raise ValueError("channels and time_points must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.config = EEGNetConfig(
            channels=channels,
            time_points=time_points,
            temporal_filters=temporal_filters,
            depth_multiplier=depth_multiplier,
            separable_filters=separable_filters,
            temporal_kernel=temporal_kernel,
            separable_kernel=separable_kernel,
            pool_kernel_1=pool_kernel_1,
            pool_kernel_2=pool_kernel_2,
            dropout=dropout,
            latent_dim=latent_dim,
        )

        spatial_filters = temporal_filters * depth_multiplier
        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            Conv2dWithMaxNorm(
                temporal_filters,
                spatial_filters,
                kernel_size=(channels, 1),
                groups=temporal_filters,
                bias=False,
                max_norm=1.0,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_kernel_1)),
            nn.Dropout(dropout),
            SeparableConv2d(spatial_filters, separable_filters, separable_kernel),
            nn.BatchNorm2d(separable_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_kernel_2)),
            nn.Dropout(dropout),
        )

        feature_dim = self._infer_feature_dim(channels, time_points)
        self.latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
        )

    def _infer_feature_dim(self, channels: int, time_points: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, time_points)
            features = self.features(dummy)
        return int(features.numel())

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"EEGNet expects input shape (batch, channels, time), got {tuple(x.shape)}")
        expected = (self.config.channels, self.config.time_points)
        actual = tuple(x.shape[1:])
        if actual != expected:
            raise ValueError(f"EEGNet expected trial shape {expected}, got {actual}")
        return self.features(x.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.latent(features)


def build_eegnet_encoder(**kwargs) -> EEGNet:
    """Factory used by training code to construct the default encoder."""

    return EEGNet(**kwargs)
