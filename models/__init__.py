"""Model components for EEG-to-CLIP retrieval."""

from models.cape_model import CAPEModel, EEGCLIPModel
from models.eegnet import EEGNet, EEGNetConfig, build_eegnet_encoder
from models.projection_head import ProjectionHead

__all__ = [
    "CAPEModel",
    "EEGCLIPModel",
    "EEGNet",
    "EEGNetConfig",
    "ProjectionHead",
    "build_eegnet_encoder",
]
