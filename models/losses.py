"""Loss functions for EEG-to-CLIP retrieval training."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _normalized_logits(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if query_embeddings.ndim != 2 or target_embeddings.ndim != 2:
        raise ValueError(
            "Contrastive losses expect 2D tensors, got "
            f"{tuple(query_embeddings.shape)} and {tuple(target_embeddings.shape)}"
        )
    if query_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError(
            "Embedding dimensions must match, got "
            f"{query_embeddings.shape[1]} and {target_embeddings.shape[1]}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    queries = F.normalize(query_embeddings, p=2, dim=-1)
    targets = F.normalize(target_embeddings, p=2, dim=-1)
    return queries @ targets.T / temperature


def direct_eeg_image_contrastive_loss(
    eeg_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    target_ids: torch.Tensor | None = None,
    temperature: float = 0.07,
    symmetric: bool = True,
) -> torch.Tensor:
    """InfoNCE loss for direct EEG-image alignment.

    Without `target_ids`, row `i` in the EEG batch is paired with row `i` in the
    image embedding batch. With `target_ids`, rows sharing the same target id are
    treated as multiple positives, which is useful for repeated EEG trials of
    the same image.
    """

    if eeg_embeddings.shape[0] != image_embeddings.shape[0]:
        raise ValueError(
            "Direct contrastive loss requires equal batch sizes, got "
            f"{eeg_embeddings.shape[0]} and {image_embeddings.shape[0]}"
        )

    logits = _normalized_logits(eeg_embeddings, image_embeddings, temperature)
    if target_ids is None:
        labels = torch.arange(logits.shape[0], device=logits.device)
        eeg_to_image = F.cross_entropy(logits, labels)
        if not symmetric:
            return eeg_to_image
        image_to_eeg = F.cross_entropy(logits.T, labels)
        return 0.5 * (eeg_to_image + image_to_eeg)

    if target_ids.ndim != 1 or target_ids.shape[0] != logits.shape[0]:
        raise ValueError(f"target_ids must have shape ({logits.shape[0]},), got {tuple(target_ids.shape)}")
    target_ids = target_ids.to(device=logits.device)
    positive_mask = target_ids[:, None] == target_ids[None, :]
    eeg_to_image = _multi_positive_cross_entropy(logits, positive_mask)
    if not symmetric:
        return eeg_to_image
    image_to_eeg = _multi_positive_cross_entropy(logits.T, positive_mask.T)
    return 0.5 * (eeg_to_image + image_to_eeg)


def _multi_positive_cross_entropy(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != positive_mask.shape:
        raise ValueError(f"logits and positive_mask shapes differ: {logits.shape} vs {positive_mask.shape}")
    if not torch.all(positive_mask.any(dim=1)):
        raise ValueError("Every row must have at least one positive target")
    positive_logits = logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
    numerator = torch.logsumexp(positive_logits, dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return -(numerator - denominator).mean()


def anchor_guided_contrastive_loss(
    eeg_embeddings: torch.Tensor,
    anchor_embeddings: torch.Tensor,
    target_anchor_indices: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE-style loss with category anchors as class-like targets."""

    if target_anchor_indices.ndim != 1:
        raise ValueError(f"target_anchor_indices must be 1D, got {tuple(target_anchor_indices.shape)}")
    if eeg_embeddings.shape[0] != target_anchor_indices.shape[0]:
        raise ValueError(
            "Batch size mismatch between EEG embeddings and anchor labels: "
            f"{eeg_embeddings.shape[0]} vs {target_anchor_indices.shape[0]}"
        )

    logits = _normalized_logits(eeg_embeddings, anchor_embeddings, temperature)
    labels = target_anchor_indices.to(device=logits.device, dtype=torch.long)
    if torch.any(labels < 0) or torch.any(labels >= anchor_embeddings.shape[0]):
        raise ValueError("target_anchor_indices contain out-of-range anchor ids")
    return F.cross_entropy(logits, labels)


def hybrid_eeg_anchor_contrastive_loss(
    eeg_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    anchor_embeddings: torch.Tensor,
    target_ids: torch.Tensor,
    target_anchor_indices: torch.Tensor,
    *,
    temperature: float = 0.07,
    lambda_anchor: float = 0.2,
    prototype_anchor_embeddings: torch.Tensor | None = None,
    lambda_prototype: float = 0.0,
    symmetric: bool = True,
) -> torch.Tensor:
    """Direct EEG-image alignment regularized by semantic anchor targets."""

    if lambda_anchor < 0:
        raise ValueError("lambda_anchor must be non-negative")
    if lambda_prototype < 0:
        raise ValueError("lambda_prototype must be non-negative")

    direct_loss = direct_eeg_image_contrastive_loss(
        eeg_embeddings,
        image_embeddings,
        target_ids=target_ids,
        temperature=temperature,
        symmetric=symmetric,
    )
    anchor_loss = anchor_guided_contrastive_loss(
        eeg_embeddings,
        anchor_embeddings,
        target_anchor_indices,
        temperature=temperature,
    )
    total = direct_loss + lambda_anchor * anchor_loss

    if lambda_prototype > 0:
        if prototype_anchor_embeddings is None:
            raise ValueError("prototype_anchor_embeddings are required when lambda_prototype > 0")
        prototype_loss = anchor_guided_contrastive_loss(
            eeg_embeddings,
            prototype_anchor_embeddings,
            target_anchor_indices,
            temperature=temperature,
        )
        total = total + lambda_prototype * prototype_loss

    return total


class DirectEEGImageContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, symmetric: bool = True) -> None:
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(
        self,
        eeg_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return direct_eeg_image_contrastive_loss(
            eeg_embeddings,
            image_embeddings,
            target_ids=target_ids,
            temperature=self.temperature,
            symmetric=self.symmetric,
        )


class AnchorGuidedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        eeg_embeddings: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        target_anchor_indices: torch.Tensor,
    ) -> torch.Tensor:
        return anchor_guided_contrastive_loss(
            eeg_embeddings,
            anchor_embeddings,
            target_anchor_indices,
            temperature=self.temperature,
        )


class HybridEEGAnchorContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        lambda_anchor: float = 0.2,
        lambda_prototype: float = 0.0,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.lambda_anchor = lambda_anchor
        self.lambda_prototype = lambda_prototype
        self.symmetric = symmetric

    def forward(
        self,
        eeg_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        target_anchor_indices: torch.Tensor,
        prototype_anchor_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hybrid_eeg_anchor_contrastive_loss(
            eeg_embeddings,
            image_embeddings,
            anchor_embeddings,
            target_ids,
            target_anchor_indices,
            temperature=self.temperature,
            lambda_anchor=self.lambda_anchor,
            prototype_anchor_embeddings=prototype_anchor_embeddings,
            lambda_prototype=self.lambda_prototype,
            symmetric=self.symmetric,
        )
