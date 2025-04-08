from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import nn


class AbstractTrunk(ABC, nn.Module):
    """
    Abstract base class for model trunks that extract features from inputs.
    A trunk is the part of the model that processes inputs up to a certain layer
    and returns intermediate representations.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the trunk model.
        Returns the intermediate representations from the model.
        """
        pass

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Return the embedding dimension of the model."""
        pass


class AbstractHead(ABC, nn.Module):
    """
    Abstract base class for model heads that process intermediate representations.
    A head is the part of the model that takes intermediate representations and
    produces the final output (e.g., logits).
    """

    @abstractmethod
    def forward(
        self,
        input_encodings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the head model.
        Takes intermediate representations and returns processed output.
        """
        pass

    @abstractmethod
    def input_ids_to_labels(
        self, input_ids: torch.Tensor, tokenizer: Any
    ) -> torch.Tensor:
        """
        Convert input IDs to labels suitable for loss calculation.
        """
        pass


def trunk_and_head_factory(
    model_type: str, pretrained_model_name_or_path: str, n_layers: int, **kwargs
) -> tuple[AbstractTrunk, AbstractHead]:
    """
    Factory function to create trunk and head models based on model type.

    Args:
        model_type: Type of model (e.g., "esm2")
        pretrained_model_name_or_path: Name or path of pretrained model
        n_layers: Number of layers for trunk
        **kwargs: Additional arguments to pass to model initialization

    Returns:
        A tuple of (trunk, head) models
    """
    if model_type == "esm2":
        from train_sae.models.esm2 import trunk_and_head_from_pretrained

        return trunk_and_head_from_pretrained(
            pretrained_model_name_or_path, n_layers, **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
