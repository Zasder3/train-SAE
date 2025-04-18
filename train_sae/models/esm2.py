# Code adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esm.py
# ruff: noqa: E501
from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForMaskedLM, EsmModel
from transformers.models.esm.modeling_esm import EsmLMHead
from transformers.models.esm.tokenization_esm import EsmTokenizer

from train_sae.models.model import AbstractHead, AbstractTrunk


def trunk_and_head_from_pretrained(
    pretrained_model_name_or_path: str, n_layers: int, **kwargs
) -> tuple[AbstractTrunk, AbstractHead]:
    """
    Create a truncated model from a pretrained ESM model.

    Args:
        pretrained_model_name_or_path (str): Name or path of pretrained model
        n_layers (int): Number of layers to keep
        **kwargs: Additional arguments to pass to from_pretrained
    """
    original_model = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path, **kwargs
    )
    esm = original_model.esm
    lm_head = original_model.lm_head
    return TruncatedEsm2(esm, n_layers), Esm2Head(esm, lm_head, n_layers)


class TruncatedEsm2(AbstractTrunk):
    """
    A modified version of ESM2 that only processes the first n layers and returns
    the intermediate representation from the residual stream.
    """

    def __init__(self, original_model: EsmModel, n_layers: int):
        """
        Initialize a truncated version of an ESM model.

        Args:
            original_model (EsmModel): The original ESM model to truncate
            n_layers (int): Number of layers to keep (must be <= original model's layer count)
        """
        super().__init__()

        if n_layers > len(original_model.encoder.layer):
            raise ValueError(
                f"Requested {n_layers} layers but model only has "
                f"{len(original_model.encoder.layer)} layers."
            )

        self.config = original_model.config
        self.embeddings = original_model.embeddings

        # Only keep the first n layers
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [
                layer
                for i, layer in enumerate(original_model.encoder.layer)
                if i < n_layers
            ]
        )

        # Keep layer norm if present in original model
        self.encoder.emb_layer_norm_after = original_model.encoder.emb_layer_norm_after

        self.pooler = original_model.pooler
        self.n_layers = n_layers

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Forward pass through truncated model."""
        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        # Prepare attention mask
        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=embedding_output.dtype
            )
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
                embedding_output.dtype
            ).min

        # Initialize outputs
        hidden_states = embedding_output

        # Process through n layers
        for layer in self.encoder.layer:
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
            )
            hidden_states = layer_outputs[0]

        # Return hidden states
        return hidden_states

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension of the model."""
        return self.embeddings.word_embeddings.embedding_dim


class Esm2Head(AbstractHead):
    """
    The head layers of the ESM-2 model that can decode the truncated intermediate.
    """

    def __init__(self, original_model: EsmModel, lm_head: EsmLMHead, n_layers: int):
        """
        Initialize the head layers of the ESM-2 model.

        Args:
            original_model (EsmModel): The original ESM model to truncate
            lm_head (EsmLMHead): The language model head from the original model
            n_layers (int): Number of layers to keep (must be <= original model's layer count)
        """
        super().__init__()

        if n_layers > len(original_model.encoder.layer):
            raise ValueError(
                f"Requested {n_layers} layers but model only has "
                f"{len(original_model.encoder.layer)} layers."
            )

        self.config = original_model.config
        self.embeddings = original_model.embeddings

        # Only keep the last n layers
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [
                layer
                for i, layer in enumerate(original_model.encoder.layer)
                if i >= n_layers
            ]
        )

        # Keep layer norm if present in original model
        self.encoder.emb_layer_norm_after = original_model.encoder.emb_layer_norm_after

        self.pooler = original_model.pooler
        self.n_layers = n_layers

        self.lm_head = lm_head

    def forward(
        self,
        input_encodings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through head layers."""
        # Prepare attention mask
        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=input_encodings.dtype
            )
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
                input_encodings.dtype
            ).min

        # Initialize outputs
        hidden_states = input_encodings

        # Process through n layers
        for layer in self.encoder.layer:
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
            )
            hidden_states = layer_outputs[0]

        if self.encoder.emb_layer_norm_after is not None:
            hidden_states = self.encoder.emb_layer_norm_after(hidden_states)

        return self.lm_head(hidden_states)

    def input_ids_to_labels(
        self, input_ids: torch.Tensor, labels: torch.Tensor, tokenizer: EsmTokenizer
    ) -> torch.Tensor:
        """Convert input IDs to labels."""
        # set all special tokens to -100
        for special_token in tokenizer.all_special_ids:
            input_ids[input_ids == special_token] = -100

        return input_ids
