import pytest
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train_sae.models.esm2 import trunk_and_head_from_pretrained


@pytest.fixture
def small_esm_model():
    return "facebook/esm2_t6_8M_UR50D"


def test_truncated_esm_hidden_state(small_esm_model):
    # Load the full model
    full_model = AutoModelForMaskedLM.from_pretrained(small_esm_model)
    tokenizer = AutoTokenizer.from_pretrained(small_esm_model)

    # Create a truncated model with all layers
    n_layers = len(full_model.esm.encoder.layer) // 2
    truncated_model, _ = trunk_and_head_from_pretrained(small_esm_model, n_layers)

    # Create a small input
    inputs = tokenizer("MVQV", return_tensors="pt")

    # Get hidden states from both models
    with torch.no_grad():
        full_output = full_model(**inputs, output_hidden_states=True).hidden_states[
            n_layers
        ]
        truncated_output = truncated_model(**inputs)

    # Check if the hidden states are equal
    assert torch.allclose(
        full_output, truncated_output, atol=1e-5
    ), "Hidden states from full and truncated models are not equal"


def test_truncated_to_head_esm(small_esm_model):
    # Load the full model
    full_model = AutoModelForMaskedLM.from_pretrained(small_esm_model)
    tokenizer = AutoTokenizer.from_pretrained(small_esm_model)

    # Create a truncated model with all layers
    n_layers = len(full_model.esm.encoder.layer) // 2
    truncated_model, head_model = trunk_and_head_from_pretrained(
        small_esm_model, n_layers
    )

    # Create a small input
    inputs = tokenizer("MVQV", return_tensors="pt")

    # Get hidden states from both models
    with torch.no_grad():
        full_logits = full_model(**inputs).logits
        truncated_output = truncated_model(**inputs)
        truncated_logits = head_model(
            truncated_output, attention_mask=inputs["attention_mask"]
        )

    # Check if the head model is the same as the last layer
    assert torch.allclose(
        truncated_logits,
        full_logits,
        atol=1e-5,
    ), "Head model does not match the last layer of the full model"
