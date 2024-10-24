import pytest
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train_sae.models.esm2 import TruncatedEsm2


@pytest.fixture
def small_esm_model():
    return "facebook/esm2_t6_8M_UR50D"


def test_truncated_esm_hidden_state(small_esm_model):
    # Load the full model
    full_model = AutoModelForMaskedLM.from_pretrained(small_esm_model)
    full_model.esm.encoder.emb_layer_norm_after = None
    tokenizer = AutoTokenizer.from_pretrained(small_esm_model)

    # Create a truncated model with all layers
    n_layers = len(full_model.esm.encoder.layer)
    truncated_model = TruncatedEsm2.from_pretrained(small_esm_model, n_layers)

    # Create a small input
    inputs = tokenizer("MVQV", return_tensors="pt")

    # Get hidden states from both models
    with torch.no_grad():
        full_output = full_model.esm(**inputs, output_hidden_states=True).hidden_states[
            -1
        ]
        truncated_output = truncated_model(**inputs)

    # Check if the hidden states are equal
    assert torch.allclose(
        full_output, truncated_output, atol=1e-5
    ), "Hidden states from full and truncated models are not equal"
