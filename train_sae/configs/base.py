from typing import Union

import torch
from pydantic import BaseModel, Field, field_validator


class RunConfig(BaseModel):
    # hyperparameters
    dataset_dir: str = Field(description="Path to the dataset directory.")
    num_steps: int = Field(description="Number of steps to train for.")
    batch_size: int = Field(description="Batch size.")
    lr: float = Field(description="Learning rate.")

    # performance config
    device: str = Field(default="cpu", description="Device to train on.")
    dtype: torch.dtype = Field(default=torch.float32, description="Data type to use.")

    # model config
    featurizing_model_name: str = Field(description="Name of the model to train.")
    n_layers: int = Field(description="Layer to extract features from.")
    sparsity: float = Field(description="Sparsity loss weight.")
    sparse_dim: int = Field(description="Sparse dimension.")

    # wandb config
    run_name: Union[str, None] = Field(default=None, description="Name of the run.")
    project_name: str = Field(description="Name of the project.")

    @field_validator("dtype")
    @classmethod
    def parse_dtype(cls, value):
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            if value.startswith("torch."):
                return getattr(torch, value.split(".")[1])
            return getattr(torch, value)
        raise ValueError(f"Value must be a string or torch.dtype, got {type(value)}")
