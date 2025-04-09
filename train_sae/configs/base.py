from typing import Literal, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from train_sae.train.tasks import TaskFactory


class RunConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # task config
    task: str = Field(
        description="The task to run for training.",
        choices=TaskFactory.task_names,
    )
    task_kwargs: dict = Field(default={}, description="Keyword arguments for the task.")

    # dataset features
    dataset_dir: str = Field(description="Path to the dataset directory.")
    samples_in_dataset: int = Field(description="Number of samples in the dataset.")
    num_test_samples: int = Field(description="Number of samples to use for testing.")

    # hyperparameters
    num_steps: int = Field(description="Number of steps to train for.")
    lr_scheduler: Literal["constant", "cosine", "linear_decay"] = Field(
        default="linear_decay", description="Learning rate scheduler."
    )
    lr_warmup_steps: int = Field(description="Number of warmup steps.")
    batch_size: int = Field(description="Batch size.")
    normalize: Union[bool, None] = Field(
        default=False, description="Whether to normalize the data."
    )
    lr: float = Field(description="Learning rate.")
    beta1: Union[float, None] = Field(default=0.9, description="Adam beta1.")
    beta2: Union[float, None] = Field(default=0.999, description="Adam beta2.")
    wd: Union[float, None] = Field(default=0.00, description="Weight decay.")

    # performance config
    compile: bool = Field(default=False, description="Whether to compile the model.")
    device: str = Field(default="cpu", description="Device to train on.")
    dtype: torch.dtype = Field(default=torch.float32, description="Data type to use.")

    # model config
    model_type: str = Field(
        description="Type of model to train.",
        choices=["esm2", "transformer"],
    )
    featurizing_model_name: Union[str, list[str]] = Field(
        description="Name of the model(s) to train."
    )
    featurizing_model_kwargs: dict = Field(
        default={}, description="Keyword arguments for the featurizing model."
    )
    n_layers: Union[int, list[int]] = Field(
        description="Layer(s) to extract features from."
    )
    sparsity: float = Field(description="Sparsity loss weight.")
    sparsity_warmup_steps: Union[int, None] = Field(
        default=1, description="Sparsity warmup steps."
    )
    sparse_dim: int = Field(description="Sparse dimension.")

    # wandb config
    run_name: Union[str, None] = Field(default=None, description="Name of the run.")
    project_name: str = Field(description="Name of the project.")

    @field_validator("dtype", mode="before")
    @classmethod
    def parse_dtype(cls, value):
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            if value.startswith("torch."):
                return getattr(torch, value.split(".")[1])
            return getattr(torch, value)
        raise ValueError(f"Value must be a string or torch.dtype, got {type(value)}")

    @field_validator("featurizing_model_name", "n_layers", mode="before")
    @classmethod
    def parse_list(cls, value):
        if isinstance(value, str):
            return [value]
        return value
