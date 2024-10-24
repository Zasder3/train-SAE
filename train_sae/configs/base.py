from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    # hyperparameters
    dataset_dir: str = Field(description="Path to the dataset directory.")
    num_steps: int = Field(description="Number of steps to train for.")
    batch_size: int = Field(description="Batch size.")
    lr: float = Field(description="Learning rate.")
    device: str = Field(default="cpu", description="Device to train on.")

    # model config
    model_name: str = Field(description="Name of the model to train.")
    n_layers: int = Field(description="Layer to extract features from.")
    sparsity: float = Field(description="Sparsity loss weight.")
    sparse_dim: int = Field(description="Sparse dimension.")

    # wandb config
    run_name: str = Field(description="Name of the run.")
    project_name: str = Field(description="Name of the project.")
