import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from train_sae.configs.base import RunConfig


def log_progress(
    losses: dict[str, torch.Tensor],
    encoded: Float[torch.Tensor, "b n s"],
    mask: Float[torch.Tensor, "b n"],
):
    # Create a single dictionary for logging
    log_dict = {}

    # Add losses to the log dictionary
    for key, value in losses.items():
        if key == "total":
            log_dict["loss"] = value
        else:
            log_dict[f"loss/{key}"] = value

    # Add L0 norm to the log dictionary
    log_dict["L0_norm"] = (encoded * mask[..., None] > 0).sum() / mask.sum()

    # Log all metrics in a single call
    wandb.log(log_dict)


def train_sae(
    featurizing_model: nn.Module,
    sae_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    config: RunConfig,
):
    # set the models to training mode
    featurizing_model.eval()
    sae_model.train()

    # Convert models to bfloat16
    featurizing_model = featurizing_model.to(torch.bfloat16)
    sae_model = sae_model.to(torch.bfloat16)

    current_step = 0
    progress_bar = tqdm(total=config.num_steps, desc="Training SAE")
    while current_step < config.num_steps:
        for batch in dataloader:
            del batch["labels"]
            for key in batch:
                batch[key] = batch[key].to(config.device)

            with torch.no_grad():
                features = featurizing_model(**batch).to(torch.bfloat16)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            encoded, decoded = sae_model(features)

            # calculate the loss
            losses = sae_model.get_losses(
                features, encoded, decoded, batch["attention_mask"]
            )
            losses["total"].backward()

            # update the weights
            optimizer.step()

            log_progress(losses, encoded, batch["attention_mask"])

            current_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(losses)

            if current_step >= config.num_steps:
                break

    progress_bar.close()
