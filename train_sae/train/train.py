import math

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from train_sae.configs.base import RunConfig

MIN_N_FLOPS_TO_SAVE = 1e17


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


def crossed_flop_boundary(cumulative_flops: int, step_flops: int) -> bool:
    # Check if the cumulative flops crossed the threshold for the most recent 1eN flops
    # or 3eN flops boundary
    nearest_power_of_10 = math.floor(math.log10(cumulative_flops))
    if cumulative_flops < MIN_N_FLOPS_TO_SAVE:
        return False
    return (
        cumulative_flops >= 10**nearest_power_of_10
        and cumulative_flops - step_flops < 10**nearest_power_of_10
    ) or (
        cumulative_flops >= 3 * 10**nearest_power_of_10
        and cumulative_flops - step_flops < 3 * 10**nearest_power_of_10
    )


def save_model(model: nn.Module, path: str):
    # Save the model to the specified path
    torch.save(model.state_dict(), path)


def evaluate_sae(
    featurizing_model: nn.Module,
    sae_model: nn.Module,
    dataloader: DataLoader,
    config: RunConfig,
):
    # Set the models to evaluation mode
    featurizing_model.eval()
    ...


def train_sae(
    featurizing_model: nn.Module,
    sae_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    config: RunConfig,
):
    cumulative_flops = 0
    # set the models to training mode
    featurizing_model.eval()
    sae_model.train()

    # Convert models to bfloat16
    featurizing_model = featurizing_model.to(torch.bfloat16)
    sae_model = sae_model.to(torch.bfloat16)

    current_step = 0
    progress_bar = tqdm(total=config.num_steps, desc="Training SAE")
    while current_step < config.num_steps:
        for batch in train_dataloader:
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

            cumulative_flops += sae_model.get_flops()
            # check if this step crossed the threshold for the most recent 1eN flops
            # or 3eN flops boundary
            if crossed_flop_boundary(cumulative_flops, sae_model.get_flops()):
                # save the model with number of flops in scientific notation
                save_model(
                    sae_model, f"model_{current_step}_flops_{cumulative_flops:.2e}.pt"
                )

                # evaluate the model
                evaluate_sae(featurizing_model, sae_model, test_dataloader, config)

            current_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(losses)

            if current_step >= config.num_steps:
                break

    progress_bar.close()
