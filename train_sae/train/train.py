import math
import os

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from train_sae.configs.base import RunConfig

MIN_N_FLOPS_TO_SAVE = 1e15


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
            log_dict["train/loss"] = value
        else:
            log_dict[f"train/loss/{key}"] = value

    # Add L0 norm to the log dictionary
    log_dict["train/L0_norm"] = (encoded * mask[..., None] > 0).sum() / mask.sum()

    # Add the dead neurons to the log dictionary
    log_dict["train/dead_neurons"] = (
        1 - (encoded * mask[..., None] > 0).any(dim=(0, 1)).float().mean()
    )

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
    sae_model.eval()
    featurizing_model.eval()

    # Initialize the loss accumulator
    total_loss = 0
    total_mse_loss = 0
    total_sparsity_loss = 0
    total_l0_norm = 0
    total_batches = 0
    neuron_is_alive = torch.zeros(
        sae_model.encoder.out_features, dtype=torch.bool, device=config.device
    )

    # Iterate over the dataloader
    for batch in dataloader:
        del batch["labels"]
        for key in batch:
            batch[key] = batch[key].to(config.device)

        with torch.no_grad():
            features = featurizing_model(**batch).to(config.dtype)
            if config.normalize:
                features *= features.shape[-1] ** 0.5 / features.norm(
                    dim=-1, keepdim=True
                )

            encoded, decoded = sae_model(features)
            neuron_is_alive |= (
                encoded * batch["attention_mask"].unsqueeze(-1) > 0
            ).any(dim=(0, 1))

            losses = sae_model.get_losses(
                features,
                encoded,
                decoded,
                batch["attention_mask"],
                config.warmup_steps,
                config,
            )

            total_loss += losses["total"].item()
            total_mse_loss += losses["mse"].item()
            total_sparsity_loss += losses["sparsity"].item()
            total_l0_norm += (
                (encoded * batch["attention_mask"].unsqueeze(-1) > 0).sum().item()
            ) / batch["attention_mask"].sum().item()
            total_batches += 1

    # Log the evaluation metrics
    wandb.log(
        {
            "eval/loss": total_loss / total_batches,
            "eval/loss/mse": total_mse_loss / total_batches,
            "eval/loss/sparsity": total_sparsity_loss / total_batches,
            "eval/L0_norm": total_l0_norm / total_batches,
            "eval/dead_neurons": 1 - neuron_is_alive.float().mean(),
        }
    )


def train_sae(
    featurizing_model: nn.Module,
    sae_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: callable,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    config: RunConfig,
):
    cumulative_flops = 0
    # set the models to training mode
    featurizing_model.eval()
    sae_model.train()

    # Convert models to dtype
    featurizing_model = featurizing_model.to(config.dtype)
    sae_model = sae_model.to(config.dtype)

    current_step = 0
    progress_bar = tqdm(total=config.num_steps, desc="Training SAE")
    while current_step < config.num_steps:
        for batch in train_dataloader:
            del batch["labels"]
            for key in batch:
                batch[key] = batch[key].to(config.device)

            scheduler(current_step)

            with torch.no_grad():
                features = featurizing_model(**batch).to(config.dtype)

                if config.normalize:
                    features *= features.shape[-1] ** 0.5 / features.norm(
                        dim=-1, keepdim=True
                    )

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            encoded, decoded = sae_model(features)

            # calculate the loss
            losses = sae_model.get_losses(
                features,
                encoded,
                decoded,
                batch["attention_mask"],
                current_step,
                config,
            )
            losses["total"].backward()

            # update the weights
            torch.nn.utils.clip_grad_norm_(sae_model.parameters(), 1.0)
            optimizer.step()

            log_progress(losses, encoded, batch["attention_mask"])

            cumulative_flops += sae_model.flops * config.batch_size * 1024
            # check if this step crossed the threshold for the most recent 1eN flops
            # or 3eN flops boundary
            if crossed_flop_boundary(
                cumulative_flops, sae_model.flops * config.batch_size * 1024
            ):
                # save the model with number of flops in scientific notation
                save_model(
                    sae_model,
                    os.path.join(
                        wandb.run.dir,
                        f"model_{current_step}_flops_{cumulative_flops:.2e}.pt",
                    ),
                )

                # evaluate the model
                evaluate_sae(featurizing_model, sae_model, test_dataloader, config)

            current_step += 1
            progress_bar.update(1)
            losses = {key: value.item() for key, value in losses.items()}
            losses["flops"] = f"{cumulative_flops:.2e}"
            progress_bar.set_postfix(losses)

            if current_step >= config.num_steps:
                break

    progress_bar.close()
