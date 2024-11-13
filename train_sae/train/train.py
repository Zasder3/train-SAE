import math
import os

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import wandb
from train_sae.configs.base import RunConfig

MIN_N_FLOPS_TO_SAVE = 1e15


def log_progress(
    losses: dict[str, torch.Tensor],
    loss_explained: Float[torch.Tensor, "1"],
    encoded: Float[torch.Tensor, "b n s"],
    mask: Float[torch.Tensor, "b n"],
    neuron_is_alive: torch.Tensor,
    lr: float,
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

    # Add the LM loss explained to the log dictionary
    log_dict["train/lm_loss_explained"] = loss_explained

    # Add the dead neurons to the log dictionary
    if neuron_is_alive is not None:
        log_dict["train/dead_neurons"] = 1 - neuron_is_alive.float().mean()

    # Add the learning rate to the log dictionary
    log_dict["hparams/lr"] = lr

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


@torch.inference_mode()
def lm_loss_explained(
    head_model: nn.Module,
    features: Float[torch.Tensor, "b n d"],
    normalizing_factor: Float[torch.Tensor, "b n d"],
    decoded: Float[torch.Tensor, "b n d"],
    mask: Float[torch.Tensor, "b n"],
    labels: Float[torch.Tensor, "b n"],
) -> Float[torch.Tensor, "1"]:
    real_logits = head_model(features / normalizing_factor, attention_mask=mask)
    reconstructed_logits = head_model(decoded / normalizing_factor, attention_mask=mask)

    real_loss = nn.functional.cross_entropy(
        real_logits.view(-1, real_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    reconstructed_loss = nn.functional.cross_entropy(
        reconstructed_logits.view(-1, reconstructed_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    null_loss = nn.functional.cross_entropy(
        torch.zeros_like(real_logits).view(-1, real_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    return 1 - (reconstructed_loss - real_loss) / (null_loss - real_loss)


def evaluate_sae(
    tokenizer: PreTrainedTokenizer,
    featurizing_model: nn.Module,
    head_model: nn.Module,
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
    total_lm_loss_explained = 0
    total_batches = 0
    neuron_is_alive = torch.zeros(
        sae_model.encoder.out_features, dtype=torch.bool, device=config.device
    )

    # Iterate over the dataloader
    for batch in dataloader:
        del batch["labels"]
        for key in batch:
            batch[key] = batch[key].to(config.device)

        with torch.inference_mode():
            features = featurizing_model(**batch).to(config.dtype)
            if config.normalize:
                normalizing_factor = featurizing_model.embed_dim**0.5 / features.norm(
                    dim=-1, keepdim=True
                )
                features *= normalizing_factor
            else:
                normalizing_factor = torch.ones_like(features)

            encoded, decoded = sae_model(features)
            neuron_is_alive |= (
                encoded * batch["attention_mask"].unsqueeze(-1) > 0
            ).any(dim=(0, 1))

            losses = sae_model.get_losses(
                features,
                encoded,
                decoded,
                batch["attention_mask"],
                config.sparsity_warmup_steps,
                config,
            )

            total_loss += losses["total"].item()
            total_mse_loss += losses["mse"].item()
            total_sparsity_loss += losses["sparsity"].item()
            total_l0_norm += (
                (encoded * batch["attention_mask"].unsqueeze(-1) > 0).sum().item()
            ) / batch["attention_mask"].sum().item()
            total_lm_loss_explained += lm_loss_explained(
                head_model,
                features,
                normalizing_factor,
                decoded,
                batch["attention_mask"],
                head_model.input_ids_to_labels(batch["input_ids"], tokenizer),
            ).item()
            total_batches += 1

    # Log the evaluation metrics
    wandb.log(
        {
            "eval/loss": total_loss / total_batches,
            "eval/loss/mse": total_mse_loss / total_batches,
            "eval/loss/sparsity": total_sparsity_loss / total_batches,
            "eval/L0_norm": total_l0_norm / total_batches,
            "eval/lm_loss_explained": total_lm_loss_explained / total_batches,
            "eval/dead_neurons": 1 - neuron_is_alive.float().mean(),
        }
    )


def train_sae(
    tokenizer: PreTrainedTokenizer,
    featurizing_model: nn.Module,
    head_model: nn.Module,
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
    head_model = head_model.to(config.dtype)
    sae_model = sae_model.to(config.dtype)

    current_step = 0
    progress_bar = tqdm(total=config.num_steps, desc="Training SAE")

    # Initialize neuron tracking
    neuron_is_alive = torch.zeros(
        sae_model.encoder.out_features, dtype=torch.bool, device=config.device
    )
    steps_since_reset = 0
    steps_per_reset = config.num_test_samples // config.batch_size

    while current_step < config.num_steps:
        for batch in train_dataloader:
            del batch["labels"]
            for key in batch:
                batch[key] = batch[key].to(config.device)

            scheduler(current_step)

            with torch.no_grad():
                features = featurizing_model(**batch).to(config.dtype)

                if config.normalize:
                    normalizing_factor = (
                        featurizing_model.embed_dim** 0.5
                        / features.norm(dim=-1, keepdim=True)
                    )
                    features *= normalizing_factor
                else:
                    normalizing_factor = torch.ones_like(features)
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

            # get lm loss explained
            loss_explained = lm_loss_explained(
                head_model,
                features,
                normalizing_factor,
                decoded,
                batch["attention_mask"],
                head_model.input_ids_to_labels(batch["input_ids"], tokenizer),
            )

            # Update neuron_is_alive tensor
            neuron_is_alive |= (encoded * batch["attention_mask"][..., None] > 0).any(
                dim=(0, 1)
            )
            # Reset and log neuron tracking if needed
            steps_since_reset += 1
            if steps_since_reset >= steps_per_reset:
                log_progress(
                    losses,
                    loss_explained,
                    encoded,
                    batch["attention_mask"],
                    neuron_is_alive,
                    optimizer.param_groups[0]["lr"],
                )

                neuron_is_alive.zero_()
                steps_since_reset = 0
            else:
                log_progress(
                    losses,
                    loss_explained,
                    encoded,
                    batch["attention_mask"],
                    None,
                    optimizer.param_groups[0]["lr"],
                )

            cumulative_flops += sae_model.flops * config.batch_size * 1024
            # check if this step crossed the threshold for the most recent 1eN flops
            # or 3eN flops boundary
            if (
                crossed_flop_boundary(
                    cumulative_flops, sae_model.flops * config.batch_size * 1024
                )
                or current_step == config.num_steps - 1
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
                evaluate_sae(
                    tokenizer,
                    featurizing_model,
                    head_model,
                    sae_model,
                    test_dataloader,
                    config,
                )

            current_step += 1
            progress_bar.update(1)
            losses = {key: value.item() for key, value in losses.items()}
            losses["flops"] = f"{cumulative_flops:.2e}"
            progress_bar.set_postfix(losses)

            if current_step >= config.num_steps:
                break

    progress_bar.close()
