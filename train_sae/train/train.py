import math
import os

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from train_sae.configs.base import RunConfig
from train_sae.saes.crosscoder import CrossCoderSAE
from train_sae.train.tasks import AbstractTask

MIN_N_FLOPS_TO_SAVE = 1e15


def log_progress(
    cross_coder_model_labels: list[tuple[str, int]],
    losses: list[dict[str, torch.Tensor]],
    losses_explained: list[Float[torch.Tensor, "1"]],
    encoded: Float[torch.Tensor, "b n s"],
    mask: Float[torch.Tensor, "b n"],
    neuron_is_alive: torch.Tensor,
    lr: float,
):
    # Create a single dictionary for logging
    log_dict = {}

    # Add losses to the log dictionary
    log_dict["train/loss"] = sum(loss["total"] for loss in losses)
    for loss, (model_name, layer_idx) in zip(losses, cross_coder_model_labels):
        for key, value in loss.items():
            if key != "total":
                log_dict[f"train/loss/{model_name}/layer_{layer_idx}/{key}"] = value

    # Add L0 norm to the log dictionary
    log_dict["train/L0_norm"] = (encoded * mask[..., None] > 0).sum() / mask.sum()

    # Add the LM loss explained to the log dictionary
    for loss_explained, (model_name, layer_idx) in zip(
        losses_explained, cross_coder_model_labels
    ):
        log_dict[f"train/lm_loss_explained/{model_name}/layer_{layer_idx}"] = (
            loss_explained
        )

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
    task: AbstractTask,
    featurizing_models: list[nn.Module],
    head_models: list[nn.Module],
    cross_coder_model: CrossCoderSAE,
    dataloader: DataLoader,
    config: RunConfig,
):
    # Set the models to evaluation mode
    for featurizing_model in featurizing_models:
        featurizing_model.eval()
    for head_model in head_models:
        head_model.eval()
    cross_coder_model.eval()

    # Initialize the loss accumulator
    total_loss = 0
    total_mse_loss = 0
    total_sparsity_loss = 0
    total_l0_norm = 0
    total_lm_loss_explained = 0
    total_batches = 0
    neuron_is_alive = torch.zeros(
        cross_coder_model.autoencoders[0].encoder.out_features,
        dtype=torch.bool,
        device=config.device,
    )

    # Iterate over the dataloader
    for batch in dataloader:
        for key in batch:
            batch[key] = batch[key].to(config.device)

        with torch.inference_mode():
            features = []
            normalizing_factors = []
            for featurizing_model in featurizing_models:
                current_features = featurizing_model(**batch).to(config.dtype)
                if config.normalize:
                    normalizing_factor = (
                        featurizing_model.embed_dim** 0.5
                        / current_features.norm(dim=-1, keepdim=True)
                    )
                    current_features *= normalizing_factor
                else:
                    normalizing_factor = torch.ones_like(current_features)
                normalizing_factors.append(normalizing_factor)
                features.append(current_features)

            encoded, decoded = cross_coder_model(features)
            neuron_is_alive |= (
                encoded * batch["attention_mask"].unsqueeze(-1) > 0
            ).any(dim=(0, 1))

            losses = cross_coder_model.get_losses(
                features,
                encoded,
                decoded,
                batch["attention_mask"],
                config.sparsity_warmup_steps,
                config,
            )

            total_loss += sum(loss["total"].item() for loss in losses)
            total_mse_loss += sum(loss["mse"].item() for loss in losses)
            total_sparsity_loss += sum(loss["sparsity"].item() for loss in losses)
            total_l0_norm += (
                (encoded * batch["attention_mask"].unsqueeze(-1) > 0).sum().item()
            ) / batch["attention_mask"].sum().item()
            total_lm_loss_explained += sum(
                lm_loss_explained(
                    head_model,
                    features[i],
                    normalizing_factors[i],
                    decoded[i],
                    batch["attention_mask"],
                    head_model.input_ids_to_labels(
                        batch["input_ids"],
                        batch["labels"],
                        task.tokenizer,
                    ),
                ).item()
                for i, head_model in enumerate(head_models)
            )
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
    task: AbstractTask,
    featurizing_models: list[nn.Module],
    head_models: list[nn.Module],
    cross_coder_model: CrossCoderSAE,
    cross_coder_model_labels: list[tuple[str, int]],
    optimizer: torch.optim.Optimizer,
    scheduler: callable,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    config: RunConfig,
):
    cumulative_flops = 0
    # set the models to training mode
    for featurizing_model in featurizing_models:
        featurizing_model.eval()
    for head_model in head_models:
        head_model.eval()
    cross_coder_model.train()

    current_step = 0
    progress_bar = tqdm(total=config.num_steps, desc="Training SAE")

    # Initialize neuron tracking
    neuron_is_alive = torch.zeros(
        cross_coder_model.autoencoders[0].encoder.out_features,
        dtype=torch.bool,
        device=config.device,
    )
    steps_since_reset = 0
    steps_per_reset = config.num_test_samples // config.batch_size

    while current_step < config.num_steps:
        for batch in train_dataloader:
            for key in batch:
                batch[key] = batch[key].to(config.device)

            scheduler(current_step)

            features = []
            normalizing_factors = []
            for featurizing_model in featurizing_models:
                with torch.no_grad():
                    current_features = featurizing_model(**batch).to(config.dtype)

                    if config.normalize:
                        normalizing_factor = (
                            featurizing_model.embed_dim** 0.5
                            / current_features.norm(dim=-1, keepdim=True)
                        )
                        current_features *= normalizing_factor
                    else:
                        normalizing_factor = torch.ones_like(current_features)
                    normalizing_factors.append(normalizing_factor)
                    features.append(current_features)

            # forward pass
            encoded, decoded = cross_coder_model(features)

            # calculate the loss
            losses = cross_coder_model.get_losses(
                features,
                encoded,
                decoded,
                batch["attention_mask"],
                current_step,
                config,
            )
            total_loss = sum(loss["total"] for loss in losses)
            total_loss.backward()

            # update the weights
            torch.nn.utils.clip_grad_norm_(cross_coder_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # get lm loss explained
            explained_losses = []
            for i, head_model in enumerate(head_models):
                loss_explained = lm_loss_explained(
                    head_model,
                    features[i],
                    normalizing_factors[i],
                    decoded[i],
                    batch["attention_mask"],
                    head_model.input_ids_to_labels(
                        batch["input_ids"],
                        batch["labels"],
                        task.tokenizer,
                    ),
                )
                explained_losses.append(loss_explained)

            # Update neuron_is_alive tensor
            with torch.no_grad():
                neuron_is_alive |= (
                    encoded * batch["attention_mask"][..., None] > 0
                ).any(dim=(0, 1))
            # Reset and log neuron tracking if needed
            steps_since_reset += 1
            if steps_since_reset >= steps_per_reset:
                log_progress(
                    cross_coder_model_labels,
                    losses,
                    explained_losses,
                    encoded,
                    batch["attention_mask"],
                    neuron_is_alive,
                    optimizer.param_groups[0]["lr"],
                )

                neuron_is_alive.zero_()
                steps_since_reset = 0
            else:
                log_progress(
                    cross_coder_model_labels,
                    losses,
                    explained_losses,
                    encoded,
                    batch["attention_mask"],
                    None,
                    optimizer.param_groups[0]["lr"],
                )

            cumulative_flops += (
                cross_coder_model.flops * config.batch_size * task.max_tokens
            )
            # check if this step crossed the threshold for the most recent 1eN flops
            # or 3eN flops boundary
            if (
                crossed_flop_boundary(
                    cumulative_flops,
                    cross_coder_model.flops * config.batch_size * task.max_tokens,
                )
                or current_step == config.num_steps - 1
            ):
                # save the model with number of flops in scientific notation
                save_model(
                    cross_coder_model,
                    os.path.join(
                        wandb.run.dir,
                        f"model_{current_step}_flops_{cumulative_flops:.2e}.pt",
                    ),
                )
                # delete unused tensors to free up memory
                del features, encoded, decoded

                # evaluate the model
                evaluate_sae(
                    task,
                    featurizing_models,
                    head_models,
                    cross_coder_model,
                    test_dataloader,
                    config,
                )

            current_step += 1
            progress_bar.update(1)
            info = {}
            info["flops"] = f"{cumulative_flops:.2e}"
            info["loss"] = total_loss.item()
            progress_bar.set_postfix(info)

            if current_step >= config.num_steps:
                break

    progress_bar.close()
