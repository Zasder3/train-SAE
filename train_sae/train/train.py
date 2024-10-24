import torch
import torch.nn as nn
import wandb
from jaxtyping import Float
from torch.utils.data import DataLoader

from train_sae.configs.base import RunConfig


def log_progress(
    losses: dict[str, torch.Tensor],
    encoded: Float[torch.Tensor, "b n s"],
    mask: Float[torch.Tensor, "b n"],
):
    # log the losses to wandb
    for key, value in losses.items():
        if key == "total":
            wandb.log({"loss": value})
        else:
            wandb.log({f"loss/{key}": value})

    # log the L0 norm of the encoded tensor
    wandb.log({"L0_norm": (encoded > 0).sum() / mask.sum()})


def train_sae(
    encoding_model: nn.Module,
    sae_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    config: RunConfig,
):
    # set the models to training mode
    encoding_model.eval()
    sae_model.train()

    current_step = 0
    while current_step < config.num_steps:
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(config.device)

            with torch.no_grad():
                encoding = encoding_model(**batch)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            decoded, encoded = sae_model(encoding.last_hidden_state)

            # calculate the loss
            losses = sae_model.get_losses(
                encoding, encoded, decoded, batch["attention_mask"]
            )
            losses["total"].backward()

            # update the weights
            optimizer.step()

            log_progress(losses, encoded, batch["attention_mask"])
