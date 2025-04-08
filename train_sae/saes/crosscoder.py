import torch
from jaxtyping import Bool, Float
from torch import nn

from train_sae.configs.base import RunConfig
from train_sae.saes.vanilla import VanillaSAE


class CrossCoderSAE(nn.Module):
    def __init__(
        self,
        n_autoencoders: int,
        embed_dim: int,
        sparse_dim: int,
        sparsity: float = 5.0,
    ):
        super().__init__()
        self.autoencoders = nn.ModuleList(
            [VanillaSAE(embed_dim, sparse_dim, sparsity) for _ in range(n_autoencoders)]
        )

    def encode(self, x: Float[torch.Tensor, "b n d"]) -> Float[torch.Tensor, "b n s"]:
        return torch.relu(
            torch.sum(
                torch.stack(
                    [autoencoder.encode(x) for autoencoder in self.autoencoders]
                ),
                dim=0,
            )
        )

    def forward(
        self, x: Float[torch.Tensor, "b n d"]
    ) -> tuple[Float[torch.Tensor, "b n s"], list[Float[torch.Tensor, "b n d"]]]:
        encoded = self.encode(x)
        decoded = [autoencoder.decode(encoded) for autoencoder in self.autoencoders]
        return encoded, decoded

    def get_losses(
        self,
        x: Float[torch.Tensor, "b n d"],
        encoded: Float[torch.Tensor, "b n s"],
        decoded: list[Float[torch.Tensor, "b n d"]],
        mask: Bool[torch.Tensor, "b n"],
        step: int,
        config: RunConfig,
    ) -> list[dict[str, Float[torch.Tensor, "1"]]]:
        module_losses = [
            autoencoder.get_losses(x, encoded, decoded[i], mask, step, config)
            for i, autoencoder in enumerate(self.autoencoders)
        ]
        return module_losses

    @property
    def flops(self) -> int:
        return sum(autoencoder.flops for autoencoder in self.autoencoders)
