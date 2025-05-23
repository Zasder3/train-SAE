from abc import ABC, abstractmethod

import torch
from jaxtyping import Bool, Float
from torch import nn

from train_sae.configs.base import RunConfig


class AbstractSAE(nn.Module, ABC):
    def __init__(self, embed_dim: int, sparse_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.sparse_dim = sparse_dim

    @abstractmethod
    def encode(self, x: Float[torch.Tensor, "b n d"]) -> Float[torch.Tensor, "b n s"]:
        pass

    @abstractmethod
    def activation_fn(
        self, x: Float[torch.Tensor, "b n s"]
    ) -> Float[torch.Tensor, "b n s"]:
        pass

    def forward(
        self, x: Float[torch.Tensor, "b n d"]
    ) -> tuple[Float[torch.Tensor, "b n s"], Float[torch.Tensor, "b n d"]]:
        encoded = self.activation_fn(self.encode(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

    @abstractmethod
    def get_losses(
        self,
        x: Float[torch.Tensor, "b n d"],
        encoded: Float[torch.Tensor, "b n s"],
        decoded: Float[torch.Tensor, "b n d"],
        mask: Bool[torch.Tensor, "b n"],
        step: int,
        config: RunConfig,
    ) -> dict[str, Float[torch.Tensor, "1"]]:
        pass

    @property
    @abstractmethod
    def flops(self) -> int:
        pass


class VanillaSAE(AbstractSAE):
    def __init__(self, embed_dim: int, sparse_dim: int, sparsity: float = 5.0):
        super().__init__(embed_dim, sparse_dim)
        self.encoder = nn.Linear(embed_dim, sparse_dim)
        self.decoder = nn.Linear(sparse_dim, embed_dim)
        self.encoder.weight.data /= (
            torch.norm(self.encoder.weight.data, dim=0) * 10
        )  # normalize weights to have norm 0.1
        self.decoder.weight.data = self.encoder.weight.data.t().clone()
        # set biases to zero
        self.encoder.bias.data.zero_()
        self.decoder.bias.data.zero_()
        self.sparsity = sparsity

    def activation_fn(
        self, x: Float[torch.Tensor, "b n s"]
    ) -> Float[torch.Tensor, "b n s"]:
        return torch.relu(x)

    def encode(self, x: Float[torch.Tensor, "b n d"]) -> Float[torch.Tensor, "b n s"]:
        return self.encoder(x)

    def get_losses(
        self,
        x: Float[torch.Tensor, "b n d"],
        encoded: Float[torch.Tensor, "b n s"],
        decoded: Float[torch.Tensor, "b n d"],
        mask: Bool[torch.Tensor, "b n"],
        step: int,
        config: RunConfig,
    ) -> dict[str, Float[torch.Tensor, "1"]]:
        mask = mask.unsqueeze(-1)
        losses = {}
        losses["mse"] = ((decoded - x).pow(2) * mask).sum() / mask.sum()
        losses["sparsity"] = (
            # this mask operation might be having some unintentional side effects
            (encoded * mask @ torch.norm(self.decoder.weight, dim=0)).sum()
            / mask.sum()
            * min((step + 1) / config.sparsity_warmup_steps, 1)
            * self.sparsity
        )
        losses["total"] = sum(losses.values())
        return losses

    @property
    def flops(self) -> int:
        return 6 * self.encoder.in_features * self.encoder.out_features
