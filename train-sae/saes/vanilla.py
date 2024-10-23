import torch
from jaxtyping import Bool, Float
from torch import nn


class VanillaSAE(nn.Module):
    def __init__(self, embed_dim: int, sparse_dim: int, sparsity: int = 5):
        super().__init__()
        self.encoder = nn.Linear(embed_dim, sparse_dim)
        self.decoder = nn.Linear(sparse_dim, embed_dim)
        self.sparsity = sparsity

    def encode(self, x: Float[torch.Tensor, "b n d"]) -> Float[torch.Tensor, "b n s"]:
        return torch.relu(self.encoder(x))

    def forward(self, x: Float[torch.Tensor, "b n d"]) -> Float[torch.Tensor, "b n d"]:
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def get_losses(
        self,
        x: Float[torch.Tensor, "b n d"],
        encoded: Float[torch.Tensor, "b n s"],
        decoded: Float[torch.Tensor, "b n d"],
        mask: Bool[torch.Tensor, "b n"],
    ) -> dict[str, Float[torch.Tensor, "1"]]:
        losses = {}
        losses["mse"] = ((decoded - x).pow(2) * mask) / mask.sum()
        losses["sparsity"] = (
            encoded * mask @ torch.norm(encoded, dim=0) / mask.sum() * self.sparsity
        )
        losses["total"] = sum(losses.values())
        return losses
