import torch
from jaxtyping import Bool, Float

from train_sae.configs.base import RunConfig
from train_sae.saes.vanilla import VanillaSAE


class TopKSAE(VanillaSAE):
    def __init__(self, embed_dim: int, sparse_dim: int, topk: float = 0.1):
        super().__init__(embed_dim, sparse_dim)
        self.topk = topk

    def encode(self, x: Float[torch.Tensor, "b n d"]) -> Float[torch.Tensor, "b n s"]:
        topk_values, topk_indices = torch.topk(x, self.topk, dim=-1)
        return_values = torch.zeros_like(x)
        return_values.scatter_(dim=-1, index=topk_indices, src=topk_values)
        return return_values

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
        losses["total"] = sum(losses.values())
        return losses
