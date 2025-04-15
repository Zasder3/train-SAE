import torch
from jaxtyping import Bool, Float
from torch import nn

from train_sae.configs.base import RunConfig
from train_sae.saes.topk import TopKSAE
from train_sae.saes.vanilla import AbstractSAE, VanillaSAE


def _get_sae_class(sae_type: str) -> type[AbstractSAE]:
    if sae_type == "vanilla":
        return VanillaSAE
    elif sae_type == "topk":
        return TopKSAE
    else:
        raise ValueError(f"Invalid SAE type: {sae_type}")


class CrossCoderSAE(nn.Module):
    def __init__(
        self,
        n_autoencoders: int,
        embed_dim: int,
        sparse_dim: int,
        sae_type: str,
        sae_kwargs: dict,
    ):
        super().__init__()
        self.autoencoders = nn.ModuleList(
            [
                _get_sae_class(sae_type)(embed_dim, sparse_dim, **sae_kwargs)
                for _ in range(n_autoencoders)
            ]
        )
        self.activation_fn = self.autoencoders[0].activation_fn

    def encode(
        self, x: list[Float[torch.Tensor, "b n d"]]
    ) -> Float[torch.Tensor, "b n s"]:
        return self.activation_fn(
            torch.sum(
                torch.stack(
                    [
                        autoencoder.encoder(x_encoder)
                        for x_encoder, autoencoder in zip(x, self.autoencoders)
                    ]
                ),
                dim=0,
            )
        )

    def forward(
        self, x: list[Float[torch.Tensor, "b n d"]]
    ) -> tuple[Float[torch.Tensor, "b n s"], list[Float[torch.Tensor, "b n d"]]]:
        encoded = self.encode(x)
        decoded = [autoencoder.decoder(encoded) for autoencoder in self.autoencoders]
        return encoded, decoded

    def get_losses(
        self,
        x: list[Float[torch.Tensor, "b n d"]],
        encoded: Float[torch.Tensor, "b n s"],
        decoded: list[Float[torch.Tensor, "b n d"]],
        mask: Bool[torch.Tensor, "b n"],
        step: int,
        config: RunConfig,
    ) -> list[dict[str, Float[torch.Tensor, "1"]]]:
        module_losses = [
            autoencoder.get_losses(x[i], encoded, decoded[i], mask, step, config)
            for i, autoencoder in enumerate(self.autoencoders)
        ]
        return module_losses

    @property
    def flops(self) -> int:
        return sum(autoencoder.flops for autoencoder in self.autoencoders)
