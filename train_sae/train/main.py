import logging
from itertools import product

import torch

import wandb
from train_sae.configs.base import RunConfig
from train_sae.configs.utils import parse_config
from train_sae.models.model import trunk_and_head_factory
from train_sae.saes.crosscoder import CrossCoderSAE
from train_sae.train.scheduler import configure_scheduler
from train_sae.train.tasks import TaskFactory
from train_sae.train.train import train_sae

logger = logging.getLogger(__name__)


def main():
    # configure the logger and wandb
    logging.basicConfig(level=logging.INFO)

    # load the configuration
    logging.info("Loading configuration")
    config = parse_config(RunConfig)
    logging.info(config)
    wandb.init(
        name=config.run_name, project=config.project_name, config=config.model_dump()
    )

    # initialize the models and datasets
    logging.info("Initializing models and datasets")
    # As a crosscoder can be seen as a generalization of a single SAE,
    # we can use it as the abstraction layer for each underlying SAE.
    # As we can train a cross-coder across a plurality of layers and models,
    # we deploy a cartesian product of featurizing models and head models.
    featurizing_models, head_models = [], []
    for featurizing_model_name, n_layers in product(
        config.featurizing_model_name, config.n_layers
    ):
        featurizing_model, head_model = trunk_and_head_factory(
            config,
            featurizing_model_name,
            n_layers,
        )
        featurizing_models.append(featurizing_model)
        head_models.append(head_model)

    logging.info(f"Number of featurizing models: {len(featurizing_models)}")

    cross_coder_model = CrossCoderSAE(
        len(featurizing_models),
        featurizing_model.embed_dim,
        config.sparse_dim,
        config.sae_type,
        config.sae_kwargs,
    ).to(config.device, config.dtype)

    if config.compile:
        featurizing_models = [
            torch.compile(featurizing_model) for featurizing_model in featurizing_models
        ]
        head_models = [torch.compile(head_model) for head_model in head_models]
        cross_coder_model = torch.compile(cross_coder_model)

    optimizer = torch.optim.Adam(
        cross_coder_model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.wd,
    )

    scheduler = configure_scheduler(optimizer, config)

    task = TaskFactory.get_task(config)
    train_dataloader, test_dataloader = task.get_dataloaders()

    # clean featurizing model name for logging
    config.featurizing_model_name = [
        name.split("/")[-1].split(".")[0] for name in config.featurizing_model_name
    ]

    # train the model
    train_sae(
        task,
        featurizing_models,
        head_models,
        cross_coder_model,
        list(product(config.featurizing_model_name, config.n_layers)),
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
        config,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
