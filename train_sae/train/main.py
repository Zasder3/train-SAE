import logging

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

import wandb
from train_sae.configs.base import RunConfig
from train_sae.configs.utils import parse_config
from train_sae.models.esm2 import TruncatedEsm2
from train_sae.saes.vanilla import VanillaSAE
from train_sae.train.datasets.fasta import FastaDataset
from train_sae.train.train import train_sae


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
    featurizing_model = TruncatedEsm2.from_pretrained(
        config.featurizing_model_name, config.n_layers, device_map=config.device
    )
    sae_model = VanillaSAE(
        featurizing_model.embed_dim, config.sparse_dim, config.sparsity
    ).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.featurizing_model_name)
    optimizer = torch.optim.AdamW(
        sae_model.parameters(),
        lr=config.lr,
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataset = FastaDataset(config.dataset_dir, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # train the model
    train_sae(featurizing_model, sae_model, optimizer, dataloader, config)


if __name__ == "__main__":
    main()
