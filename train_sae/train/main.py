import logging

import numpy as np
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
    test_set_generator = np.random.default_rng(42)
    all_indices = np.arange(config.samples_in_dataset())
    test_set_generator.shuffle(all_indices)
    train_dataset = FastaDataset(
        config.dataset_dir, tokenizer, indices=all_indices[: -config.num_test_samples]
    )
    test_dataset = FastaDataset(
        config.dataset_dir, tokenizer, indices=all_indices[-config.num_test_samples :]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # train the model
    train_sae(
        featurizing_model,
        sae_model,
        optimizer,
        train_dataloader,
        test_dataloader,
        config,
    )


if __name__ == "__main__":
    main()
