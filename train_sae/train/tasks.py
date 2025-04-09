from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from train_sae.configs.base import RunConfig
from train_sae.train.datasets.fasta import FastaDataset


class AbstractTask(ABC):
    tokenizer: AutoTokenizer
    max_tokens: int

    @abstractmethod
    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Get the train and test dataloaders for the task.

        Returns:
            A tuple containing (train_dataloader, test_dataloader)
        """
        pass


class ESMMLMTask(AbstractTask):
    def __init__(self, config: RunConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.featurizing_model_name)
        self.max_tokens = 1024

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Get the train and test dataloaders for the ESM MLM task."""
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        test_set_generator = np.random.default_rng(42)
        all_indices = np.arange(self.config.samples_in_dataset)
        test_set_generator.shuffle(all_indices)
        train_dataset = FastaDataset(
            self.config.dataset_dir,
            self.tokenizer,
            indices=all_indices[: -self.config.num_test_samples],
        )
        test_dataset = FastaDataset(
            self.config.dataset_dir,
            self.tokenizer,
            indices=all_indices[-self.config.num_test_samples :],
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True,
        )
        return train_dataloader, test_dataloader


# TODO: make sure we get a working version of this
class GrokkingTask(AbstractTask):
    def __init__(self, config: RunConfig):
        self.config = config
        self.tokenizer = None
        self.max_tokens = 4

        self.prime = config.task_kwargs.get("prime", 97)
        self.task_name = config.task_kwargs.get("task_name", "grokking")
        assert self.task_name in ["addition", "multiplication", "division"]

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        if self.task_name == "addition":
            data = []
            for a, b in product(range(self.prime), range(self.prime)):
                data.append([a, 97, b, 98, (a + b) % self.prime])
        elif self.task_name == "multiplication":
            data = []
            for a, b in product(range(self.prime), range(self.prime)):
                data.append([a, 97, b, 98, (a * b) % self.prime])
        elif self.task_name == "division":
            data = []
            for a, b in product(range(self.prime), range(self.prime)):
                data.append([a, 97, b, 98, (a // b) % self.prime])

        self.data = torch.tensor(data)

        # split the data into train and test
        generator = torch.Generator().manual_seed(42)
        train_data, test_data = torch.utils.data.random_split(
            self.data, [0.5, 0.5], generator=generator
        )
        self.train_dataloader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collator,
        )
        self.test_dataloader = DataLoader(
            test_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collator,
        )

        return self.train_dataloader, self.test_dataloader

    def collator(self, data: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Mirror a HuggingFace collator."""
        return {
            "input_ids": torch.stack([d[:-1] for d in data]),
            "labels": torch.stack([[-100, -100, -100, d[-1]] for d in data]),
            "attention_mask": torch.ones(len(data), self.max_tokens),
        }


class TaskFactory:
    @staticmethod
    def get_task(config: RunConfig) -> AbstractTask:
        if config.task == "esm_mlm":
            return ESMMLMTask(config)
        elif config.task == "grokking":
            return GrokkingTask(config)

    @property
    def task_names(self) -> list[str]:
        return ["esm_mlm", "grokking"]
