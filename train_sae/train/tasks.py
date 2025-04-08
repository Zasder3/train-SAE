from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from train_sae.configs.base import RunConfig
from train_sae.train.datasets.fasta import FastaDataset


class AbstractTask(ABC):
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
class GrokkingTask(AbstractTask): ...


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
