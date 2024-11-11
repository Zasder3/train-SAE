import random

try:
    import pyfastx
except ImportError:
    pyfastx = None
from typing import Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class FastaDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 1022,
        indices: Optional[list[int]] = None,
        world_size: int = 1,
        rank: int = 0,
    ):
        if pyfastx is None:
            raise ImportError("pyfastx is required to use FastaDataset")

        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(pyfastx.Fasta(path))))
        self.fasta = pyfastx.Fasta(path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.world_size = world_size
        self.rank = rank

    def __len__(self):
        return len(self.indices) // self.world_size

    def __getitem__(self, idx: int, max_len: Optional[int] = None):
        max_len = max_len or self.max_len

        idx = idx * self.world_size + self.rank
        seq = self.fasta[self.indices[idx]].seq
        if len(seq) > max_len:
            start = random.randint(0, len(seq) - max_len)
            seq = seq[start : start + max_len]
        return self.tokenizer(seq)
