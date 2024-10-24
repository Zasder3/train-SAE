import random

try:
    import pyfastx
except ImportError:
    pyfastx = None
from torch.utils.data import Dataset
from transformers import PretrainedTokenizer


class FastaDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: PretrainedTokenizer,
        max_len: int = 1022,
        world_size: int = 1,
        rank: int = 0,
    ):
        if pyfastx is None:
            raise ImportError("pyfastx is required to use FastaDataset")
        self.fasta = pyfastx.Fasta(path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.world_size = world_size
        self.rank = rank

    def __len__(self):
        return len(self.fasta) // self.world_size

    def __getitem__(self, idx):
        idx = idx * self.world_size + self.rank
        seq = self.fasta[idx].seq
        if len(seq) > self.max_len:
            start = random.randint(0, len(seq) - self.max_len)
            seq = seq[start : start + self.max_len]
        return self.tokenizer(seq, return_tensors="pt")
