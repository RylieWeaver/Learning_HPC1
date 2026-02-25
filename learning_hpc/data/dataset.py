# General
import random

# Torch
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

# Learning HPC1
from learning_hpc.distributed import ParallelState, rank0_write, is_rank0



def create_random_dna_string(path, n_bases=1_000_000, seed=42):
    """
    This file should easily be small enough to load even with 1M bases.

    Make sure that n_bases is larger than the chunk_size you planso that you
    can sample large chunks without worrying about padding! (explained below)
    It's not the focus of this repo to deal with padding, so we keep it simple.
    """
    # Create
    rng = random.Random(seed)
    bases = ["A", "C", "G", "T"]
    seq = "".join(rng.choices(bases, k=n_bases))
    # Save/Return
    rank0_write(path, seq, mode="w")
    return seq


class DNADataset(IterableDataset):
    def __init__(self, path, chunk_size=8192, seed=42, parallel_state=None):
        with open(path, "r", encoding="utf-8") as handle:
            self.dna_string = handle.read().strip().upper()
        self.dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.chunk_size = chunk_size
        self.seed = seed
        self.parallel_state = parallel_state if parallel_state else ParallelState()
    
    def _encode(self, sequence):
        return [self.dna_vocab.get(ch) for ch in sequence]

    def __iter__(self):
        """
        Note that the parametrization of start_idx having a maximum of len(seq) - chunk_size
        ensures that we always have enough sequence left to return a full chunk (i.e. we don't 
        need to worry about padding). A production repo would worry about padding, but this
        makes things simpler for learning purposes. The sneaky error that could arise is if
        your chunk size is larger than your sequence length, but the default dataset sequence
        length is 100M which should be big enough.
        """
        worker = torch.utils.data.get_worker_info()
        base_seed = self.seed + (worker.id if worker is not None else 0)
        rng = random.Random(base_seed)
        while True:
            if (self.parallel_state.sp_size == 1 or is_rank0(self.parallel_state.sp_group)):
                if self.chunk_size > len(self.dna_string):
                    raise ValueError(f"Chunk size {self.chunk_size} must be smaller than dataset sequence length {len(self.dna_string)}")
                start_idx = rng.randint(0, len(self.dna_string) - self.chunk_size)
                chunk = self.dna_string[start_idx : start_idx + self.chunk_size]
                chunk = self._encode(chunk)  # Str -> List[int]
                yield torch.tensor(chunk, dtype=torch.long)
            else:
                yield torch.empty(self.chunk_size, dtype=torch.long)
