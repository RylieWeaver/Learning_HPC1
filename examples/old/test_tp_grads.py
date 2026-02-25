# General
import argparse
from pathlib import Path

# Torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Learning HPC1
from learning_hpc.train import TrainerConfig, Trainer
from learning_hpc.model import DNATransformerConfig, DNATransformer
from learning_hpc.data import DNADataset, create_random_dna_string
from learning_hpc.distributed import init_parallel_state, is_rank0, rank0_print
from learning_hpc.utils import Config



# Commands:
# - torchrun --standalone --nproc_per_node=8 test_tp_grads.py  --data_parallel_size 2 --sequence_parallel_size 2 --tensor_parallel_size 2
# - nohup torchrun --standalone --nproc_per_node=8 test_tp_grads.py  --data_parallel_size 2 --sequence_parallel_size 2 --tensor_parallel_size 2 > output.txt 2>&1 &
# - pkill -u "$(whoami)" -f 'test_tp_grads.py'


# Notes:
# - We forego a lot of things for simplicity here, including train/val/test splits, pathing, and
#   model/optimizer hyperparameters. The focus here is on 3D distributed training.


import math
import torch.nn as nn
import torch.nn.functional as F
from learning_hpc.distributed import gather_parallel_group
class CPLinear(nn.Module):
    """
    CPLinear: Column-Parallel Linear Layer
    Splits the output features across tensor parallel ranks.
    They are concatenated after the linear transformation.
    """
    def __init__(self, in_features, out_features, parallel_state, bias=True, gather_output=True):
        super().__init__()
        # Read/check
        self.tp_group = parallel_state.tp_group
        self.tp_size = parallel_state.tp_size
        assert out_features % self.tp_size == 0, "out_features must be divisible by tensor_par_size for Column Parallelism"
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.out_features_per_partition = out_features // self.tp_size # NOTE: This is the important part

        # Define
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition)) if bias else None
        
        # Xavier-uniform that takes into account the tensor-partitioning
        a = math.sqrt(6.0 / (in_features + out_features))
        nn.init.uniform_(self.weight, -a, a)

    def forward(self, x):                                                       # [B, S_sub, D_in]
        out = F.linear(x, self.weight, self.bias)                               # [B, S_sub, D_out // tp_size]
        out = gather_parallel_group(                                            # [B, S_sub, D_out]
            out, self.tp_group, self.tp_size, 
            dim=-1, gather_output=self.gather_output
        )
        """
        In the interest of clarity, this is what gather_tp_group does here:
        if self.gather_output and self.tp_size > 1:
            gathered = [torch.empty_like(out) for _ in range(self.tp_size)]     # [B, S_sub, D_out // tp_size] (tp_size times)
            dist.all_gather(gathered, out, group=self.tp_group)                 # [B, S_sub, D_out // tp_size] (tp_size times) (but filled in now)
            out = torch.cat(gathered, dim=-1)                                   # [B, S_sub, D_out]
        """
        return out


class ToyModelConfig(Config):
    def __init__(self):
        super().__init__()
        self.hidden_size = 5
        self.output_size = 2


class ToyModel(torch.nn.Module):
    def __init__(self, config: ToyModelConfig, parallel_state=None):
        super().__init__()
        self.cfg = config
        self.embedding = nn.Embedding(num_embeddings=4, embedding_dim=self.cfg.hidden_size)
        self.linear = CPLinear(self.cfg.hidden_size, self.cfg.output_size, parallel_state=parallel_state)
    
    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)



if __name__ == "__main__":
    # Setup (just reading args here for flexibility in calling the script)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_parallel_size", type=int, default=2, help="Data parallel size")
    parser.add_argument("--sequence_parallel_size", type=int, default=2, help="Sequence parallel size")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size")
    args = parser.parse_args()
    dp_size = args.data_parallel_size
    sp_size = args.sequence_parallel_size
    tp_size = args.tensor_parallel_size

    # Distributed setup
    parallel_state = init_parallel_state(
        dp_size=dp_size,
        sp_size=sp_size,
        tp_size=tp_size,
        model_type="DDP",  # NOTE: Make sure this matches how we wrap the model below!!!
    )
    dp_rank = parallel_state.dp_rank
    sp_rank = parallel_state.sp_rank
    tp_rank = parallel_state.tp_rank
    torch.cuda.set_device(parallel_state.local_rank)
    device = torch.device(f"cuda:{parallel_state.local_rank}")

    # Init distributed model/optimizer
    # model_cfg = ToyModelConfig()
    # model = ToyModel(model_cfg, parallel_state=parallel_state).to(device)

    model_cfg = DNATransformerConfig(
        vocab_size=4,  # DNA has nucleotides: [A, C, G, T]
        max_seq_len=60,
        dim=32,
        num_heads=8,  # NOTE: This must be divisible by tp_size
        num_layers=6,
    )
    model = DNATransformer(model_cfg, parallel_state).to(device)

    model = DDP(model, process_group=parallel_state.dpsp_group, device_ids=[device])  # Alternative DDP wrapping can be uncommented
    # model = FSDP(model, process_group=parallel_state.dpsp_group, device_id=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Do one forward pass
    model.train()
    input_data = torch.tensor([[0, 0, 0, 1, 1, 2] * (10)], device=device)  # Batch size 1. Make sure each class has a different number of appearances
    optimizer.zero_grad()
    output = model(input_data)  # [1, chunk_size, output_size]
    loss = output.mean()  # Local loss
    loss.backward()  # After loss backward the grads will have been averaged over dpsp_group

    # Examine Grads:
    ## 1) Make sure the grads are the same across all dpsp_ranks (scatter mean and assert close)
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Gather grads from all dpsp_ranks
            gathered_grads = [torch.empty_like(param.grad) for _ in range(parallel_state.dp_size * parallel_state.sp_size)]
            dist.all_gather(gathered_grads, param.grad, group=parallel_state.dpsp_group)
            # Compute mean grad across dpsp_ranks
            mean_grad = torch.stack(gathered_grads, dim=0).mean(dim=0)
            # Assert all gathered grads are close to mean grad
            for g in gathered_grads:
                assert torch.allclose(g, mean_grad, atol=1e-6), \
                f"Grad mismatch in param {name} on tp_rank {tp_rank}. g: {g}, mean_grad: {mean_grad}"
    
    # ## 2) Write and print the grads for all tp_ranks
    # if dp_rank == 0 and sp_rank == 0:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad and param.grad is not None:
    #             print(f"Param: {name}, TP Rank: {tp_rank}, Grad: {param.grad}")

    # Cleanup
    dist.destroy_process_group()
