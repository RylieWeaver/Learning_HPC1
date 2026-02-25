# General
import math
import argparse

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._functional_collectives import all_gather_tensor_autograd

# Learning HPC1
from learning_hpc.distributed import init_parallel_state, is_rank0, rank0_print, gather_parallel_group



# Commands:
# - torchrun --standalone --nproc_per_node=2 new_all_gather.py  --tensor_parallel_size 2
# - nohup torchrun --standalone --nproc_per_node=2 new_all_gather.py  --tensor_parallel_size 2 > output.txt 2>&1 &
# - pkill -u "$(whoami)" -f 'new_all_gather.py'


# Notes:
# - We forego a lot of things for simplicity here, including train/val/test splits, pathing, and
#   model/optimizer hyperparameters. The focus here is on 3D distributed training.


# def set_all_random_seeds(seed: int):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     random.seed(seed)
#     np.random.seed(seed)


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
        self.out_features_per_partition = out_features // self.tp_size  # NOTE: This is the important part

        # Define
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition)) if bias else None
        
        # Xavier-uniform that takes into account the tensor-partitioning
        a = math.sqrt(6.0 / (in_features + out_features))
        nn.init.uniform_(self.weight, -a, a)

    def forward(self, x):                                                       # [B, S, D_in]
        # This sums the grads for backpropagation
        if self.tp_size > 1:
            x = _F_Identity_B_AllReduce.apply(self.tp_group, x)                 # [B, S, D_in]
        # Apply the sharded Linear
        out = F.linear(x, self.weight, self.bias)                               # [B, S, D_out // tp_size]
        # Gather output with its backward pass
        if self.gather_output:
            out = _F_Gather_B_Split.apply(out, self.tp_group, 2)               # [B, S, D_out]
        
        return out



if __name__ == "__main__":
    # Setup (just reading args here for flexibility in calling the script)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size")
    args = parser.parse_args()
    tp_size = args.tensor_parallel_size

    # Distributed setup
    # set_all_random_seeds(42)  # This should make the linear layer weights the same on all ranks
    parallel_state = init_parallel_state(
        dp_size=1,
        sp_size=1,
        tp_size=tp_size,
        model_type="DDP",  # NOTE: Make sure this matches how we wrap the model below!!!
    )
    dp_rank = parallel_state.dp_rank
    sp_rank = parallel_state.sp_rank
    tp_rank = parallel_state.tp_rank
    torch.cuda.set_device(parallel_state.local_rank)
    device = torch.device(f"cuda:{parallel_state.local_rank}")

    # Init model/optimizer
    input_dim = 4
    hidden_dim = 4
    output_dim = 4
    # model = nn.Sequential(
    #     CPLinear(in_features=input_dim, out_features=hidden_dim, parallel_state=parallel_state),
    #     nn.SiLU(),
    #     CPLinear(in_features=hidden_dim, out_features=output_dim, parallel_state=parallel_state),
    # ).to(device)
    model = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=hidden_dim),
        CPLinear(in_features=hidden_dim, out_features=hidden_dim, parallel_state=parallel_state),
        nn.Linear(in_features=hidden_dim, out_features=output_dim)
    ).to(device)
    # Broadcast model params that are not split across tensor parallel
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, CPLinear):
                continue
            for p in module.parameters(recurse=False):
                dist.broadcast(p, src=0, group=parallel_state.tp_group)
    model = DDP(model, process_group=parallel_state.dpsp_group, device_ids=[device])
    # model = FSDP(model, process_group=parallel_state.dpsp_group, device_id=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Do one forward pass
    model.train()
    B = 8
    S = 10
    input_data = torch.empty(B, S, input_dim, device=device)  # [B, S, D_in]
    labels = torch.empty(B, S, output_dim, device=device)  # [B, S, D_out]
    if is_rank0():
        input_data = torch.randn(B, S, input_dim).to(device)
        labels = torch.randn(B, S, output_dim).to(device)
    dist.broadcast(input_data, src=0, group=parallel_state.tp_group)  # Make sure all ranks have the same input
    dist.broadcast(labels, src=0, group=parallel_state.tp_group)      # Make sure all ranks have the same labels
    # rank0_print(f"Input data: {input_data}")
    optimizer.zero_grad()
    output = model(input_data)  # [1, S, output_size]
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, labels) / tp_size  # Local loss
    loss.backward()  # After loss backward the grads will have been averaged over dpsp_group

    # Examine Grads:
    for name, param in model.named_parameters():
        print(f"{name}: \n\tValue: {param.detach()}\n\tGrad: {param.grad}\n")
        dist.barrier()

    # Cleanup
    dist.destroy_process_group()
