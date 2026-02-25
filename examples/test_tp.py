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
# - torchrun --standalone --nproc_per_node=2 test_tp.py --tensor_parallel_size 2
# - nohup torchrun --standalone --nproc_per_node=2 test_tp.py  --tensor_parallel_size 2 > output.txt 2>&1 &
# - pkill -u "$(whoami)" -f 'test_tp.py'



if __name__ == "__main__":
    # Setup (just reading args here for flexibility in calling the script)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_parallel_size", type=int, default=2, help="Sequence parallel size")
    parser.add_argument
    parser.add_argument("--data_parallel_size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--lrearning_rate", type=float, default=3e-5, help="Learning rate")
    args = parser.parse_args()
    tp_size = args.tensor_parallel_size

    # Distributed setup
    parallel_state = init_parallel_state(
        dp_size=1,
        sp_size=sp_size,
        tp_size=1,
    )
    print(f"[rank {dist.get_rank()}] sp_group ranks: {dist.get_world_size(group=parallel_state.sp_group)}", flush=True)
    dp_rank = parallel_state.dp_rank
    sp_rank = parallel_state.sp_rank
    tp_rank = parallel_state.tp_rank
    torch.cuda.set_device(parallel_state.local_rank)
    device = torch.device(f"cuda:{parallel_state.local_rank}")


    # Init model/optimizer
    model_cfg = DNATransformerConfig(
        vocab_size=4,  # DNA has nucleotides: [A, C, G, T]
        max_seq_len=chunk_size,
        dim=model_dim,
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
