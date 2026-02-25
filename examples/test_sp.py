# General
import math
import argparse

# Torch
import torch
import torch.nn as nn
import torch.distributed as dist

# Learning HPC1
from learning_hpc.model import DNATransformerConfig, DNATransformer
from learning_hpc.distributed import init_parallel_state, is_rank0, Out_TPLinear, In_TPLinear, tp_broadcast_params, tp_examine_params, rank0_print
from learning_hpc.utils import set_all_random_seeds



# Commands:
# - torchrun --standalone --nproc_per_node=2 test_sp.py  --sequence_parallel_size 2
# - nohup torchrun --standalone --nproc_per_node=2 test_sp.py  --sequence_parallel_size 2 > output.txt 2>&1 &
# - pkill -u "$(whoami)" -f 'test_sp.py'


if __name__ == "__main__":
    # Setup (just reading args here for flexibility in calling the script)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_parallel_size", type=int, default=2, help="Sequence parallel size")
    args = parser.parse_args()
    sp_size = args.sequence_parallel_size

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
    set_all_random_seeds(42 + tp_rank)  # This seed is chosen explicitly
    vocab_size = 4  # DNA has 4 nucleotides: [A, C, G, T]
    model_cfg = DNATransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=10,
        dim=4,        # NOTE: This must be divisible by num_heads
        num_heads=1,  # NOTE: This must be divisible by tp_size
        num_layers=1,
    )
    model = DNATransformer(model_cfg, parallel_state).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Create Data (should be the same on all ranks)
    B = 1
    S = 4
    input_data = torch.empty(B, S, dtype=torch.long, device=device)     # [B, S]
    labels = torch.empty(B, S, dtype=torch.long, device=device)      # [B, S]
    if is_rank0():
        input_data = torch.randint(0, vocab_size, (B, S), dtype=torch.long, device=device)
        labels = torch.randint(0, vocab_size, (B, S), dtype=torch.long, device=device)
    dist.broadcast(input_data, src=0, group=parallel_state.sp_group)  # Make sure all ranks have the same input
    dist.broadcast(labels, src=0, group=parallel_state.sp_group)      # Make sure all ranks have the same labels

    # Do one forward/backward pass so that we can inspect grads
    model.train()
    optimizer.zero_grad()
    output, labels = model(input_data, labels)      # [1, S, output_size]
    output = output.view(-1, vocab_size)            # [B*S, output_size]
    labels = labels.view(-1)                        # [B*S]
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, labels) / sp_size  # Local loss
    loss.backward()  # After loss backward the grads will have been averaged over dpsp_group

    # Examine Grads:
    for name, param in model.named_parameters():
        print(f"{name}: \n\tValue: {param.detach()}\n\tGrad: {param.grad}\n")

    # Cleanup
    dist.destroy_process_group()
