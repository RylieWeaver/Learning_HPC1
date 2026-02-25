# General
import math
import argparse

# Torch
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Learning HPC1
from learning_hpc.model import DNATransformerConfig, DNATransformer
from learning_hpc.distributed import init_parallel_state, rank0_print, check_param_sync, resolve_device, ParallelState
from learning_hpc.utils import set_all_random_seeds



# Commands:
# - torchrun --standalone --nproc_per_node=4 test_dpsp.py  --data_parallel_size 2  --sequence_parallel_size 2
# - nohup torchrun --standalone --nproc_per_node=4 test_dpsp.py  --data_parallel_size 2  --sequence_parallel_size 2 > output.txt 2>&1 &
# - pkill -u "$(whoami)" -f 'test_dpsp.py'


if __name__ == "__main__":
    # Setup (just reading args here for flexibility in calling the script)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_parallel_size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Sequence parallel size")
    args = parser.parse_args()
    dp_size = args.data_parallel_size
    sp_size = args.sequence_parallel_size

    # Distributed setup
    parallel_state = init_parallel_state(
        dp_size=dp_size,
        sp_size=sp_size,
    )
    device = resolve_device(parallel_state=parallel_state)
    set_all_random_seeds(42)

    # Show distributed setup
    message = f"""
    [rank {parallel_state.rank}] Initialized distributed process with:
        - dp_rank: {parallel_state.dp_rank}
        - sp_rank: {parallel_state.sp_rank}
        - data parallel size: {parallel_state.dp_size}
        - sequence parallel size: {parallel_state.sp_size}
        - world size: {parallel_state.world_size}
    """
    print(message)

    # Init model/optimizer
    vocab_size = V = 4  # DNA has 4 nucleotides: [A, C, G, T]
    model_cfg = DNATransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=10,
        dim=4,        # NOTE: This must be divisible by num_heads
        num_heads=1,
        num_layers=1,
    )
    model = DNATransformer(model_cfg).to(device)
    dist_model = DNATransformer(model_cfg, parallel_state).to(device)
    dist_model.load_state_dict(model.state_dict())  # Ensure that the models start with the same parameters (for testing purposes)
    dist_model = DDP(dist_model, device_ids=[parallel_state.local_rank], process_group=parallel_state.world_group)
    opt = torch.optim.Adam(model.parameters(), lr=3e-5)
    dist_opt = torch.optim.Adam(dist_model.parameters(), lr=3e-5)

    # Check that parameters are synchronized
    check_param_sync(dist_model, parallel_state.world_group)

    # Create Data
    B = 8
    S = 8
    input_ids = torch.randint(
        0, V, (B, S), dtype=torch.long, device=device           # [B, S]
    )
    # Minibatch splitting
    b = (B // parallel_state.dp_size) if B % parallel_state.dp_size == 0 else (B // parallel_state.dp_size + 1)
    mb_start_idx = b * parallel_state.dp_rank
    mb_end_idx = min(mb_start_idx + b, B)
    mb_input_ids = input_ids.clone()[mb_start_idx:mb_end_idx]   # [B/DP, S] = [b, S]
    # NOTE: We don't split into subsequences for sequence
    # parallelism here (done within the model itself)

    # Do one forward/backward pass on non-distributed model
    model.train()
    opt.zero_grad()
    logits, labels = model(input_ids)                           # [B, S-1, V]
    logits = logits.reshape(-1, V)                              # [B*S-1, V]
    labels = labels.reshape(-1)                                 # [B*S-1]
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    loss.backward()

    # Do one forward/backward pass on distributed model
    dist_model.train()
    dist_opt.zero_grad()
    dist_logits, dist_labels = dist_model(mb_input_ids)         # [b, s, V] or [b, s-1, V] (s = subsequence length)
    dist_logits = dist_logits.reshape(-1, V)                    # [b*s-1, V]
    dist_labels = dist_labels.reshape(-1)                       # [b*s-1]
    dist_loss_fn = nn.CrossEntropyLoss()
    # Minibatch subsequence loss
    """
    NOTE: (feel free to ignore the loss scaling here)
    
    DDP averages the grad over its parallel group, which
    is actually not mathematically equivalant to the non-distributed
    case because some ranks may have slightly more or less data.
    Normally, this is just glossed over and makes minimal impact, but
    since we're testing for exact equivalence here, we need to account
    for this difference by cancelling the DDP group (which is over 
    world_size) and doing our own proportional scaling based on the 
    amount of data.
    """
    proportion = (dist_labels.numel() / labels.numel())
    dist_loss = dist_loss_fn(dist_logits, dist_labels) * proportion * parallel_state.world_size
    dist_loss.backward()  # After loss backward the grads will have been averaged over dpsp_group
    rank0_print(f"[rank {parallel_state.rank}] Loss: {loss.item()} | Dist Loss: {dist_loss.item()}")

    # Examine One Parameter Value/Grad for Both Models:
    name, param = list(model.named_parameters())[0]
    dist_name, dist_param = list(dist_model.module.named_parameters())[0]
    message = f"""
    [rank {parallel_state.rank} | dp_rank: {parallel_state.dp_rank} | sp_rank: {parallel_state.sp_rank}] Parameter: {name}
        - Value: {param.detach()}
        - Dist Value: {dist_param.detach()}
        - Values Close: {torch.allclose(param.detach(), dist_param.detach(), atol=1e-3)}
        - Grad: {param.grad}
        - Dist Grad: {dist_param.grad}
        - Grads Close: {torch.allclose(param.grad, dist_param.grad, atol=1e-3)}
    """
    rank0_print(message)

    # Carry out grad
    opt.step()
    dist_opt.step()

    # Check param sync again
    check_param_sync(dist_model, parallel_state.world_group)

    # Examine One Parameter for Both Models (should be the same):
    name, param = list(model.named_parameters())[0]
    dist_name, dist_param = list(dist_model.module.named_parameters())[0]
    message = f"""
    [rank {parallel_state.rank} | dp_rank: {parallel_state.dp_rank} | sp_rank: {parallel_state.sp_rank}] Parameter: {name}
        - Value: {param.detach()}
        - Dist Value: {dist_param.detach()}
        - Values Close: {torch.allclose(param.detach(), dist_param.detach(), atol=1e-3)}
    """
    rank0_print(message)

    # Cleanup
    dist.destroy_process_group()
