import argparse
import torch
import torch.distributed as dist
from torch.autograd import Function

from learning_hpc.distributed import init_parallel_state, is_rank0


class MyFnNoCtx(Function):
    @staticmethod
    def forward(ctx, x):
        return x ** 2

    @staticmethod
    def backward(ctx, g):
        return 7 * g


def print_tp_ordered(parallel_state, msg: str):
    tp_group = parallel_state.tp_group
    tp_rank  = dist.get_rank(tp_group)
    tp_size  = dist.get_world_size(tp_group)
    for r in range(tp_size):
        dist.barrier(tp_group)
        if tp_rank == r:
            print(f"[tp_rank {r}] {msg}", flush=True)
        dist.barrier(tp_group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    args = parser.parse_args()

    parallel_state = init_parallel_state(dp_size=1, sp_size=1, tp_size=args.tensor_parallel_size, model_type="DDP")
    torch.cuda.set_device(parallel_state.local_rank)
    device = torch.device(f"cuda:{parallel_state.local_rank}")

    # same x on all TP ranks
    x = torch.empty((), device=device)
    if is_rank0():
        x = torch.tensor(3.0, device=device)
        print(f"Initial x: {x.item()}")
    dist.broadcast(x, src=0, group=parallel_state.tp_group)

    # Good: apply()
    x1 = x.clone().requires_grad_(True)
    y1 = MyFnNoCtx.apply(x1)
    y1.backward()
    print_tp_ordered(parallel_state, f"apply:   x={x1.item()} grad={x1.grad.item()}")  # 2.0

    # Bad: forward() directly (no autograd node -> no backward rule)
    x2 = x.clone().requires_grad_(True)
    y2 = MyFnNoCtx.forward(None, x2)   # bypasses apply()
    y2.backward()
    print_tp_ordered(parallel_state, f"forward: x={x2.item()} grad={x2.grad.item()}")  # 2.0 from mul, NOT from your custom backward

    dist.destroy_process_group()
