# General
from typing import Union, Optional

# Torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Learning HPC1



def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_rank0(group: Union[dist.ProcessGroup, None] = None) -> bool:
    if not is_dist():
        return True
    if group is None:
        return dist.get_rank() == 0
    return dist.get_rank(group) == 0

def rank0_print(*args, **kwargs):
    if is_rank0():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

def rank0_write(path, message: str, mode: str = "a"):
    if is_rank0():
        with path.open(mode) as f:
            f.write(f"{message}")

def unwrap_model(model):
    if is_dist() and isinstance(model, DDP):
        return model.module
    return model

@torch.no_grad()
def reduce_scalar(x, device: torch.device, group: dist.ProcessGroup, dtype=torch.float32, average=True) -> float:
    if not is_dist():
        return float(x)
    
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=dtype, device=device).detach()
    
    dist.barrier()
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    if average:
        x /= dist.get_world_size(group)
    return float(x.item())
