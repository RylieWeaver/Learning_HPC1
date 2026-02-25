# General

# Torch
import torch
import torch.nn as nn
import torch.distributed as dist

# Learning HPC1
from .utils import rank0_print



@torch.no_grad()
def check_param_sync(
    model: nn.Module,
    group: dist.ProcessGroup,
    eps: float = 1e-3,
) -> bool:
    """
    Check that all model params are synced across the world.
    """
    # Setup
    check = True
    mistaken_params = set()

    # Get modules
    modules = dict(model.named_modules())

    # Iterate check through modules
    for name, module in modules.items():
        for param_name, param in module.named_parameters(recurse=False):
            # Gather all params across the world
            param_list = [torch.empty_like(param) for _ in range(dist.get_world_size(group))]
            dist.all_gather(param_list, param.contiguous(), group=group)

            # Check that all params are close to each other
            for other_param in param_list:
                if not torch.allclose(param, other_param, atol=eps):
                    check = False
                    mistaken_params.add(f"{name}.{param_name}")
                    break
    # Show
    if check:
        rank0_print("[check_param_sync] All model parameters are synced!")
    else:
        rank0_print(
            f"[check_param_sync] Found {len(mistaken_params)} mistaken parameters (not synced): {mistaken_params}"
        )
    return check
