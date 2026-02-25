# General
import os
import warnings
from dataclasses import dataclass
from typing import Optional

# Torch
import torch
import torch.distributed as dist

# Learning HPC1
from .utils import rank0_print



@dataclass(frozen=True)
class ParallelState:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "none"
    model_type: str = "default"

    dp_size: int = 1
    sp_size: int = 1
    tp_size: int = 1

    dp_rank: int = 0
    sp_rank: int = 0
    tp_rank: int = 0

    dp_group: Optional[dist.ProcessGroup] = None
    sp_group: Optional[dist.ProcessGroup] = None
    tp_group: Optional[dist.ProcessGroup] = None
    dpsp_group: Optional[dist.ProcessGroup] = None
    sptp_group: Optional[dist.ProcessGroup] = None


def rank2coords(rank: int, dp_size: int, sp_size: int, tp_size: int) -> tuple[int, int, int]:
    tp_rank = rank % tp_size
    sp_rank = (rank // tp_size) % sp_size
    dp_rank = rank // (tp_size * sp_size)
    return dp_rank, sp_rank, tp_rank

def coords2rank(dp_rank: int, sp_rank: int, tp_rank: int, dp_size: int, sp_size: int, tp_size: int) -> int:
    return (dp_rank * sp_size * tp_size) + (sp_rank * tp_size) + tp_rank


def build_groups(dp_size: int, sp_size: int, tp_size: int):
    """
    Groups must be defined for all processes. Group membership can be determined
    by when the idx for all the fixed parallelism dimensions match the current process's
    coords.
    """
    # Setup
    rank = dist.get_rank()  # NOTE: This could be passed but is just read from ground-truth dist
    dp_rank, sp_rank, tp_rank = rank2coords(rank, dp_size, sp_size, tp_size)
    my_tp_group = my_sp_group = my_dp_group = my_dpsp_group = my_sptp_group = None

    # TP groups: one for each (dp_rank, sp_rank), varying tp_rank
    for d in range(dp_size):
        for s in range(sp_size):
            ranks = [coords2rank(d, s, t, dp_size, sp_size, tp_size) for t in range(tp_size)]
            g = dist.new_group(ranks)
            # Match group by fixed dp and sp coords
            if d == dp_rank and s == sp_rank:
                my_tp_group = g

    # SP groups: one for each (dp_rank, tp_rank), varying sp_rank
    for d in range(dp_size):
        for t in range(tp_size):
            ranks = [coords2rank(d, s, t, dp_size, sp_size, tp_size) for s in range(sp_size)]
            g = dist.new_group(ranks)
            # Match group by fixed dp and tp coords
            if d == dp_rank and t == tp_rank:
                my_sp_group = g

    # DP groups: one for each (sp_rank, tp_rank), varying dp_rank
    for s in range(sp_size):
        for t in range(tp_size):
            ranks = [coords2rank(d, s, t, dp_size, sp_size, tp_size) for d in range(dp_size)]
            g = dist.new_group(ranks)
            # Match group by fixed sp and tp coords
            if s == sp_rank and t == tp_rank:
                my_dp_group = g

    # DP-SP group: one for each tp_rank, varying (dp_rank, sp_rank)
    # NOTE: This is used for loss/gradient all-reduce
    for t in range(tp_size):
        ranks = [coords2rank(d, s, t, dp_size, sp_size, tp_size) for d in range(dp_size) for s in range(sp_size)]
        g = dist.new_group(ranks)
        # Match group by fixed tp coords
        if t == tp_rank:
            my_dpsp_group = g

    # SP-TP group: one for each dp_rank, varying (sp_rank, tp_rank)
    # NOTE: This can be used to broadcast data across SP and TP dims
    for d in range(dp_size):
        ranks = [coords2rank(d, s, t, dp_size, sp_size, tp_size) for s in range(sp_size) for t in range(tp_size)]
        g = dist.new_group(ranks)
        # Match group by fixed dp coords
        if d == dp_rank:
            my_sptp_group = g

    # Check to make sure the process found its groups
    assert my_dp_group and my_sp_group and my_tp_group and my_dpsp_group and my_sptp_group, "Failed to build process groups"
    return (dp_rank, sp_rank, tp_rank), (my_dp_group, my_sp_group, my_tp_group, my_dpsp_group, my_sptp_group)


def init_parallel_state(
    dp_size: int,
    sp_size: int,
    tp_size: int,
    model_type: str = "default",
    silence_warnings_nonzero_rank: bool = True,
) -> ParallelState:
    # Early return for non-distributed
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return ParallelState()
    
    # Check parallelism sizes
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == dp_size * sp_size * tp_size, "world_size must equal dp_size * sp_size * tp_size"

    # Initialize process group
    backend = "nccl" # NOTE: May need to be expanded for other hardware systems
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://", device_id=local_rank)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if silence_warnings_nonzero_rank and rank != 0:
        warnings.filterwarnings("ignore")

    (dp_rank, sp_rank, tp_rank), (dp_group, sp_group, tp_group, dpsp_group, sptp_group) = build_groups(dp_size, sp_size, tp_size)

    rank0_print(
        f"[ParallelState] world_size={world_size} dp={dp_size} sp={sp_size} tp={tp_size} backend={backend}"
    )

    return ParallelState(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        dp_size=dp_size,
        sp_size=sp_size,
        tp_size=tp_size,
        dp_rank=dp_rank,
        sp_rank=sp_rank,
        tp_rank=tp_rank,
        dp_group=dp_group,
        sp_group=sp_group,
        tp_group=tp_group,
        dpsp_group=dpsp_group,
        sptp_group=sptp_group,
        backend=backend,
        model_type=model_type,
    )
