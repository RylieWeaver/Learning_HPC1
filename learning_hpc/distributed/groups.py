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

    dp_size: int = 1
    sp_size: int = 1

    dp_rank: int = 0
    sp_rank: int = 0

    dp_group: Optional[dist.ProcessGroup] = None
    sp_group: Optional[dist.ProcessGroup] = None
    world_group: Optional[dist.ProcessGroup] = None
    device: Optional[torch.device] = None


def rank2coords(rank: int, dp_size: int, sp_size: int) -> tuple[int, int]:
    sp_rank = (rank % sp_size)
    dp_rank = rank // (sp_size)
    return dp_rank, sp_rank

def coords2rank(dp_rank: int, sp_rank: int, dp_size: int, sp_size: int) -> int:
    return (dp_rank * sp_size) + sp_rank

def build_groups(dp_size: int, sp_size: int):
    """
    Groups must be defined for all processes. Group membership can be determined
    by when the idx for all the fixed parallelism dimensions match the current process's
    coords.
    """
    # Setup
    rank = dist.get_rank()  # NOTE: This could be passed but is just read from ground-truth dist
    dp_rank, sp_rank = rank2coords(rank, dp_size, sp_size)
    my_sp_group = my_dp_group = my_world_group = None

    # SP groups: one for each (dp_rank), varying sp_rank
    for d in range(dp_size):
        ranks = [coords2rank(d, s, dp_size, sp_size) for s in range(sp_size)]
        g = dist.new_group(ranks)
        # Match group by fixed dp coords
        if d == dp_rank:
            my_sp_group = g

    # DP groups: one for each (sp_rank), varying dp_rank
    for s in range(sp_size):
        ranks = [coords2rank(d, s, dp_size, sp_size) for d in range(dp_size)]
        g = dist.new_group(ranks)
        # Match group by fixed sp coords
        if s == sp_rank:
            my_dp_group = g

    # World group (all ranks)
    ranks = list(range(dp_size * sp_size))
    my_world_group = dist.new_group(ranks)

    # Check to make sure the process found its groups
    assert my_dp_group and my_sp_group and my_world_group, "Failed to build process groups"
    return (dp_rank, sp_rank), (my_dp_group, my_sp_group, my_world_group)


def init_parallel_state(
    master_addr: Optional[str] = None,
    master_port: Optional[int] = None,
    dp_size: int = 1,
    sp_size: int = 1,
    silence_warnings_nonzero_rank: bool = True,
) -> ParallelState:
    # Read args
    if master_addr is not None:
        os.environ['MASTER_ADDR'] = str(master_addr)
    if master_port is not None: 
        os.environ['MASTER_PORT'] = str(master_port)
    backend = "nccl"

    # Assign canonically named env variables if SLURM
    ## NOTE: Torchrun and other launchers will set these variables automatically
    if "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = str(os.environ["SLURM_NTASKS"])
        os.environ["RANK"] = str(os.environ["SLURM_PROCID"])

    # Early return for non-distributed
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return ParallelState()
    if dp_size == 1 and sp_size == 1:
        return ParallelState()
    
    # Configure world
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    num_gpus_per_node = torch.cuda.device_count()
    local_rank = int(rank) % int(num_gpus_per_node)
    assert world_size == dp_size * sp_size, "world_size must equal dp_size * sp_size"

    # Set local rank device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            #init_method=f"tcp://{args.master_addr}:{args.master_port}",
            init_method='env://',
            rank=rank,
            world_size=world_size,
            device_id=local_rank,
        )

    # Set env variables given the rank
    if silence_warnings_nonzero_rank and rank != 0:
        warnings.filterwarnings("ignore")

    (dp_rank, sp_rank), (dp_group, sp_group, world_group) = build_groups(dp_size, sp_size)

    rank0_print(
        f"[ParallelState] world_size={world_size} dp={dp_size} sp={sp_size} backend={backend}"
    )

    return ParallelState(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        dp_size=dp_size,
        sp_size=sp_size,
        dp_rank=dp_rank,
        sp_rank=sp_rank,
        dp_group=dp_group,
        sp_group=sp_group,
        world_group=world_group,
        backend=backend,
        device=device,
    )
