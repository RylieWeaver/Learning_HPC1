# General

# Torch
import torch
import torch.nn as nn
import torch.distributed as dist

# Learning HPC1



@torch.no_grad()
def sync_params(
    model: nn.Module,
    parallel_state,
    src: int = 0,
) -> None:
    """
    Sync params across 3D parallelism dimensions
      - DP: Sync all params across data parallel group.
      - SP: Sync all params across sequence parallel group.
      - TP: Only sync replicated params (e.g. norm gamma, beta),
            not sharded ones (e.g. weights).
    """
    dpsp_group = getattr(parallel_state, "dpsp_group", None)
    tp_group = getattr(parallel_state, "tp_group", None)

    for module in model.modules():
        # NOTE: 'sharded_param_names' lets us know which params not to sync across TP
        sharded_param_names = set(getattr(module, "sharded_param_names", ()))
        for pname, p in module.named_parameters(recurse=False):
            if p is None:
                continue

            # Sync across SP/DP
            if dpsp_group is not None:
                dpsp_src = dist.get_global_rank(dpsp_group, 0)
                dist.broadcast(p.data, src=dpsp_src, group=dpsp_group)

            # Sync across TP only if this param is replicated in TP
            if tp_group is not None and pname not in sharded_param_names:
                tp_src = dist.get_global_rank(tp_group, 0)
                dist.broadcast(p.data, src=tp_src, group=tp_group)


@torch.no_grad()
def check_param_sync(
    model: nn.Module,
    parallel_state,
    check_dp: bool = True,
    check_sp: bool = True,
    check_tp: bool = True,
    eps: float = 1e-3,
    check_tp_diff: bool = True,
    tp_diff_eps: float = 1e-6,
) -> bool:
    """
    Check for param should be sync across 3D parallelism dimensions
      - DP: Sometimes params are not synced (e.g. norm params) (option to skip DP check).
      - SP: Sometimes params are not synced (e.g. norm params) (option to skip SP check).
      - TP: Option to skip TP check.

    Note that for TP, replicated params are expected to be synced and unreplicated (sharded) 
    params are expected to be different, so we have the option to check both.
    """
    # Setup
    check = True
    mistaken_params = set()
    dp_group = getattr(parallel_state, "dp_group", None)
    sp_group = getattr(parallel_state, "sp_group", None)
    tp_group = getattr(parallel_state, "tp_group", None)

    # Get modules
    modules = dict(model.named_modules())

    # Iterate check through modules
    for name, module in modules.items():
        # NOTE: 'sharded_param_names' lets us know which params shouldn't be synced across TP
        sharded_param_names = getattr(module, "sharded_param_names", ())
        # Iterate through params
        for pname, p in module.named_parameters(recurse=False):
            if p is None:
                continue
            param_check = True
            ref = p.data.detach().clone()
            # Check DP
            if check_dp and dp_group is not None:
                # Broadcast from group rank 0, compare, and check if any mismatch via reduce-max (rather than check all-to-all)
                group_src = dist.get_global_rank(dp_group, 0)
                dist.broadcast(ref, src=group_src, group=dp_group)
                mismatch = torch.tensor(
                    [0 if torch.allclose(p.data, ref, atol=eps, rtol=0) else 1],
                    device=p.device,
                    dtype=torch.int32,
                )
                dist.all_reduce(mismatch, op=dist.ReduceOp.MAX, group=dp_group)
                param_check = param_check and (mismatch.item() == 0)
                dist.barrier(group=dp_group)  # Sync before next param check
            # Check SP
            if check_sp and sp_group is not None:
                # Broadcast from group rank 0, compare, and check if any mismatch via reduce-max (rather than check all-to-all)
                group_src = dist.get_global_rank(sp_group, 0)
                dist.broadcast(ref, src=group_src, group=sp_group)
                mismatch = torch.tensor(
                    [0 if torch.allclose(p.data, ref, atol=eps, rtol=0) else 1],
                    device=p.device,
                    dtype=torch.int32,
                )
                dist.all_reduce(mismatch, op=dist.ReduceOp.MAX, group=sp_group)
                param_check = param_check and (mismatch.item() == 0)
                dist.barrier(group=sp_group)  # Sync before next param check
            # Check TP
            if check_tp and tp_group is not None:
                if pname not in sharded_param_names:
                    # Broadcast from group rank 0, compare, and check if any mismatch via reduce-max (rather than check all-to-all)
                    group_src = dist.get_global_rank(tp_group, 0)
                    dist.broadcast(ref, src=group_src, group=tp_group)
                    mismatch = torch.tensor(
                        [0 if torch.allclose(p.data, ref, atol=eps, rtol=0) else 1],
                        device=p.device,
                        dtype=torch.int32,
                    )
                    dist.all_reduce(mismatch, op=dist.ReduceOp.MAX, group=tp_group)
                    param_check = param_check and (mismatch.item() == 0)
                    dist.barrier(group=tp_group)  # Sync before next param check
                elif check_tp_diff:
                    # Broadcast from group rank 0, compare, and check if ALL MATCH via reduce-max (rather than check all-to-all)
                    group_src = dist.get_global_rank(tp_group, 0)
                    dist.broadcast(ref, src=group_src, group=tp_group)
                    match = torch.tensor(
                        [1 if torch.allclose(p.data, ref, atol=tp_diff_eps, rtol=0) else 0],
                        device=p.device,
                        dtype=torch.int32,
                    )
                    dist.all_reduce(match, op=dist.ReduceOp.MAX, group=tp_group)
                    param_check = param_check and (match.item() == 1)
                    dist.barrier(group=tp_group)  # Sync before next param check
            # Overall check
            check = check and param_check
            if not param_check:
                mistaken_params.add(f"{name}.{pname}")
            # Print values for mistaken params
            for mp in mistaken_params:
                if mp == f"{name}.{pname}":
                    print(f"[Rank {dist.get_rank()}] MISMATCH in param: {mp}, value: {p.data.flatten()}")
    return check, mistaken_params
