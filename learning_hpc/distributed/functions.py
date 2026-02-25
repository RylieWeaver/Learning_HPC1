# General
import math

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function

# Learning HPC1



"""
Torch autograd functions give us a way to define custom forward/backward passes
that align with our distributed operations. 

Some important things to know are:
- The backward pass must return the same number of outputs as the forward pass 
  inputs (other than ctx). For non-tensors that don't require grads (e.g., group, dim), 
  we just end up returning a bunch of corresponding None values.
"""


class _F_Gather_B_ReduceScatter(Function):
    @staticmethod
    def forward(ctx, x_shard, group, dim: int = 1):
        ctx.group = group
        ctx.dim = dim
        ctx.rank = dist.get_rank(group=group)
        ctx.world = dist.get_world_size(group=group)

        shards = [torch.empty_like(x_shard) for _ in range(ctx.world)]      # [B, S // sp_size, D]
        dist.all_gather(shards, x_shard.contiguous(), group=group)          # list of [B, S // sp_size, D]
        return torch.cat(shards, dim=ctx.dim)                               # [B, S, D]

    @staticmethod
    def backward(ctx, grad_full):                                           # [B, S, D]
        grad_chunks = list(
            torch.chunk(grad_full.contiguous(), ctx.world, dim=ctx.dim)     # list of [B, S // sp_size, D]
        )
        grad_shard = torch.empty_like(grad_chunks[0])
        dist.reduce_scatter(
            grad_shard, grad_chunks, op=dist.ReduceOp.SUM, group=ctx.group  # [B, S // sp_size, D]
        )
        return grad_shard, None, None
