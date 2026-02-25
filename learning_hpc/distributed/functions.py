# General
import math

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import ReduceOp

# Learning HPC1
from .pt_functions import _AllReduce



"""
Torch autograd functions give us a way to define custom forward/backward passes
that align with our distributed operations. 

Some important things to know are:
- The backward pass must return the same number of outputs as the forward pass 
  inputs (other than ctx). For non-tensors that don't require grads (e.g., group, dim), 
  we just end up returning a bunch of corresponding None values.
"""

class _F_Identity_B_AllReduce(Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        # NOTE: Might want to clone the tensor here but not sure
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _AllReduce.apply(ReduceOp.SUM, ctx.group, grad_output)
        return (grad_output, None)


class _F_Gather_B_Split(Function):
    """
    This is usually used after a linear layer which shards the input. In this case,
    each shard of the parallelized linear layer produces a shard of the output. In some
    cases, those outputs will then be gathered in the forward. If the outputs are gathered,
    then this function will receive the entire gradient in the backward pass, which must 
    be split to give only the gradient wrt the sharded parameters.

    Forward:    [B, S, D_out // tp_size]  -->  [B, S, D_out]             (sharded output --> gathered output)
    Backward:   [B, S, D_out]             -->  [B, S, D_out // tp_size]  (sharded gradient --> gathered gradient)
    """
    @staticmethod
    def forward(ctx, x_shard, group, dim: int = -1):
        ctx.group = group
        ctx.dim = dim
        ctx.rank = dist.get_rank(group=group)
        ctx.world = dist.get_world_size(group=group)

        # Gather and concatenate shards
        shards = [torch.empty_like(x_shard) for _ in range(ctx.world)]
        dist.all_gather(shards, x_shard.contiguous(), group=group)
        return torch.cat(shards, dim=dim)

    @staticmethod
    def backward(ctx, grad):                                        # [B, S, D]
        grad_shards = torch.chunk(grad, ctx.world, dim=ctx.dim)     # list of [B, S, D//world]
        grad_shard = grad_shards[ctx.rank].contiguous()             # [B, S, D//world]
        return grad_shard, None, None


class _F_AllReduce_B_Identity(Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        return _AllReduce.apply(ReduceOp.SUM, group, tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


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



"""
By default, weights are initialized by the shape of local weight matrix, whereas it should be the global shape.
To fix this, just initialize manually with the in/out size (e.g. Xavier init here). 

Note this is only sometimes a problem. Some initializations depend on only the input size, some on only the size,
and some on both. If the initialization depends only on input size and CPLinear is used, then there'll be no difference
since input size is not sharded. To be safe, it's best to just always do manual initialization for tensor parallelism.
"""

class Out_TPLinear(nn.Module):
    """
    Out_TPLinear: Output Tensor-Parallel Linear Layer
    Splits the output features across tensor parallel ranks.
    """
    def __init__(self, in_features, out_features, parallel_state, gather_output=True):
        super().__init__()
        # Read/check
        self.tp_group = parallel_state.tp_group
        self.tp_size = dist.get_world_size(group=self.tp_group)
        assert out_features % self.tp_size == 0, \
            f"out_features {out_features} must be divisible by tensor_par_size {self.tp_size} for Output Tensor-Parallelism"
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.out_features_per_partition = out_features // self.tp_size  # NOTE: This is the important part

        # Define
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition))

        # Store something for later functions to know which params are TP-sharded
        self.sharded_param_names = ("weight", "bias")
        
        # Initialize (local fan in and global fan in are the same)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

        # # Initialize with global in/out dimensions
        # a = math.sqrt(6.0 / (in_features + out_features))
        # nn.init.uniform_(self.weight, -a, a)
        
    def forward(self, x):                                                       # [B, S, D_in]
        # This sums the grads for backpropagation
        if self.tp_size > 1:
            x = _F_Identity_B_AllReduce.apply(x, self.tp_group)                 # [B, S, D_in]
        # Apply the sharded Linear
        out = F.linear(x, self.weight, self.bias)                               # [B, S, D_out // tp_size]
        # Gather output with its backward pass
        if self.gather_output and self.tp_size > 1:
            out = _F_Gather_B_Split.apply(out, self.tp_group, 2)            # [B, S, D_out]
        
        return out


class In_TPLinear(nn.Module):
    """
    In_TPLinear: Input Tensor-Parallel Linear Layer
    Splits the input features across tensor parallel ranks.
    """
    def __init__(self, in_features, out_features, parallel_state):
        super().__init__()
        # Read/check
        self.tp_group = parallel_state.tp_group
        self.tp_size = dist.get_world_size(group=self.tp_group)
        assert in_features % self.tp_size == 0, \
            f"in_features {in_features} must be divisible by tensor_par_size {self.tp_size} for Input Tensor-Parallelism"
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_partition = in_features // self.tp_size  # NOTE: This is the important part

        # Define
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features_per_partition))
        self.bias = nn.Parameter(torch.zeros(self.out_features))

        # Store something for later functions to know which params are TP-sharded (only weight here)
        self.sharded_param_names = ("weight",)
        
        # Initialize with global in/out dimensions
        a = math.sqrt(6.0 / (in_features + out_features))
        nn.init.uniform_(self.weight, -a, a)

    def forward(self, x):                                                       # [B, S, D_in]
        # Apply the sharded Linear
        out = F.linear(x, self.weight, bias=None)                               # [B, S, D_out // tp_size]
        # Sum the sharded outputs across TP ranks
        out = _F_AllReduce_B_Identity.apply(out, self.tp_group)                 # [B, S, D_out]
        # Add bias once
        out = out + self.bias
        return out


class AttnGather(nn.Module):
    """
    AttnGather: Used to all-gather Key/Value tensors for attention computation.
    By using local Q and global K/V, attention has memory-scaling of O(S^2 // sp_size).
    """
    def __init__(self, parallel_state, dim: int = 1):
        super().__init__()
        # Read/check
        self.sp_group = parallel_state.sp_group
        self.dim = dim

    def forward(self, x):                                                       # [B, S_sub, D]
        # Gather the sharded sequence dimension
        x = _F_Gather_B_ReduceScatter.apply(x, self.sp_group, self.dim)         # [B, S, D]
        return x
