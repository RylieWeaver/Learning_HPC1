# General
import math
from typing import Optional

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange

# Learning HPC1
from learning_hpc.distributed import ParallelState, Out_TPLinear, In_TPLinear, AttnGather, _F_Gather_B_ReduceScatter
from learning_hpc.utils.config import Config



class Attention(nn.Module):
    def __init__(self, dim, num_heads, parallel_state):
        super().__init__()
        # Read/check
        self.sp_group = parallel_state.sp_group
        self.sp_size = parallel_state.sp_size
        self.sp_rank = parallel_state.sp_rank
        self.tp_group = parallel_state.tp_group
        self.tp_size = parallel_state.tp_size
        self.tp_rank = parallel_state.tp_rank
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert num_heads % self.tp_size == 0, "num_heads must be divisible by tensor_par_size"
        self.num_heads = num_heads
        self.num_heads_local = num_heads // self.tp_size
        self.head_dim = dim // num_heads

        # Modules
        self.qkv = Out_TPLinear(dim, 3*dim, parallel_state, gather_output=False)
        self.out_proj = In_TPLinear(self.num_heads * self.head_dim, dim, parallel_state)

    def causal_mask(self, attn_logits, device):                                             # attn_logits: [B, S_sub, S, H_local]
        # Setup
        B, S_sub, S, H_local = attn_logits.size()
        start_idx = self.sp_rank * S_sub
        end_idx = (self.sp_rank + 1) * S_sub
        q_pos = torch.arange(start_idx, end_idx, device=device)[:, None]                    # [S_sub, 1]
        k_pos = torch.arange(S, device=device)[None, :]                                     # [1, S]

        # Make and apply mask
        mask = (k_pos > q_pos)[None, :, :, None]                                            # [S_sub, S] --> [1, S_sub, S, 1] (expanded for broadcasting)
        attn_logits = attn_logits.masked_fill(mask, float("-inf"))
        return attn_logits                                                                  # [B, S_sub, S, H_local]

    def forward(self, x):                                                                   # [B, S_sub, D]
        # Setup
        B, S_sub, D = x.size()

        # Compute local QKV
        qkv = self.qkv(x)                                                                   # [B, S_sub, 3*D // tp_size]
        qkv = qkv.view(B, S_sub, self.num_heads_local, 3 * self.head_dim)                   # [B, S_sub, H_local, 3*D_head]
        q, k, v = qkv.split(self.head_dim, dim=-1)                                          # [B, S_sub, H_local, D_head] (each)

        # Collect global K,V
        if self.sp_size > 1:
            k = _F_Gather_B_ReduceScatter.apply(k, self.sp_group, 1)                        # [B, S, H_local, D_head]
            v = _F_Gather_B_ReduceScatter.apply(v, self.sp_group, 1)                        # [B, S, H_local, D_head]

        # Compute attention scores
        # NOTE: This is no longer O(S^2) memory!!! Instead, it is O(S^2 // sp_size)
        attn_logits = torch.einsum("bshd,bShd->bsSh", q, k) / math.sqrt(self.head_dim)      # [B, S_sub, S, H_local]
        attn_logits = self.causal_mask(attn_logits, x.device)                               # [B, S_sub, S, H_local]
        attn_weights = F.softmax(attn_logits, dim=2)                                        # [B, S_sub, S, H_local]

        # Compute value add
        attn_scaled_values = torch.einsum("bsSh,bShd->bshd", attn_weights, v)               # [B, S_sub, H_local, D_head]

        # Concat heads and output projection
        value = rearrange(attn_scaled_values, "b s h d -> b s (h d)")                       # [B, S_sub, D // tp_size]
        value = self.out_proj(value)                                                        # [B, S_sub, D // tp_size]
        return value                                                                        # [B, S_sub, D]


class TPMLP(nn.Module):
    def __init__(self, dim, parallel_state):
        super().__init__()
        self.dim = dim
        self.hidden_dim = 4 * dim
        self.tp_group = parallel_state.tp_group
        self.tp_size = parallel_state.tp_size
        self.tp_rank = parallel_state.tp_rank
        self.fc1 = Out_TPLinear(dim, self.hidden_dim, parallel_state=parallel_state, gather_output=False)
        self.fc2 = In_TPLinear(self.hidden_dim, dim, parallel_state=parallel_state)

    def forward(self, x):       # [B, S_sub, D]
        x = self.fc1(x)         # [B, S_sub, 4*D // tp_size]
        x = F.silu(x)           # [B, S_sub, 4*D // tp_size]
        x = self.fc2(x)         # [B, S_sub, D]
        return x                # [B, S_sub, D]


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, parallel_state):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, parallel_state)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = TPMLP(dim, parallel_state)

    def forward(self, x):                   # [B, S_sub, D]
        h = self.ln1(x)                     # [B, S_sub, D]
        attn = self.attn(h)                 # [B, S_sub, D]
        x = x + attn                        # [B, S_sub, D]
        x = x + self.mlp(self.ln2(x))       # [B, S_sub, D]
        return x                            # [B, S_sub, D]


class DNATransformerConfig(Config):
    def __init__(
        self,
        vocab_size: int = 4,
        max_seq_len: int = 1024,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "dim": self.dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }


class DNATransformer(nn.Module):
    def __init__(self, cfg: DNATransformerConfig, parallel_state: Optional[ParallelState] = None):
        super().__init__()
        # Read
        self.cfg = cfg
        vocab_size = cfg.vocab_size
        max_seq_len = cfg.max_seq_len
        dim = cfg.dim
        num_heads = cfg.num_heads
        num_layers = cfg.num_layers
        # NOTE: the class ParallelState just holds non-distributed parallelism info by default
        parallel_state = parallel_state if parallel_state is not None else ParallelState()
        self.sp_group = parallel_state.sp_group
        self.sp_size = parallel_state.sp_size
        self.sp_rank = parallel_state.sp_rank
        self.tp_group = parallel_state.tp_group
        self.tp_size = parallel_state.tp_size
        self.tp_rank = parallel_state.tp_rank

        # Modules
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, parallel_state=parallel_state)
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, vocab_size),  # NOTE: This is not tensor-parallelized because vocab_size is usually small
        )

    def forward(self, x):
        # Setup
        B, S = x.shape

        # Get sp-aware idx
        if self.sp_size > 1:
            S_sub = S // self.sp_size
            seq_offset = self.sp_rank * (S // self.sp_size)
        else:
            S_sub = S
            seq_offset = 0
        start_idx = seq_offset
        end_idx = seq_offset + S_sub

        # Apply sp-aware idx to tokens and positional embeddings
        tokens = x[:, start_idx:end_idx]                                # [B, S_sub]
        positions = (                                                   # [1, S_sub] --> [B, S_sub]
            torch.arange(start_idx, end_idx, device=tokens.device)
            .unsqueeze(0)
            .expand(B, S_sub)
        )

        # Transformer blocks
        x = self.token_emb(tokens) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.lm_head(x)
        return x
