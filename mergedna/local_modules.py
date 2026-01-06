import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .merging import bipartite_soft_matching


class LocalEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        vocab_size: int = 4,
        merge_ratio: float = 0.5,
        max_seq_len: int = 2048,
        num_heads: int = 4,
        dropout: float = 0.1
    ):

        super().__init__()
        self.dim = dim
        self.merge_ratio = merge_ratio

        # DNA base embedding
        self.embed = nn.Embedding(vocab_size, dim)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        self.local_attn = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=min(num_heads, dim), 
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  
        )

        self.metric_proj = nn.Linear(dim, max(dim // 4, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:


        if x.dim() == 3:
            x_idx = x.argmax(dim=-1)
        else:
            x_idx = x

        B, N = x_idx.shape
        device = x_idx.device

        x = self.embed(x_idx)  # [B, N, D]
        x = x + self.pos_embed[:, :N, :]

        x = self.local_attn(x)

        r = int(N * (1 - self.merge_ratio))
        r = min(r, N // 2)

        metric = self.metric_proj(x)
        node_indices, top_b_indices = bipartite_soft_matching(metric, r)

        if r == 0 or node_indices is None:
            source_map = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            return x, source_map

        src_idx = top_b_indices * 2 + 1

        dst_local = torch.gather(node_indices, 1, top_b_indices)
        dst_idx = dst_local * 2  

        src_features = torch.gather(
            x, 1, src_idx.unsqueeze(-1).expand(-1, -1, self.dim)
        )

        x_out = x.clone()
        x_out.scatter_add_(
            1, dst_idx.unsqueeze(-1).expand(-1, -1, self.dim), src_features
        )

        # Normalize merged
        counts = torch.ones(B, N, 1, device=device)
        counts.scatter_add_(
            1,
            dst_idx.unsqueeze(-1),
            torch.ones(B, r, 1, device=device)
        )
        x_out = x_out / counts

        ownership = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).clone()

        ownership.scatter_(1, src_idx, dst_idx)

        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, src_idx, False)

        flat_features = x_out[mask].view(B, N - r, self.dim)

        new_indices = torch.cumsum(mask.long(), dim=1) - 1
        source_map = torch.gather(new_indices, 1, ownership)

        return flat_features, source_map


class LocalDecoder(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        dropout: float = 0.1
    ):

        super().__init__()
        self.dim = dim

        # Refinement layer after unmerging
        self.refine = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=min(num_heads, dim),
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        # Final projection
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        z_l: torch.Tensor,
        source_map: torch.Tensor
    ) -> torch.Tensor:

        B, L, D = z_l.shape
        N = source_map.shape[1]

        # Unmerge: broadcast L tokens back to N positions using source_map
        indices = source_map.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
        x_unmerged = torch.gather(z_l, 1, indices)  # [B, N, D]

        # Apply refinement attention
        x_refined = self.refine(x_unmerged)

        return self.proj(x_refined)
