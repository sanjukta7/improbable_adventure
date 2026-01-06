"""
Local Encoder and Decoder modules for MergeDNA.

The Local Encoder performs:
1. DNA base embedding
2. Positional encoding
3. Local context processing via transformer
4. Token merging via bipartite soft matching

The Local Decoder performs:
1. Token unmerging using source map
2. Refinement via transformer layer
3. Projection back to feature space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .merging import bipartite_soft_matching


class LocalEncoder(nn.Module):
    """
    Local Encoder: Embeds DNA bases and performs token merging.

    Maps N bases -> L tokens (L < N) using bipartite soft matching.
    Tracks merging via a source map for later unmerging.

    As per the MergeDNA paper:
    - Stacks differentiable token merging blocks with local-window constraints
    - Learns to segment DNA into variable-length units based on local structure
    - Allocates shorter tokens to dense information regions, longer to repetitive
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int = 4,
        merge_ratio: float = 0.5,
        max_seq_len: int = 2048,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            dim: Model dimension
            vocab_size: Number of DNA bases (4: A, C, G, T)
            merge_ratio: Target ratio of tokens to keep (0.5 = reduce by half)
            max_seq_len: Maximum sequence length for positional embeddings
            num_heads: Number of attention heads for local context
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.merge_ratio = merge_ratio

        # DNA base embedding
        self.embed = nn.Embedding(vocab_size, dim)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        # Local attention for context before merging
        # This captures local patterns before deciding what to merge
        self.local_attn = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=min(num_heads, dim),  # Ensure heads don't exceed dim
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )

        # Metric projection for similarity computation in bipartite matching
        self.metric_proj = nn.Linear(dim, max(dim // 4, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the local encoder.

        Args:
            x: Input tensor of shape [B, N] (indices) or [B, N, vocab_size] (one-hot)

        Returns:
            x_merged: Merged tokens of shape [B, L, D] where L < N
            source_map: Ownership map [B, N] pointing each original token to its merged index
        """
        # Handle one-hot input
        if x.dim() == 3:
            x_idx = x.argmax(dim=-1)
        else:
            x_idx = x

        B, N = x_idx.shape
        device = x_idx.device

        # 1. Embed and add positional encoding
        x = self.embed(x_idx)  # [B, N, D]
        x = x + self.pos_embed[:, :N, :]

        # 2. Apply local attention for context
        x = self.local_attn(x)

        # 3. Determine number of merges
        # merge_ratio=0.5 means we want to keep 50% of tokens
        # So we remove (1 - merge_ratio) * N tokens
        r = int(N * (1 - self.merge_ratio))
        # Cannot merge more than 50% in one bipartite pass
        r = min(r, N // 2)

        # 4. Compute metrics and find matches via bipartite matching
        metric = self.metric_proj(x)
        node_indices, top_b_indices = bipartite_soft_matching(metric, r)

        # If no merges needed (sequence too short)
        if r == 0 or node_indices is None:
            source_map = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            return x, source_map

        # 5. Execute the merge
        # Global index of the Source (B) tokens being merged (odd indices)
        src_idx = top_b_indices * 2 + 1

        # Global index of the Destination (A) tokens they merge into
        dst_local = torch.gather(node_indices, 1, top_b_indices)
        dst_idx = dst_local * 2  # Even indices

        # Get features of source tokens
        src_features = torch.gather(
            x, 1, src_idx.unsqueeze(-1).expand(-1, -1, self.dim)
        )

        # Add source features to destination (merge by averaging)
        x_out = x.clone()
        x_out.scatter_add_(
            1, dst_idx.unsqueeze(-1).expand(-1, -1, self.dim), src_features
        )

        # Normalize merged tokens (divide by count)
        counts = torch.ones(B, N, 1, device=device)
        counts.scatter_add_(
            1,
            dst_idx.unsqueeze(-1),
            torch.ones(B, r, 1, device=device)
        )
        x_out = x_out / counts

        # 6. Build ownership map and remove merged tokens
        # Initially, everyone owns themselves
        ownership = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).clone()

        # Mark merged tokens as owned by their destination
        ownership.scatter_(1, src_idx, dst_idx)

        # Create mask of kept tokens
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, src_idx, False)

        # Extract kept tokens
        flat_features = x_out[mask].view(B, N - r, self.dim)

        # Renumber source map to new indices
        new_indices = torch.cumsum(mask.long(), dim=1) - 1
        source_map = torch.gather(new_indices, 1, ownership)

        return flat_features, source_map


class LocalDecoder(nn.Module):
    """
    Local Decoder: Unmerges tokens back to original length and refines.

    Maps L tokens -> N bases using the source map (ownership matrix).
    Applies refinement attention after unmerging.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            dim: Model dimension
            num_heads: Number of attention heads for refinement
            dropout: Dropout rate
        """
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
        """
        Forward pass through the local decoder.

        Args:
            z_l: Merged tokens of shape [B, L, D]
            source_map: Ownership map [B, N] (indices into L)

        Returns:
            x_hat: Reconstructed features of shape [B, N, D]
        """
        B, L, D = z_l.shape
        N = source_map.shape[1]

        # Unmerge: broadcast L tokens back to N positions using source_map
        indices = source_map.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
        x_unmerged = torch.gather(z_l, 1, indices)  # [B, N, D]

        # Apply refinement attention
        x_refined = self.refine(x_unmerged)

        # Final projection
        return self.proj(x_refined)
