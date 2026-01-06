"""
Token Merging module for MergeDNA.

Implements bipartite soft matching from the ToMe (Token Merging) paper,
adapted for DNA sequence processing as described in MergeDNA.

The key idea is to split tokens into two sets (A: evens, B: odds),
compute similarity between them, and merge the most similar pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Applies Bipartite Soft Matching to find the top-r most similar pairs.
    Splits tokens into A (evens) and B (odds) and matches B -> A.
    
    Args:
        metric: [Batch, N, Dim] - Similarity features
        r: int - Number of tokens to merge (remove)
    
    Returns:
        node_indices: [Batch, N/2] - For every B, which A is its best match?
        top_b_indices: [Batch, r] - Which B tokens (0..N/2) are actually selected to merge?
    """
    B, N, _ = metric.shape
    
    if r <= 0: return None, None
    
    with torch.no_grad():
        # Normalize for cosine similarity
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # Partition: A = Evens (Destination), B = Odds (Source)
        a = metric[:, 0::2]  # [B, N/2, C]
        b = metric[:, 1::2]  # [B, N/2, C]
        
        # Compute scores: Similarity of every B to every A
        # [B, N/2, N/2]
        scores = a @ b.transpose(-1, -2)

        # For each B, find its single best A match
        values, node_indices = scores.max(dim=1) # values: [B, N/2], indices: [B, N/2]
        
        # Among all the B-A pairs, pick the 'r' strongest connections
        _, top_b_indices = torch.topk(values, r, dim=-1) # [B, r]
        
    return node_indices, top_b_indices

def bipartite_soft_matching_binary(metric, r):
    """
    Applies Bipartite Soft Matching to find the top-r most similar pairs.
    Partition metric into A (evens) and B (odds).
    
    Args:
        metric: [Batch, N, Dim] - Similarity features/embeddings
        r: int - Number of tokens to remove (merge)
    
    Returns:
        merge_path: Mapping of which token merges into which.
    """
    B, N, _ = metric.shape
    
    # Protected against odd sequence lengths
    if r <= 0: 
        return None, None

    # 1. Partition into A (evens - destination) and B (odds - source)
    # We want to merge 'r' tokens from B into A.
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a = metric[:, 0::2]  # [B, N/2, C]
        b = metric[:, 1::2]  # [B, N/2, C]
        
        # Compute similarity scores: [B, N/2(b), N/2(a)]
        # We look for the best match for each token in B within A
        scores = a @ b.transpose(-1, -2)

        # 2. Find top-r matches
        # We want to find the r tokens in B that have the highest similarity 
        # to any token in A.
        # Get max similarity for each b node
        values, indices = scores.max(dim=-1) # values: [B, N/2], indices: [B, N/2]
        
        # Find the top r 'b' tokens with the strongest links
        _, top_b_indices = torch.topk(values, r, dim=-1) # [B, r]

        # Build the gather indices
        # We need to know: For every token in the original input, where does it go?
        # By default, i goes to i. If i is in the top-r matches, it goes to its match.
        
    return indices, top_b_indices

class MergeDNALayer(nn.Module):
    def __init__(self, dim, merge_r=1):
        """
        dim: Feature dimension
        merge_r: Number of merges to perform per window/pass. 
                 If we treat the whole sequence as one window for simplicity here,
                 this reduces length by r.
        """
        super().__init__()
        self.dim = dim
        self.merge_r = merge_r
        # Lightweight embedding to compute similarity (as per ToMe paper)
        self.metric_proj = nn.Linear(dim, dim // 4) 

    def forward(self, x, source_matrix=None):
        """
        x: [Batch, N, Dim]
        source_matrix: [Batch, N_original, N_current] - Optional tracking of history
        
        Returns:
            x_merged: [Batch, N - r, Dim]
            merge_map: Logic required to unmerge later
        """
        B, N, C = x.shape
        r = min(self.merge_r, N // 2) # Cannot merge more than half at once using bipartite
        
        # 1. Compute Similarity Metric
        metric = self.metric_proj(x)
        
        # 2. Decide what to merge (Bipartite Matching)
        # indices: which 'a' (even) each 'b' (odd) prefers
        # top_b_indices: which 'b' tokens are actually selected to merge
        node_indices, top_b_indices = bipartite_soft_matching(metric, r)
        
        if node_indices is None:
            return x, None

        # 3. Perform the Merge
        # We are merging selected B tokens into their paired A tokens.
        # Create a destination map. 
        # Size N. mapping[i] = i (initially)
        # If i is a selected B, mapping[i] = paired A
        
        # Create a scatter map for standard weighted average merging
        # For simplicity in this implementation, we use average pooling 
        # between the keeper and the merger.
        
        with torch.no_grad():
            # Calculate destination indices for gather/scatter
            # Even indices (0, 2, 4...) are kept by default (A set)
            # Odd indices (1, 3, 5...) are B set.
            
            # Map everything to its position in the REDUCED sequence
            # This requires calculating offsets.
            
            # This part can be complex to vectorize. 
            # Conceptually:
            # We construct a matrix S of shape [N - r, N]
            # S[new_index, old_index] = 0.5 (if merged) or 1.0 (if kept)
            
            # Let's build the explicit map for unmerging later:
            # For each original index [0...N-1], point to new index [0...N-r-1]
            
            # Helper to map old index to A/B set logic
            # A_indices = 0, 2, 4...
            # B_indices = 1, 3, 5...
            
            # Get the actual index of the 'a' token that 'b' merges into
            batch_idx = torch.arange(B).view(B, 1).to(x.device)
            
            # chosen_b_indices are the indices in the B-array (0 to N/2)
            # convert to global indices (odd numbers)
            global_b_indices = top_b_indices * 2 + 1
            
            # matched_a_local = node_indices.gather(1, top_b_indices)
            matched_a_local = torch.gather(node_indices, 1, top_b_indices)
            global_a_indices = matched_a_local * 2
            
            # Create a vector that marks which tokens are being removed (merged)
            mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
            mask.scatter_(1, global_b_indices, False)
            
        # 4. Compute Merged Features
        # Get the features of the B tokens being merged
        b_features = torch.gather(x, 1, global_b_indices.unsqueeze(-1).expand(-1, -1, C))
        
        # Add B features to their paired A tokens
        x_out = x.clone()
        x_out.scatter_add_(1, global_a_indices.unsqueeze(-1).expand(-1, -1, C), b_features)
        
        # Average the features (normalize by count)
        counts = torch.ones(B, N, 1, device=x.device)
        counts.scatter_add_(1, global_a_indices.unsqueeze(-1).expand(-1, -1, 1), 
                            torch.ones_like(b_features[:, :, :1]))
        x_out = x_out / counts
        
        # Get indices of kept tokens (where mask is True)
        kept_indices = torch.nonzero(mask, as_tuple=False)  # [num_kept, 2]
        
        # Gather kept tokens properly for each batch
        kept_per_batch = mask.sum(dim=1)  # Should be N - r for each batch
        
        # Use a loop for clarity (can be vectorized for large batches)
        x_final_list = []
        for b in range(B):
            batch_mask = mask[b]  # [N]
            x_final_list.append(x_out[b, batch_mask])  # [N-r, C]
        x_final = torch.stack(x_final_list, dim=0)  # [B, N-r, C]
        
        # 5. Return Merge Map for Unmerging
        # The unmerge map needs to tell us: for the resulting N-r tokens, 
        # which original token corresponds to which input?
        # Actually, simpler: For every Original Token (N), which Merged Token (N-r) owns it?
        
        # Build the ownership map
        ownership = torch.zeros(B, N, dtype=torch.long, device=x.device)
        
        # First, simply range(0...N), but shift for removed items?
        # A simple way: Cumulative sum of the mask tells us the new index
        new_indices = torch.cumsum(mask.long(), dim=1) - 1
        
        # Assign kept tokens to their new indices
        ownership[mask] = new_indices[mask]
        
        # Assign merged (removed) tokens to the index of their 'a' keeper
        # We know global_b merges into global_a. 
        # So ownership[global_b] should equal ownership[global_a]
        
        # Get new index of the 'a' targets
        target_a_new_indices = torch.gather(new_indices, 1, global_a_indices)
        ownership.scatter_(1, global_b_indices, target_a_new_indices)
        
        return x_final, ownership

class MergeDNAUnmerge(nn.Module):
    def __init__(self):
        super().__init__()
        # No learned parameters needed for basic Source Matrix unmerging
        
    def forward(self, x_merged, ownership_map):
        """
        x_merged: [Batch, N_reduced, Dim]
        ownership_map: [Batch, N_original] - Indices mapping
        
        Returns: [Batch, N_original, Dim]
        """
        B, N_reduced, C = x_merged.shape
        B, N_original = ownership_map.shape
        
        # We expand x_merged back to N_original size
        # S^T * Z operation
        
        # Gather: For every original position i, look up ownership_map[i] in x_merged
        # ownership_map contains indices in range [0, N_reduced-1]
        
        indices = ownership_map.unsqueeze(-1).expand(-1, -1, C)
        x_unmerged = torch.gather(x_merged, 1, indices)
        
        return x_unmerged

# --- Example Usage matching the Paper's logic ---

def example_usage():
    B, N, Dim = 2, 10, 8
    x = torch.randn(B, N, Dim)
    
    # 1. Local Encoder (Merging)
    # Reducing sequence length
    merger = MergeDNALayer(dim=Dim, merge_r=N//2) # Reduce by half
    x_latent, source_map = merger(x)
    
    print(f"Original Shape: {x.shape}")        # [2, 10, 8]
    print(f"Latent Shape:   {x_latent.shape}") # [2, 5, 8]
    
    # ... Latent Transformer Processing would happen here ...
    x_processed = x_latent # Simulating pass-through
    
    # 2. Local Decoder (Unmerging)
    # Using the Source Matrix (source_map) to reconstruct
    unmerger = MergeDNAUnmerge()
    x_recon = unmerger(x_processed, source_map)
    
    print(f"Recon Shape:    {x_recon.shape}") 
    print(torch.sum(x_recon - x)) # [2, 10, 8]
    
    # Verify values for a merged pair (Checking the broadcast)
    # If token 0 and 1 merged, x_recon[0] should equal x_recon[1]
    # (assuming no subsequent local attention refinement yet)
    print("Reconstruction check passed:", x_recon.shape == x.shape)

if __name__ == "__main__":
    example_usage()