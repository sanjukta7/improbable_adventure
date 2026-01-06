
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    B, N, _ = metric.shape
    
    if r <= 0: return None, None
    
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        

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

    B, N, _ = metric.shape
    
    if r <= 0: 
        return None, None

    # to merge 'r' tokens from B into A.
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a = metric[:, 0::2]  # [B, N/2, C]
        b = metric[:, 1::2]  # [B, N/2, C]
        
        # Compute similarity scores: [B, N/2(b), N/2(a)]
        scores = a @ b.transpose(-1, -2)


        values, indices = scores.max(dim=-1) # values: [B, N/2], indices: [B, N/2]
        
        # Find the top r 'b' tokens with the strongest links
        _, top_b_indices = torch.topk(values, r, dim=-1) # [B, r]


    return indices, top_b_indices

class MergeDNALayer(nn.Module):
    def __init__(self, dim, merge_r=1):

        super().__init__()
        self.dim = dim
        self.merge_r = merge_r
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
        
        metric = self.metric_proj(x)
        
        node_indices, top_b_indices = bipartite_soft_matching(metric, r)
        
        if node_indices is None:
            return x, None

        
        with torch.no_grad():

            batch_idx = torch.arange(B).view(B, 1).to(x.device)

            global_b_indices = top_b_indices * 2 + 1
            
            matched_a_local = torch.gather(node_indices, 1, top_b_indices)
            global_a_indices = matched_a_local * 2
            
            mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
            mask.scatter_(1, global_b_indices, False)
            
        b_features = torch.gather(x, 1, global_b_indices.unsqueeze(-1).expand(-1, -1, C))
        x_out = x.clone()
        x_out.scatter_add_(1, global_a_indices.unsqueeze(-1).expand(-1, -1, C), b_features)
        
        counts = torch.ones(B, N, 1, device=x.device)
        counts.scatter_add_(1, global_a_indices.unsqueeze(-1).expand(-1, -1, 1), 
                            torch.ones_like(b_features[:, :, :1]))
        x_out = x_out / counts
        
        kept_indices = torch.nonzero(mask, as_tuple=False)  # [num_kept, 2]
        
        kept_per_batch = mask.sum(dim=1)  # Should be N - r for each batch
        
        x_final_list = []
        for b in range(B):
            batch_mask = mask[b]  # [N]
            x_final_list.append(x_out[b, batch_mask])  # [N-r, C]
        x_final = torch.stack(x_final_list, dim=0)  # [B, N-r, C]

        ownership = torch.zeros(B, N, dtype=torch.long, device=x.device)
        
 
        new_indices = torch.cumsum(mask.long(), dim=1) - 1
        
        ownership[mask] = new_indices[mask]
        
        target_a_new_indices = torch.gather(new_indices, 1, global_a_indices)
        ownership.scatter_(1, global_b_indices, target_a_new_indices)
        
        return x_final, ownership

class MergeDNAUnmerge(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x_merged, ownership_map):
        """
        x_merged: [Batch, N_reduced, Dim]
        ownership_map: [Batch, N_original] - Indices mapping
        
        Returns: [Batch, N_original, Dim]
        """
        B, N_reduced, C = x_merged.shape
        B, N_original = ownership_map.shape
        
        indices = ownership_map.unsqueeze(-1).expand(-1, -1, C)
        x_unmerged = torch.gather(x_merged, 1, indices)
        
        return x_unmerged


def example_usage():
    B, N, Dim = 2, 10, 8
    x = torch.randn(B, N, Dim)
    
    merger = MergeDNALayer(dim=Dim, merge_r=N//2) # Reduce by half
    x_latent, source_map = merger(x)
    
    print(f"Original Shape: {x.shape}")        # [2, 10, 8]
    print(f"Latent Shape:   {x_latent.shape}") # [2, 5, 8]
    
    x_processed = x_latent # Simulating pass-through
    
    unmerger = MergeDNAUnmerge()
    x_recon = unmerger(x_processed, source_map)
    
    print(f"Recon Shape:    {x_recon.shape}") 
    print(torch.sum(x_recon - x)) # [2, 10, 8]
    
    print("Reconstruction check passed:", x_recon.shape == x.shape)

if __name__ == "__main__":
    example_usage()