"""
MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization.

This package implements the MergeDNA architecture for adaptive DNA sequence
tokenization and pre-training, as described in:
"MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging"
(arXiv:2511.14806)

Main Components:
- MergeDNAModel: The main model combining local and latent encoders/decoders
- LocalEncoder: Embeds DNA bases and performs token merging
- LocalDecoder: Unmerges tokens back to original length
- DNADataset: PyTorch Dataset for DNA sequences
- bipartite_soft_matching: Token merging algorithm

Example Usage:
    from mergedna import MergeDNAModel, LocalEncoder, LocalDecoder, DNADataset

    # Create model
    local_enc = LocalEncoder(dim=64, merge_ratio=0.5)
    local_dec = LocalDecoder(dim=64)
    model = MergeDNAModel(local_enc, local_dec, dim=64)

    # Create dataset
    sequences = ["ATGCATGC...", "GCTAGCTA..."]
    dataset = DNADataset(sequences)
"""

from .backbone import (
    MergeDNAModel,
    LatentEncoder,
    LatentDecoder,
    GlobalTokenSelector,
    TransformerBlock,
    FlashAttention,
    FeedForward,
)

from .local_modules import (
    LocalEncoder,
    LocalDecoder,
)

from .merging import (
    bipartite_soft_matching,
    MergeDNALayer,
    MergeDNAUnmerge,
)

from .dataloader import (
    DNADataset,
    DNA_VOCAB,
    IDX_TO_BASE,
    load_sequences,
    merge_sequences,
    get_split_sequences,
    create_dataloader,
    collate_dna_sequences,
    sequence_to_tensor,
    tensor_to_sequence,
    dataloader,  # Backward compatibility alias
)

__version__ = "0.1.0"

__all__ = [
    # Model
    "MergeDNAModel",
    "LocalEncoder",
    "LocalDecoder",
    "LatentEncoder",
    "LatentDecoder",
    "GlobalTokenSelector",
    # Building blocks
    "TransformerBlock",
    "FlashAttention",
    "FeedForward",
    # Merging
    "bipartite_soft_matching",
    "MergeDNALayer",
    "MergeDNAUnmerge",
    # Data
    "DNADataset",
    "DNA_VOCAB",
    "IDX_TO_BASE",
    "load_sequences",
    "merge_sequences",
    "get_split_sequences",
    "create_dataloader",
    "collate_dna_sequences",
    "sequence_to_tensor",
    "tensor_to_sequence",
    "dataloader",
]
