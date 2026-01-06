#!/usr/bin/env python3
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from mergedna.dataloader import (
    sequence_to_tensor,
    tensor_to_sequence,
    DNA_VOCAB,
    IDX_TO_BASE
)
from mergedna.backbone import MergeDNAModel
from mergedna.local_modules import LocalEncoder, LocalDecoder


def load_model(checkpoint_path: str, device: torch.device) -> MergeDNAModel:
    """
    Load a pre-trained MergeDNA model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Loaded MergeDNAModel in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model args from checkpoint or use defaults
    args = checkpoint.get("args", {})
    dim = args.get("dim", 64)
    latent_enc_depth = args.get("latent_enc_depth", 2)
    latent_dec_depth = args.get("latent_dec_depth", 2)
    merge_ratio = args.get("merge_ratio", 0.5)

    # Initialize model
    local_encoder = LocalEncoder(dim=dim, merge_ratio=merge_ratio).to(device)
    local_decoder = LocalDecoder(dim=dim).to(device)

    model = MergeDNAModel(
        local_encoder=local_encoder,
        local_decoder=local_decoder,
        dim=dim,
        latent_enc_depth=latent_enc_depth,
        latent_dec_depth=latent_dec_depth,
        vocab_size=4
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown') + 1}")
    print(f"Training loss: {checkpoint.get('loss', 'unknown'):.4f}")

    return model


@torch.no_grad()
def reconstruct_sequence(
    model: MergeDNAModel,
    sequence: str,
    device: torch.device
) -> dict:
    """
    Reconstruct a DNA sequence using the MergeDNA model.

    Args:
        model: Pre-trained MergeDNA model
        sequence: Input DNA sequence string
        device: Device for computation

    Returns:
        Dictionary with reconstruction results:
        - original: Original sequence
        - reconstructed: Reconstructed sequence
        - accuracy: Reconstruction accuracy
        - mean_error: Mean cross-entropy error
        - per_position_error: Per-position error array
        - compression_ratio: Achieved compression ratio
    """
    model.eval()

    # Convert to tensor
    x = sequence_to_tensor(sequence).unsqueeze(0).to(device)
    x_onehot = F.one_hot(x, num_classes=4).float()

    # Forward pass
    logits, predictions, info = model(x_onehot)

    # Calculate reconstruction error
    loss = F.cross_entropy(
        logits.view(-1, 4),
        x.view(-1),
        reduction="none"
    )
    per_position_error = loss.view(x.shape)

    # Get reconstructed sequence
    reconstructed = tensor_to_sequence(predictions[0])

    # Calculate accuracy
    accuracy = (predictions == x).float().mean().item()

    return {
        "original": sequence,
        "reconstructed": reconstructed,
        "accuracy": accuracy,
        "mean_error": per_position_error.mean().item(),
        "per_position_error": per_position_error[0].cpu().numpy(),
        "compression_ratio": info["compression_ratio"]
    }


@torch.no_grad()
def extract_embedding(
    model: MergeDNAModel,
    sequence: str,
    device: torch.device,
    pooling: str = "mean"
) -> np.ndarray:
    """
    Extract latent embedding for a DNA sequence.

    Args:
        model: Pre-trained MergeDNA model
        sequence: Input DNA sequence string
        device: Device for computation
        pooling: Pooling method ("mean", "max", or "cls")

    Returns:
        Latent embedding as numpy array
    """
    model.eval()

    x = sequence_to_tensor(sequence).unsqueeze(0).to(device)
    z_latent, _ = model.encode(x)

    if pooling == "mean":
        embedding = z_latent.mean(dim=1)
    elif pooling == "max":
        embedding = z_latent.max(dim=1)[0]
    elif pooling == "cls":
        embedding = z_latent[:, 0]
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

    return embedding.cpu().numpy()[0]


def compute_anomaly_score(
    model: MergeDNAModel,
    sequence: str,
    device: torch.device
) -> float:
    """
    Compute anomaly score based on reconstruction error.

    Higher scores indicate more anomalous sequences.

    Args:
        model: Pre-trained MergeDNA model
        sequence: Input DNA sequence string
        device: Device for computation

    Returns:
        Anomaly score (mean reconstruction error)
    """
    result = reconstruct_sequence(model, sequence, device)
    return result["mean_error"]


def parse_args():
    parser = argparse.ArgumentParser(description="MergeDNA Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Single DNA sequence to process"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="File with sequences (one per line)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for results"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["reconstruct", "embed", "anomaly"],
        default="reconstruct",
        help="Inference mode"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "max", "cls"],
        default="mean",
        help="Pooling method for embeddings"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Get sequences
    sequences = []
    if args.sequence:
        sequences = [args.sequence]
    elif args.input_file:
        with open(args.input_file, "r") as f:
            sequences = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Provide either --sequence or --input_file")
        return

    print(f"Processing {len(sequences)} sequence(s)...")

    results = []
    for i, seq in enumerate(sequences):
        if args.mode == "reconstruct":
            result = reconstruct_sequence(model, seq, device)
            results.append(result)
            print(f"\nSequence {i + 1}:")
            print(f"  Original:      {result['original'][:50]}...")
            print(f"  Reconstructed: {result['reconstructed'][:50]}...")
            print(f"  Accuracy:      {result['accuracy'] * 100:.2f}%")
            print(f"  Mean Error:    {result['mean_error']:.4f}")
            print(f"  Compression:   {result['compression_ratio']:.2f}x")

        elif args.mode == "embed":
            embedding = extract_embedding(model, seq, device, args.pooling)
            results.append(embedding)
            print(f"Sequence {i + 1}: embedding shape = {embedding.shape}")

        elif args.mode == "anomaly":
            score = compute_anomaly_score(model, seq, device)
            results.append({"sequence": seq[:50], "anomaly_score": score})
            print(f"Sequence {i + 1}: anomaly score = {score:.4f}")

    # Save results if output file specified
    if args.output_file:
        if args.mode == "embed":
            np.save(args.output_file, np.array(results))
            print(f"\nEmbeddings saved to {args.output_file}")
        else:
            import json
            with open(args.output_file, "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = []
                for r in results:
                    if isinstance(r, dict):
                        r_copy = r.copy()
                        if "per_position_error" in r_copy:
                            r_copy["per_position_error"] = r_copy["per_position_error"].tolist()
                        serializable_results.append(r_copy)
                    else:
                        serializable_results.append(r)
                json.dump(serializable_results, f, indent=2)
            print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
