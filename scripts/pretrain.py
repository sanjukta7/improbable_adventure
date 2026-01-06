#!/usr/bin/env python3

#unused scripts, wrote earlier as reference. 

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from mergedna.dataloader import (
    load_sequences,
    merge_sequences,
    create_dataloader,
    DNADataset,
    collate_dna_sequences
)
from mergedna.backbone import MergeDNAModel
from mergedna.local_modules import LocalEncoder, LocalDecoder


def parse_args():
    parser = argparse.ArgumentParser(description="MergeDNA Pre-training")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/human_nontata_promoters",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument("--dim", type=int, default=64, help="Model dimension")
    parser.add_argument("--latent_enc_depth", type=int, default=2, help="Latent encoder depth")
    parser.add_argument("--latent_dec_depth", type=int, default=2, help="Latent decoder depth")
    parser.add_argument("--merge_ratio", type=float, default=0.5, help="Token merge ratio")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lambda_latent", type=float, default=0.25, help="Latent MTR loss weight")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Max sequence length")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, lambda_latent=0.25):
    """Train for one epoch."""
    model.train()

    epoch_losses = {"total": [], "mtr": [], "latent": [], "amtm": []}

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        optimizer.zero_grad()

        try:
            loss, logs = model.forward_train(batch, lambda_latent=lambda_latent)
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_losses["total"].append(loss.item())
        epoch_losses["mtr"].append(logs["loss_mtr"])
        epoch_losses["latent"].append(logs["loss_latent"])
        epoch_losses["amtm"].append(logs["loss_amtm"])

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "mtr": f"{logs['loss_mtr']:.3f}",
            "amtm": f"{logs['loss_amtm']:.3f}"
        })

    return {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "mergedna_pretrain.pt")

    # Load data
    print(f"Loading data from {args.data_dir}...")
    sequences_dict = load_sequences(args.data_dir)
    sequences = merge_sequences(sequences_dict)
    print(f"Loaded {len(sequences)} sequences")

    # Create dataloader
    dataloader = create_dataloader(
        sequences,
        batch_size=args.batch_size,
        shuffle=True,
        max_len=args.max_seq_len,
        num_workers=args.num_workers
    )
    print(f"Number of batches: {len(dataloader)}")

    # Initialize model
    local_encoder = LocalEncoder(
        dim=args.dim,
        merge_ratio=args.merge_ratio
    ).to(device)

    local_decoder = LocalDecoder(dim=args.dim).to(device)

    model = MergeDNAModel(
        local_encoder=local_encoder,
        local_decoder=local_decoder,
        dim=args.dim,
        latent_enc_depth=args.latent_enc_depth,
        latent_dec_depth=args.latent_dec_depth,
        vocab_size=4
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    history = {"total": [], "mtr": [], "latent": [], "amtm": [], "lr": []}
    best_loss = float("inf")

    print("\nStarting pre-training...")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        epoch_losses = train_epoch(
            model, dataloader, optimizer, device,
            lambda_latent=args.lambda_latent
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Update history
        for key in ["total", "mtr", "latent", "amtm"]:
            history[key].append(epoch_losses[key])
        history["lr"].append(current_lr)

        print(
            f"  Total: {epoch_losses['total']:.4f} | "
            f"MTR: {epoch_losses['mtr']:.4f} | "
            f"Latent: {epoch_losses['latent']:.4f} | "
            f"AMTM: {epoch_losses['amtm']:.4f}"
        )

        # Save best model
        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": best_loss,
                "history": history,
                "args": vars(args)
            }, checkpoint_path)
            print(f"  Saved best model (loss: {best_loss:.4f})")

    print("\n" + "=" * 60)
    print("Pre-training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
