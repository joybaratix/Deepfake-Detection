"""
Training Script for Deepfake Detection CNN.
Supports two-phase transfer learning, early stopping, and LR scheduling.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.model import DeepfakeDetector, get_model_summary
from src.dataset import create_data_loaders
from src.utils import plot_training_history


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Stats
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(resume_from=None):
    """
    Full training pipeline with two-phase transfer learning.

    Phase 1: Frozen backbone — train only the classification head (5 epochs)
    Phase 2: Unfreeze backbone — fine-tune the entire model (remaining epochs)
    """
    print("=" * 60)
    print("  Deepfake Detection — Training Pipeline")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print()

    # ─── Data ────────────────────────────────────────────
    print("Loading dataset...")
    loaders = create_data_loaders()
    print()

    # ─── Model ───────────────────────────────────────────
    start_epoch = 0
    model = DeepfakeDetector(pretrained=True)

    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resuming from epoch {start_epoch}")

    model.to(config.DEVICE)
    get_model_summary(model)

    # ─── Loss & Optimizer ────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    # ═══ PHASE 1: Frozen Backbone (Classification Head Only) ═══
    phase1_epochs = min(5, config.NUM_EPOCHS)
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Training Classification Head ({phase1_epochs} epochs)")
    print(f"{'='*60}")

    model.freeze_backbone()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE * 10,  # Higher LR for head-only training
        weight_decay=config.WEIGHT_DECAY,
    )

    for epoch in range(start_epoch, phase1_epochs):
        print(f"\nEpoch {epoch + 1}/{phase1_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, config.DEVICE
        )
        val_loss, val_acc = validate(
            model, loaders["val"], criterion, config.DEVICE
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, val_acc, val_loss, "best_model.pth")

    # ═══ PHASE 2: Fine-tune Entire Model ═══
    remaining_epochs = config.NUM_EPOCHS - phase1_epochs
    if remaining_epochs > 0:
        print(f"\n{'='*60}")
        print(f"  PHASE 2: Fine-tuning Full Model ({remaining_epochs} epochs)")
        print(f"{'='*60}")

        model.unfreeze_backbone()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            patience=config.LR_SCHEDULER_PATIENCE,
            factor=config.LR_SCHEDULER_FACTOR,
            min_lr=config.MIN_LR,
        )

        patience_counter = 0

        for epoch in range(phase1_epochs, config.NUM_EPOCHS):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS} (LR: {current_lr:.2e})")

            train_loss, train_acc = train_one_epoch(
                model, loaders["train"], criterion, optimizer, config.DEVICE
            )
            val_loss, val_acc = validate(
                model, loaders["val"], criterion, config.DEVICE
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

            # LR scheduling
            scheduler.step(val_loss)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                _save_checkpoint(model, optimizer, epoch, val_acc, val_loss, "best_model.pth")
                print(f"  ★ New best model saved! (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")

            # Early stopping
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping triggered at epoch {epoch + 1}")
                break

    # ─── Save Final Model & History ──────────────────────
    _save_checkpoint(model, optimizer, epoch, val_acc, val_loss, "final_model.pth")

    # Save training history
    history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_path = os.path.join(config.RESULTS_DIR, "training_curves.png")
    plot_training_history(history, save_path=plot_path)

    # ─── Summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"{'='*60}")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Best Val Loss:     {best_val_loss:.4f}")
    print(f"  Models saved to:   {config.MODELS_DIR}")
    print(f"  Results saved to:  {config.RESULTS_DIR}")
    print(f"{'='*60}")

    return history


def _save_checkpoint(model, optimizer, epoch, val_acc, val_loss, filename):
    """Save model checkpoint."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_acc,
        "val_loss": val_loss,
    }, path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr

    train(resume_from=args.resume)
