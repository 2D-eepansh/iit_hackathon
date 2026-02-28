"""Main training script for SVAMITVA multi-class feature extraction."""

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.unified_dataset import (
    UnifiedMultiClassDataset,
    DEFAULT_SOURCES,
    get_train_transform,
    get_val_transform,
)
from src.losses.multiclass_loss import MultiClassCompositeLoss
from src.models.model_factory import create_model
from src.training.train_one_epoch import train_one_epoch, validate_multiclass


# ── TIFF-level train / val split (no spatial leakage) ─────────────────────
# PB TIFFs (EPSG:32643)  — Road + Built-Up strong
# CG TIFFs (EPSG:32644)  — Road + Bridge + Built-Up
# Hold out 1 PB + 1 CG TIFF for validation.
TRAIN_TIFFS = [
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho",          # PB
    "TIMMOWAL_37695_ORI",                                # PB
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO",  # CG-2
    "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO",        # CG-3  (has 5 bridges)
]
VAL_TIFFS = [
    "28996_NADALA_ORTHO",                                # PB
    "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO",   # CG-3  (has 1 bridge)
]

# Training configuration
CONFIG = {
    # Data
    "image_size": 512,
    "patches_per_image": 50,   # random patches sampled per TIFF per epoch
    "val_ratio": 0.2,

    # Training
    "batch_size": 4,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "encoder_lr": 3e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "accumulation_steps": 1,

    # Model
    "architecture": "Unet",
    "encoder_name": "efficientnet-b4",
    "encoder_weights": "imagenet",
    "classes": 4,              # 0=Background, 1=Road, 2=Bridge, 3=Built-Up
    "seed": 42,
    "use_gradient_checkpointing": False,

    # Loss
    "ce_weight": 0.5,
    "dice_weight": 0.5,

    # DataLoader
    "num_workers": min(4, os.cpu_count() or 4),
    "persistent_workers": True,
    "prefetch_factor": 2,

    # Output
    "output_dir": "outputs",
    "checkpoint_dir": "outputs/checkpoints",
}


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility."""
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def setup_cuda_optimizations() -> None:
    """Enable CUDA optimizations for better performance."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from the unified PB + CG dataset.

    Split is controlled by TRAIN_TIFFS / VAL_TIFFS lists to prevent spatial
    leakage between train and validation.

    Args:
        config: Training configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = UnifiedMultiClassDataset(
        sources=DEFAULT_SOURCES,
        split="train",
        transform=get_train_transform(config["image_size"]),
        patch_size=config["image_size"],
        patches_per_image=config.get("patches_per_image", 50),
        train_tiffs=TRAIN_TIFFS,
        val_tiffs=VAL_TIFFS,
    )

    val_dataset = UnifiedMultiClassDataset(
        sources=DEFAULT_SOURCES,
        split="val",
        transform=get_val_transform(config["image_size"]),
        patch_size=config["image_size"],
        patches_per_image=config.get("patches_per_image", 50),
        train_tiffs=TRAIN_TIFFS,
        val_tiffs=VAL_TIFFS,
    )

    nw = config["num_workers"]
    pw = config["persistent_workers"] if nw > 0 else False
    pf = config["prefetch_factor"] if nw > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=pf,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=pf,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader


def main() -> None:
    """Main training function."""
    config = CONFIG

    # Setup
    set_seed(config["seed"])
    setup_cuda_optimizations()
    Path(config["output_dir"]).mkdir(exist_ok=True)
    Path(config["checkpoint_dir"]).mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("TRAINING CONFIGURATION  —  SVAMITVA Multi-Class Pipeline")
    print("=" * 80)
    print(f"Device:        {device}")
    if torch.cuda.is_available():
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
        print(f"CUDA:          {torch.version.cuda}")
    print(f"Dataset:       Unified PB + CG ({len(TRAIN_TIFFS)} train, {len(VAL_TIFFS)} val TIFFs)")
    print(f"Classes:       {config['classes']}  "
          "(0=BG, 1=Road, 2=Bridge, 3=Built-Up Area)")
    print(f"Image size:    {config['image_size']}")
    print(f"Batch size:    {config['batch_size']}")
    print(f"Epochs:        {config['num_epochs']}")
    print(f"Workers:       {config['num_workers']}")
    print("=" * 80)

    # Create dataloaders
    print("\nCreating SVAMITVA dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        architecture=config["architecture"],
        encoder_name=config["encoder_name"],
        encoder_weights=config["encoder_weights"],
        in_channels=3,
        classes=config["classes"],
        use_gradient_checkpointing=config["use_gradient_checkpointing"],
    )
    model = model.to(device)

    # Multi-class loss
    criterion = MultiClassCompositeLoss(
        num_classes=config["classes"],
        ce_weight=config["ce_weight"],
        dice_weight=config["dice_weight"],
    )

    # Optimizer with differential learning rates
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config["encoder_lr"]},
            {"params": decoder_params, "lr": config["learning_rate"]},
        ],
        weight_decay=config["weight_decay"],
    )

    # Scheduler  (monitors val_iou — same key name as binary pipeline)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        min_lr=1e-6,
    )

    # AMP scaler
    scaler = GradScaler("cuda")

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    best_iou = 0.0
    epoch_times = []

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.time()

        print(f"\n{'─' * 80}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'─' * 80}")

        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            max_grad_norm=config["max_grad_norm"],
            accumulation_steps=config["accumulation_steps"],
        )

        # Validate  (multi-class)
        val_metrics = validate_multiclass(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=config["classes"],
        )

        # Update scheduler on mean IoU
        scheduler.step(val_metrics["val_iou"])

        # GPU memory
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        else:
            mem_allocated = mem_reserved = 0.0

        # ETA
        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = config["num_epochs"] - epoch
        eta_seconds = int(avg_epoch_time * remaining_epochs)
        h, m, s = eta_seconds // 3600, (eta_seconds % 3600) // 60, eta_seconds % 60
        eta_str = f"{h:02d}:{m:02d}:{s:02d}"

        # Print metrics
        print(f"\nTrain Loss:  {train_metrics['train_loss']:.4f}")
        print(f"Val Loss:    {val_metrics['val_loss']:.4f}")
        print(f"Val mIoU:    {val_metrics['val_iou']:.4f}")
        print(f"Val mDice:   {val_metrics['val_dice']:.4f}")
        print(f"Epoch Time:  {train_metrics['train_time']:.2f}s")
        print(f"GPU Memory:  {mem_allocated:.2f}GB / {mem_reserved:.2f}GB")
        print(f"LR:          {scheduler.get_last_lr()[0]:.6f}")
        print(f"ETA:         {eta_str}")

        # Save best model
        if val_metrics["val_iou"] > best_iou:
            best_iou = val_metrics["val_iou"]
            checkpoint_path = Path(config["checkpoint_dir"]) / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_iou": best_iou,
                    "config": config,
                    "metrics": {**train_metrics, **val_metrics},
                },
                checkpoint_path,
            )
            print(f"\n✓ Saved best model (mIoU: {best_iou:.4f})")

        # Save latest checkpoint
        checkpoint_path = Path(config["checkpoint_dir"]) / "latest_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_iou": best_iou,
                "config": config,
                "metrics": {**train_metrics, **val_metrics},
            },
            checkpoint_path,
        )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation mIoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
