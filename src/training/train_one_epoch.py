"""Training and validation functions."""

import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 1,
) -> dict[str, float]:
    """
    Train model for one epoch with optimized settings.

    Args:
        model: Segmentation model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        device: Device to train on.
        max_grad_norm: Maximum gradient norm for clipping.
        accumulation_steps: Gradient accumulation steps.

    Returns:
        Dictionary with training metrics.
    """
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        # Move to device
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Forward pass with AMP
        with autocast(device_type="cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
            loss = loss / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Optimizer step with accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        running_loss += loss.item() * accumulation_steps
    
    # Handle remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    epoch_time = time.time() - start_time
    mean_loss = running_loss / len(dataloader)
    
    return {
        "train_loss": mean_loss,
        "train_time": epoch_time,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Validate model with vectorized metrics.

    Args:
        model: Segmentation model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Device to validate on.
        threshold: Threshold for binary prediction.

    Returns:
        Dictionary with validation metrics.
    """
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0
    
    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Forward pass
        with autocast(device_type="cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        
        # Compute predictions once
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Vectorized metrics on GPU
        iou, dice = compute_metrics_batch(preds, masks)
        
        running_loss += loss.item()
        total_iou += iou
        total_dice += dice
        num_batches += 1
    
    return {
        "val_loss": running_loss / num_batches,
        "val_iou": total_iou / num_batches,
        "val_dice": total_dice / num_batches,
    }


def compute_metrics_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> tuple[float, float]:
    """
    Compute IoU and Dice in single pass (vectorized, GPU).

    Args:
        preds: Predicted binary masks (B, C, H, W).
        targets: Ground truth masks (B, C, H, W).
        smooth: Smoothing constant.

    Returns:
        Tuple of (iou, dice) scores.
    """
    # Flatten spatial dimensions
    preds_flat = preds.flatten(1)
    targets_flat = targets.flatten(1)
    
    # Compute metrics
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    cardinality = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    
    iou = ((intersection + smooth) / (union + smooth)).mean()
    dice = ((2.0 * intersection + smooth) / (cardinality + smooth)).mean()
    
    return iou.item(), dice.item()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-class equivalents (used by SVAMITVA / MultiClassDataset pipeline)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "Background",
    1: "Road",
    2: "Bridge",
    3: "Built-Up Area",
}


@torch.no_grad()
def validate_multiclass(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 4,
) -> dict[str, float]:
    """
    Validate model for multi-class segmentation with per-class IoU breakdown.

    Accumulates global intersection / union per class across all batches for
    accurate IoU computation, then prints a per-class breakdown table.
    Classes with zero ground-truth pixels are excluded from macro mIoU.

    Args:
        model: Segmentation model with num_classes output channels.
        dataloader: Validation dataloader (masks are long tensors).
        criterion: Multi-class loss function.
        device: Device to validate on.
        num_classes: Total number of classes including background.
        num_classes: Total number of classes including background.

    Returns:
        Dictionary with val_loss, val_iou (macro foreground mIoU), val_dice,
        and per_class_iou / per_class_dice dicts keyed by class index.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    # ── Global accumulators for per-class metrics ────────────────────────────
    global_intersection = torch.zeros(num_classes, device=device, dtype=torch.float64)
    global_union = torch.zeros(num_classes, device=device, dtype=torch.float64)
    global_cardinality = torch.zeros(num_classes, device=device, dtype=torch.float64)

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)   # (B, H, W) long

        with autocast(device_type="cuda"):
            logits = model(images)                    # (B, C, H, W)
            loss = criterion(logits, masks)

        preds = logits.argmax(dim=1)                  # (B, H, W)

        # Accumulate per-class intersection and union across batches
        for c in range(num_classes):
            p = (preds == c).float()
            t = (masks == c).float()
            inter = (p * t).sum()
            global_intersection[c] += inter
            global_union[c] += p.sum() + t.sum() - inter
            global_cardinality[c] += p.sum() + t.sum()

        running_loss += loss.item()
        num_batches += 1

    n = max(num_batches, 1)
    smooth = 1e-6

    # ── Per-class IoU and Dice from global accumulators ──────────────────────
    per_class_iou: dict[int, float] = {}
    per_class_dice: dict[int, float] = {}
    for c in range(num_classes):
        iou_c = (global_intersection[c] + smooth) / (global_union[c] + smooth)
        dice_c = (2.0 * global_intersection[c] + smooth) / (global_cardinality[c] + smooth)
        per_class_iou[c] = iou_c.item()
        per_class_dice[c] = dice_c.item()

    # Foreground-only macro average (classes 1 .. C-1)
    fg_ious = [per_class_iou[c] for c in range(1, num_classes)]
    fg_dices = [per_class_dice[c] for c in range(1, num_classes)]
    macro_miou = sum(fg_ious) / len(fg_ious) if fg_ious else 0.0
    macro_mdice = sum(fg_dices) / len(fg_dices) if fg_dices else 0.0

    # ── Print per-class breakdown ────────────────────────────────────────────
    print("\n  Per-Class IoU Breakdown:")
    for c in range(1, num_classes):
        name = CLASS_NAMES.get(c, f"Class {c}")
        print(f"    {name:14s}:  IoU={per_class_iou[c]:.4f}   Dice={per_class_dice[c]:.4f}")
    print(f"    {'Macro FG':14s}:  mIoU={macro_miou:.4f}  mDice={macro_mdice:.4f}")

    return {
        "val_loss":       running_loss / n,
        "val_iou":        macro_miou,          # key kept as val_iou for scheduler compat
        "val_dice":       macro_mdice,
        "per_class_iou":  per_class_iou,
        "per_class_dice": per_class_dice,
    }
