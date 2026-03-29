"""Training and validation functions."""

import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from src.datasets.unified_dataset import CLASS_NAMES


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 1,
    ema=None,
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
            if ema is not None:
                ema.update(model)  # per-step EMA update (decay=0.999 meaningful at this granularity)
        
        running_loss += loss.item() * accumulation_steps
    
    # Handle remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
    
    epoch_time = time.time() - start_time
    mean_loss = running_loss / len(dataloader)
    
    return {
        "train_loss": mean_loss,
        "train_time": epoch_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-class validation (used by SVAMITVA / UnifiedMultiClassDataset pipeline)
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def validate_multiclass(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 4,
    use_multiscale: bool = False,
    use_road_refinement: bool = False,
    use_tta: bool = False,
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
        use_multiscale: Enable multi-scale inference (1.0x + 0.75x).
        use_road_refinement: Apply morphological refinement to Road class.

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
    global_gt_pixels = torch.zeros(num_classes, device=device, dtype=torch.float64)

    # Sanity-check accumulators for unique class tracking
    all_unique_preds = set()
    all_unique_gt = set()

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)   # (B, H, W) long

        with autocast(device_type="cuda"):
            logits = model(images)                    # (B, C, H, W)

            # Multi-scale inference: average logits at 1.0x and 0.75x scales
            if use_multiscale:
                B, C, H, W = logits.shape
                # Downsample input to 0.75x
                images_small = F.interpolate(images, scale_factor=0.75, mode='bilinear', align_corners=False)
                logits_small = model(images_small)
                # Upsample logits back to original size
                logits_small_up = F.interpolate(logits_small, size=(H, W), mode='bilinear', align_corners=False)
                # Average logits
                logits = (logits + logits_small_up) / 2.0

            # Test-time augmentation: average with flipped predictions
            if use_tta:
                B, C, H, W = logits.shape
                # Horizontal flip
                logits_hflip = torch.flip(model(torch.flip(images, [3])), [3])
                # Vertical flip
                logits_vflip = torch.flip(model(torch.flip(images, [2])), [2])
                logits = (logits + logits_hflip + logits_vflip) / 3.0

            loss = criterion(logits, masks)

        preds = logits.argmax(dim=1)                  # (B, H, W)
        
        # Morphological refinement for Road class (class 1)
        if use_road_refinement:
            B, H, W = preds.shape
            for b in range(B):
                pred_np = preds[b].cpu().numpy().astype(np.uint8)
                road_mask = (pred_np == 1).astype(np.uint8)

                # Apply morphological closing (5x5 kernel for better gap filling)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                road_mask_refined = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

                # Remove small connected components (< 200 pixels)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask_refined, connectivity=8)
                for label_idx in range(1, num_labels):  # skip background (0)
                    area = stats[label_idx, cv2.CC_STAT_AREA]
                    if area < 200:
                        road_mask_refined[labels == label_idx] = 0

                # Merge refined road mask back into predictions
                pred_np[road_mask == 1] = 0  # clear old road pixels
                pred_np[road_mask_refined == 1] = 1  # set refined road pixels
                preds[b] = torch.from_numpy(pred_np).to(device)

        # Debug: track unique classes seen
        all_unique_preds.update(torch.unique(preds).cpu().tolist())
        all_unique_gt.update(torch.unique(masks).cpu().tolist())

        # Accumulate per-class intersection and union across batches
        for c in range(num_classes):
            p = (preds == c).float()
            t = (masks == c).float()
            inter = (p * t).sum()
            global_intersection[c] += inter
            global_union[c] += p.sum() + t.sum() - inter
            global_cardinality[c] += p.sum() + t.sum()
            global_gt_pixels[c] += t.sum()

        running_loss += loss.item()
        num_batches += 1

    n = max(num_batches, 1)
    smooth = 1e-6

    # ── Sanity prints ────────────────────────────────────────────────────────
    print(f"\n  Unique predicted classes: {sorted(all_unique_preds)}")
    print(f"  Unique GT classes:       {sorted(all_unique_gt)}")

    # ── Per-class IoU and Dice from global accumulators ──────────────────────
    per_class_iou: dict[int, float] = {}
    per_class_dice: dict[int, float] = {}
    for c in range(num_classes):
        iou_c = (global_intersection[c] + smooth) / (global_union[c] + smooth)
        dice_c = (2.0 * global_intersection[c] + smooth) / (global_cardinality[c] + smooth)
        per_class_iou[c] = iou_c.item()
        per_class_dice[c] = dice_c.item()

    # Foreground-only macro average — exclude classes with zero GT pixels
    excluded_classes: list[str] = []
    fg_ious: list[float] = []
    fg_dices: list[float] = []
    for c in range(1, num_classes):
        gt_px = global_gt_pixels[c].item()
        if gt_px < 1.0:
            excluded_classes.append(CLASS_NAMES.get(c, f"Class {c}"))
        else:
            fg_ious.append(per_class_iou[c])
            fg_dices.append(per_class_dice[c])

    macro_miou = sum(fg_ious) / len(fg_ious) if fg_ious else 0.0
    macro_mdice = sum(fg_dices) / len(fg_dices) if fg_dices else 0.0

    # ── Print per-class breakdown ────────────────────────────────────────────
    print("\n  Per-Class IoU Breakdown:")
    for c in range(1, num_classes):
        name = CLASS_NAMES.get(c, f"Class {c}")
        gt_px = int(global_gt_pixels[c].item())
        if gt_px == 0:
            # IoU = (0+ε)/(0+ε) = 1.0 is numerically valid but semantically meaningless
            print(f"    {name:14s}:  IoU=  N/A    Dice=  N/A    GT=0px  [absent in val]")
        else:
            print(f"    {name:14s}:  IoU={per_class_iou[c]:.4f}   Dice={per_class_dice[c]:.4f}   GT={gt_px:,}px")
    if excluded_classes:
        print(f"    Excluded from macro mIoU (no GT pixels): {excluded_classes}")
    print(f"    {'Macro FG':14s}:  mIoU={macro_miou:.4f}  mDice={macro_mdice:.4f}")

    return {
        "val_loss":       running_loss / n,
        "val_iou":        macro_miou,          # key kept as val_iou for scheduler compat
        "val_dice":       macro_mdice,
        "per_class_iou":  per_class_iou,
        "per_class_dice": per_class_dice,
    }
