"""
Model Audit Script — Clean, No-TTA, No-Postprocessing Evaluation.

Evaluates the retrained model's RAW output quality:
  - Confidence analysis on softmax probabilities
  - Confusion matrix and class separation
  - Per-class IoU, Precision, Recall, F1
  - Failure pattern analysis
  - Visual sample generation

This script intentionally avoids any TTA, multiscale, road refinement,
or postprocessing to measure PURE model learning.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.unified_dataset import (
    CLASS_NAMES,
    DEFAULT_SOURCES,
    UnifiedMultiClassDataset,
    get_val_transform,
)
from src.models.model_factory import create_model

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path("outputs/checkpoints/best_model.pth")
OUTPUT_DIR = Path("outputs/audit")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_CLASSES = 5

VAL_TIFFS = [
    "28996_NADALA_ORTHO",
    "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO",
]
TRAIN_TIFFS = [
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho",
    "TIMMOWAL_37695_ORI",
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO",
    "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# ── 1. Model Loading ───────────────────────────────────────────────────────

def load_model():
    """Load EMA checkpoint with architecture verification."""
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]

    model = create_model(
        architecture=cfg["architecture"],
        encoder_name=cfg["encoder_name"],
        encoder_weights=None,
        in_channels=3,
        classes=cfg["classes"],
    )

    # Verify key counts match
    sd = ckpt["model_state_dict"]
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    if missing or unexpected:
        print(f"WARNING: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  Missing: {list(missing)[:5]}")
        if unexpected:
            print(f"  Unexpected: {list(unexpected)[:5]}")

    model.load_state_dict(sd, strict=True)
    model.to(DEVICE)
    model.eval()

    return model, cfg, ckpt


def create_val_loader(cfg):
    """Create deterministic val loader."""
    val_dataset = UnifiedMultiClassDataset(
        sources=DEFAULT_SOURCES,
        split="val",
        transform=get_val_transform(cfg.get("image_size", 768)),
        patch_size=cfg.get("image_size", 768),
        patches_per_image=cfg.get("patches_per_image", 50),
        train_tiffs=TRAIN_TIFFS,
        val_tiffs=VAL_TIFFS,
    )
    return DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


# ── 2. Inference + Collection ──────────────────────────────────────────────

def run_clean_inference(model, val_loader):
    """
    Run inference with NO TTA, NO multiscale, NO postprocessing.
    Collect raw softmax probabilities and argmax predictions.
    """
    # Accumulators
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    class_gt_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_pred_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_tp = np.zeros(NUM_CLASSES, dtype=np.int64)

    # Confidence accumulators
    # Per-class: sum of max-softmax for pixels predicted as that class
    class_conf_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_conf_count = np.zeros(NUM_CLASSES, dtype=np.int64)

    # Confidence histogram bins: [0, 0.4), [0.4, 0.7), [0.7, 1.0]
    conf_bins = np.zeros((NUM_CLASSES, 3), dtype=np.int64)  # per class: low/med/high

    # Entropy accumulator
    total_entropy_sum = 0.0
    total_pixels = 0

    # Store some samples for visualization
    vis_samples = []
    max_vis = 12

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Audit inference")):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)

            # Raw softmax probabilities
            probs = F.softmax(logits.float(), dim=1)  # (B, C, H, W)
            max_probs, preds = probs.max(dim=1)  # (B, H, W) each

            # Move to CPU/numpy
            probs_np = probs.cpu().numpy()       # (B, C, H, W)
            preds_np = preds.cpu().numpy()        # (B, H, W)
            masks_np = masks.cpu().numpy()        # (B, H, W)
            max_probs_np = max_probs.cpu().numpy()  # (B, H, W)

            B = preds_np.shape[0]

            # Confusion matrix + per-class metrics
            for b in range(B):
                p_flat = preds_np[b].flatten()
                t_flat = masks_np[b].flatten()

                for i in range(NUM_CLASSES):
                    for j in range(NUM_CLASSES):
                        confusion_matrix[i, j] += np.sum((t_flat == i) & (p_flat == j))

                for c in range(NUM_CLASSES):
                    gt_mask = (t_flat == c)
                    pred_mask = (p_flat == c)
                    class_gt_pixels[c] += gt_mask.sum()
                    class_pred_pixels[c] += pred_mask.sum()
                    class_tp[c] += (gt_mask & pred_mask).sum()

            # Confidence analysis
            for b in range(B):
                mp = max_probs_np[b].flatten()
                pr = preds_np[b].flatten()

                for c in range(NUM_CLASSES):
                    mask_c = (pr == c)
                    if mask_c.sum() == 0:
                        continue
                    confs = mp[mask_c]
                    class_conf_sum[c] += confs.sum()
                    class_conf_count[c] += len(confs)

                    # Bin into low/med/high
                    conf_bins[c, 0] += (confs < 0.4).sum()
                    conf_bins[c, 1] += ((confs >= 0.4) & (confs < 0.7)).sum()
                    conf_bins[c, 2] += (confs >= 0.7).sum()

            # Entropy
            for b in range(B):
                p = probs_np[b]  # (C, H, W)
                ent = -np.sum(p * np.log(p + 1e-10), axis=0)  # (H, W)
                total_entropy_sum += ent.sum()
                total_pixels += ent.size

            # Collect visualization samples
            if len(vis_samples) < max_vis:
                images_cpu = images.cpu()
                for b in range(min(B, max_vis - len(vis_samples))):
                    vis_samples.append({
                        "image": images_cpu[b],
                        "pred": preds_np[b],
                        "gt": masks_np[b],
                        "max_prob": max_probs_np[b],
                        "probs": probs_np[b],
                    })

    return {
        "confusion_matrix": confusion_matrix,
        "class_gt_pixels": class_gt_pixels,
        "class_pred_pixels": class_pred_pixels,
        "class_tp": class_tp,
        "class_conf_sum": class_conf_sum,
        "class_conf_count": class_conf_count,
        "conf_bins": conf_bins,
        "total_entropy_sum": total_entropy_sum,
        "total_pixels": total_pixels,
        "vis_samples": vis_samples,
    }


# ── 3. Metric Computation ─────────────────────────────────────────────────

def compute_all_metrics(data):
    """Compute comprehensive metrics from accumulated data."""
    cm = data["confusion_matrix"]
    gt = data["class_gt_pixels"]
    pred = data["class_pred_pixels"]
    tp = data["class_tp"]
    conf_sum = data["class_conf_sum"]
    conf_count = data["class_conf_count"]
    conf_bins = data["conf_bins"]

    metrics = {}

    # Per-class IoU, Precision, Recall, F1
    per_class = {}
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES.get(c, f"Class_{c}")
        iou = tp[c] / (gt[c] + pred[c] - tp[c] + 1e-10) if (gt[c] + pred[c] - tp[c]) > 0 else 0.0
        precision = tp[c] / (pred[c] + 1e-10) if pred[c] > 0 else 0.0
        recall = tp[c] / (gt[c] + 1e-10) if gt[c] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-10) if (precision + recall) > 0 else 0.0
        mean_conf = conf_sum[c] / conf_count[c] if conf_count[c] > 0 else 0.0

        total_bins = conf_bins[c].sum()
        if total_bins > 0:
            low_pct = conf_bins[c, 0] / total_bins * 100
            med_pct = conf_bins[c, 1] / total_bins * 100
            high_pct = conf_bins[c, 2] / total_bins * 100
        else:
            low_pct = med_pct = high_pct = 0.0

        per_class[name] = {
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "gt_pixels": int(gt[c]),
            "pred_pixels": int(pred[c]),
            "tp_pixels": int(tp[c]),
            "mean_confidence": float(mean_conf),
            "conf_low_pct": float(low_pct),
            "conf_med_pct": float(med_pct),
            "conf_high_pct": float(high_pct),
        }

    metrics["per_class"] = per_class

    # Foreground mIoU (exclude background, exclude classes with 0 GT)
    fg_ious = []
    for c in range(1, NUM_CLASSES):
        name = CLASS_NAMES.get(c, f"Class_{c}")
        if gt[c] > 0:
            fg_ious.append(per_class[name]["iou"])
    metrics["fg_miou"] = float(np.mean(fg_ious)) if fg_ious else 0.0

    # Overall pixel accuracy
    total_correct = tp.sum()
    total_px = gt.sum()
    metrics["pixel_accuracy"] = float(total_correct / total_px) if total_px > 0 else 0.0

    # Mean entropy
    metrics["mean_entropy"] = float(data["total_entropy_sum"] / data["total_pixels"]) if data["total_pixels"] > 0 else 0.0

    # Global mean confidence
    global_conf = conf_sum.sum() / conf_count.sum() if conf_count.sum() > 0 else 0.0
    metrics["global_mean_confidence"] = float(global_conf)

    # Global confidence distribution
    total_bins = conf_bins.sum()
    if total_bins > 0:
        metrics["global_conf_low_pct"] = float(conf_bins[:, 0].sum() / total_bins * 100)
        metrics["global_conf_med_pct"] = float(conf_bins[:, 1].sum() / total_bins * 100)
        metrics["global_conf_high_pct"] = float(conf_bins[:, 2].sum() / total_bins * 100)
    else:
        metrics["global_conf_low_pct"] = 0.0
        metrics["global_conf_med_pct"] = 0.0
        metrics["global_conf_high_pct"] = 0.0

    # Confusion matrix
    metrics["confusion_matrix"] = cm.tolist()

    # Confusion matrix normalized by row (GT)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums > 0)
    metrics["confusion_matrix_normalized"] = cm_norm.tolist()

    return metrics


# ── 4. Report Generation ──────────────────────────────────────────────────

def print_report(metrics, ckpt_meta):
    """Print comprehensive audit report to console."""
    lines = []
    sep = "=" * 80

    lines.append(f"\n{sep}")
    lines.append("  MODEL AUDIT REPORT — CLEAN INFERENCE (NO TTA / NO POSTPROCESSING)")
    lines.append(sep)

    # Section 1: Model Load Status
    lines.append("\n## SECTION 1: MODEL LOAD STATUS")
    lines.append(f"  Checkpoint:    {CHECKPOINT_PATH}")
    lines.append(f"  Architecture:  {ckpt_meta['architecture']}")
    lines.append(f"  Encoder:       {ckpt_meta['encoder']}")
    lines.append(f"  Classes:       {ckpt_meta['classes']}")
    lines.append(f"  Best Epoch:    {ckpt_meta['epoch']}")
    lines.append(f"  Saved mIoU:    {ckpt_meta['best_iou']:.4f}")
    lines.append(f"  Loss Config:   OHEM-CE({ckpt_meta['ce_weight']}) + SmoothedDice({ckpt_meta['dice_weight']})")
    lines.append(f"  EMA Decay:     {ckpt_meta['ema_decay']}")
    lines.append(f"  Status:        LOADED SUCCESSFULLY (strict=True, 0 missing, 0 unexpected)")

    # Section 2: Confidence Analysis
    lines.append(f"\n## SECTION 2: CONFIDENCE ANALYSIS")
    lines.append(f"  Global Mean Confidence: {metrics['global_mean_confidence']:.4f}")
    lines.append(f"  Mean Prediction Entropy: {metrics['mean_entropy']:.4f}")
    lines.append(f"  Global Confidence Distribution:")
    lines.append(f"    HIGH (>0.7):   {metrics['global_conf_high_pct']:.1f}%")
    lines.append(f"    MEDIUM (0.4-0.7): {metrics['global_conf_med_pct']:.1f}%")
    lines.append(f"    LOW (<0.4):    {metrics['global_conf_low_pct']:.1f}%")
    lines.append(f"\n  Per-Class Confidence Breakdown:")
    lines.append(f"    {'Class':20s} {'MeanConf':>9s} {'High%':>7s} {'Med%':>7s} {'Low%':>7s}")
    lines.append(f"    {'-'*20} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")
    for name, m in metrics["per_class"].items():
        lines.append(
            f"    {name:20s} {m['mean_confidence']:9.4f} {m['conf_high_pct']:6.1f}% {m['conf_med_pct']:6.1f}% {m['conf_low_pct']:6.1f}%"
        )

    # Section 3: Class Separation (Confusion Matrix)
    lines.append(f"\n## SECTION 3: CLASS SEPARATION (Normalized Confusion Matrix)")
    lines.append(f"  Rows = Ground Truth, Cols = Predicted")
    cm_norm = np.array(metrics["confusion_matrix_normalized"])
    names_short = [CLASS_NAMES.get(i, f"C{i}")[:10] for i in range(NUM_CLASSES)]
    gt_label = "GT \\ Pred"
    header = f"    {gt_label:>12s}" + "".join(f"{n:>12s}" for n in names_short)
    lines.append(header)
    for i in range(NUM_CLASSES):
        row = f"    {names_short[i]:>12s}"
        for j in range(NUM_CLASSES):
            val = cm_norm[i, j]
            row += f"{val:12.4f}"
        lines.append(row)

    # Key confusion pairs
    lines.append(f"\n  Key Confusion Pairs:")
    confusion_pairs = [
        (1, 3, "Road -> Built-Up"),
        (3, 1, "Built-Up -> Road"),
        (1, 0, "Road -> Background"),
        (3, 0, "Built-Up -> Background"),
        (4, 0, "Water -> Background"),
        (0, 3, "Background -> Built-Up"),
        (0, 1, "Background -> Road"),
    ]
    for gt_c, pred_c, label in confusion_pairs:
        if cm_norm[gt_c].sum() > 0:
            lines.append(f"    {label:30s}: {cm_norm[gt_c, pred_c]*100:5.1f}%")

    # Section 4: Per-Class Metrics
    lines.append(f"\n## SECTION 4: PER-CLASS PERFORMANCE")
    lines.append(f"    {'Class':20s} {'IoU':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'GT px':>12s} {'Pred px':>12s}")
    lines.append(f"    {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*12}")
    for name, m in metrics["per_class"].items():
        iou_str = f"{m['iou']:.4f}" if m['gt_pixels'] > 0 else "   N/A"
        lines.append(
            f"    {name:20s} {iou_str:>8s} {m['precision']:8.4f} {m['recall']:8.4f} {m['f1']:8.4f} {m['gt_pixels']:12,} {m['pred_pixels']:12,}"
        )
    lines.append(f"\n  Foreground mIoU (excl. BG, excl. absent classes): {metrics['fg_miou']:.4f}")
    lines.append(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")

    # Section 5: Before vs After comparison
    lines.append(f"\n## SECTION 5: BEFORE vs AFTER COMPARISON")
    lines.append(f"  Previous best (Phase 4, memory): mIoU=0.5880 (with TTA+MS), train-val mIoU=0.5523")
    lines.append(f"  Previous per-class (with TTA): Road=0.531, Built-Up=0.645")
    lines.append(f"")
    lines.append(f"  Current (CLEAN, no TTA, no MS, no postproc):")
    lines.append(f"    Foreground mIoU: {metrics['fg_miou']:.4f}")
    for name, m in metrics["per_class"].items():
        if name != "Background":
            lines.append(f"    {name}: IoU={m['iou']:.4f}")
    lines.append(f"")
    lines.append(f"  NOTE: Previous 'clean' val_iou (no TTA) was 0.5523 (Phase 4 best epoch).")
    lines.append(f"  NOTE: Previous model had 4 classes; current has 5 (added Water).")
    lines.append(f"  Fair comparison is current clean mIoU vs previous clean mIoU (0.5523 vs {metrics['fg_miou']:.4f}).")

    # Section 6: Failure Analysis
    lines.append(f"\n## SECTION 6: FAILURE ANALYSIS")

    # Check for classes with very low IoU
    for name, m in metrics["per_class"].items():
        if name == "Background":
            continue
        if m["gt_pixels"] == 0:
            lines.append(f"  [{name}] ABSENT in val set (0 GT pixels) — cannot evaluate")
        elif m["iou"] < 0.1:
            lines.append(f"  [{name}] CRITICAL FAILURE: IoU={m['iou']:.4f} — model is NOT learning this class")
        elif m["iou"] < 0.3:
            lines.append(f"  [{name}] WEAK: IoU={m['iou']:.4f} — class is poorly separated")
        elif m["recall"] < 0.3:
            lines.append(f"  [{name}] LOW RECALL: {m['recall']:.4f} — model misses most instances")
        elif m["precision"] < 0.3:
            lines.append(f"  [{name}] LOW PRECISION: {m['precision']:.4f} — many false positives")
        else:
            lines.append(f"  [{name}] OK: IoU={m['iou']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}")

    # Check confidence issues
    low_conf_classes = []
    for name, m in metrics["per_class"].items():
        if m["mean_confidence"] < 0.5 and m["gt_pixels"] > 0:
            low_conf_classes.append(name)
    if low_conf_classes:
        lines.append(f"\n  LOW CONFIDENCE CLASSES: {low_conf_classes}")
        lines.append(f"  These classes have mean softmax confidence < 0.5, indicating uncertain predictions.")

    # Check for over/under prediction
    for name, m in metrics["per_class"].items():
        if m["gt_pixels"] > 0:
            ratio = m["pred_pixels"] / m["gt_pixels"]
            if ratio > 2.0:
                lines.append(f"  [{name}] OVER-PREDICTION: pred/gt ratio = {ratio:.2f}x")
            elif ratio < 0.3:
                lines.append(f"  [{name}] UNDER-PREDICTION: pred/gt ratio = {ratio:.2f}x")

    lines.append(sep)
    return "\n".join(lines)


# ── 5. Visualization ──────────────────────────────────────────────────────

def generate_visualizations(vis_samples, output_dir):
    """Generate visual validation panels."""
    output_dir.mkdir(parents=True, exist_ok=True)

    color_map = {
        0: [0, 0, 0],        # Background - black
        1: [255, 255, 0],    # Road - yellow
        2: [255, 0, 0],      # Bridge - red
        3: [0, 0, 255],      # Built-Up - blue
        4: [0, 255, 255],    # Water - cyan
    }

    for idx, sample in enumerate(vis_samples[:12]):
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        # Denormalize image
        img = sample["image"].numpy().transpose(1, 2, 0)
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img, 0, 1)

        # GT mask as RGB
        gt = sample["gt"]
        gt_rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
        for c, color in color_map.items():
            gt_rgb[gt == c] = color

        # Pred mask as RGB
        pred = sample["pred"]
        pred_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for c, color in color_map.items():
            pred_rgb[pred == c] = color

        # Confidence heatmap
        max_prob = sample["max_prob"]

        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(gt_rgb)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_rgb)
        axes[2].set_title("Prediction (Raw)")
        axes[2].axis("off")

        im = axes[3].imshow(max_prob, cmap="RdYlGn", vmin=0, vmax=1)
        axes[3].set_title("Confidence Map")
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(output_dir / f"audit_sample_{idx:02d}.png", dpi=120, bbox_inches="tight")
        plt.close()

    # Generate confusion matrix visualization
    return True


def generate_confusion_matrix_plot(metrics, output_dir):
    """Generate normalized confusion matrix heatmap."""
    cm_norm = np.array(metrics["confusion_matrix_normalized"])
    names = [CLASS_NAMES.get(i, f"C{i}") for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Normalized Confusion Matrix (Row = GT, Col = Pred)", fontsize=14)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.3f}", ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_confidence_distribution_plot(data, output_dir):
    """Generate confidence distribution histogram per class."""
    conf_bins = data["conf_bins"]
    names = [CLASS_NAMES.get(i, f"C{i}") for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(NUM_CLASSES)
    width = 0.25

    totals = conf_bins.sum(axis=1, keepdims=True).astype(np.float64)
    totals[totals == 0] = 1  # avoid div by zero
    pcts = conf_bins / totals * 100

    bars1 = ax.bar(x - width, pcts[:, 2], width, label="High (>0.7)", color="green", alpha=0.8)
    bars2 = ax.bar(x,         pcts[:, 1], width, label="Medium (0.4-0.7)", color="orange", alpha=0.8)
    bars3 = ax.bar(x + width, pcts[:, 0], width, label="Low (<0.4)", color="red", alpha=0.8)

    ax.set_xlabel("Class")
    ax.set_ylabel("% of Predicted Pixels")
    ax.set_title("Confidence Distribution by Class")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30)
    ax.legend()
    ax.set_ylim(0, 105)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  MODEL AUDIT — CLEAN INFERENCE EVALUATION")
    print("  No TTA | No Multiscale | No Post-Processing")
    print("=" * 80)
    print()

    # Step 1: Load model
    print("[1/6] Loading model...")
    model, cfg, ckpt = load_model()
    ckpt_meta = {
        "architecture": cfg["architecture"],
        "encoder": cfg["encoder_name"],
        "classes": cfg["classes"],
        "epoch": ckpt.get("epoch", "N/A"),
        "best_iou": ckpt.get("best_iou", 0),
        "ce_weight": cfg.get("ce_weight", "N/A"),
        "dice_weight": cfg.get("dice_weight", "N/A"),
        "ema_decay": cfg.get("ema_decay", "N/A"),
    }
    print(f"  Loaded: {cfg['architecture']}/{cfg['encoder_name']}, {cfg['classes']} classes, epoch {ckpt_meta['epoch']}")
    print()

    # Step 2: Create val loader
    print("[2/6] Creating validation dataloader...")
    val_loader = create_val_loader(cfg)
    print(f"  Val patches: {len(val_loader.dataset)}, batches: {len(val_loader)}")
    print()

    # Step 3: Run clean inference
    print("[3/6] Running clean inference (NO TTA, NO multiscale, NO postprocessing)...")
    t0 = time.time()
    data = run_clean_inference(model, val_loader)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s, {data['total_pixels']:,} pixels evaluated")
    print()

    # Step 4: Compute metrics
    print("[4/6] Computing metrics...")
    metrics = compute_all_metrics(data)
    print()

    # Step 5: Generate report
    print("[5/6] Generating report...")
    report = print_report(metrics, ckpt_meta)
    print(report)

    # Step 6: Generate visualizations
    print("\n[6/6] Generating visualizations...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_visualizations(data["vis_samples"], OUTPUT_DIR)
    generate_confusion_matrix_plot(metrics, OUTPUT_DIR)
    generate_confidence_distribution_plot(data, OUTPUT_DIR)
    print(f"  Saved to {OUTPUT_DIR}/")

    # Save JSON report
    # Remove non-serializable items
    json_metrics = {k: v for k, v in metrics.items()}
    with open(OUTPUT_DIR / "audit_report.json", "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"  JSON report: {OUTPUT_DIR / 'audit_report.json'}")

    # Save text report
    with open(OUTPUT_DIR / "audit_report.txt", "w") as f:
        f.write(report)
    print(f"  Text report: {OUTPUT_DIR / 'audit_report.txt'}")

    print("\nAudit complete.")


if __name__ == "__main__":
    main()
