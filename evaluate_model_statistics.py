"""
Model Evaluation Statistics Script for SVAMITVA Multi-Class Segmentation.

Generates human-readable performance metrics from trained model predictions
suitable for non-technical stakeholders (judges, decision makers, reviewers).

Metrics computed:
  - Pixel Accuracy (overall success rate)
  - Infrastructure Detection Accuracy (non-background only)
  - Per-class accuracy, precision, recall, F1 score
  - Confusion matrix
  - Final success percentage

Output formats:
  - Console summary (human-readable)
  - evaluation_report.json (detailed statistics)
  - evaluation_report.txt (formatted text report)

Usage:
    python evaluate_model_statistics.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.unified_dataset import UnifiedMultiClassDataset, get_val_transform, DEFAULT_SOURCES
from src.models.model_factory import create_model


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path("outputs/checkpoints/best_model.pth")
OUTPUT_DIR = Path("outputs")
REPORT_JSON = OUTPUT_DIR / "evaluation_report.json"
REPORT_TXT = OUTPUT_DIR / "evaluation_report.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 4

# Validation dataset configuration (matching train.py)
VAL_TIFFS = [
    "28996_NADALA_ORTHO",  # PB
    "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO",  # CG-3
]

TRAIN_TIFFS = [
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho",
    "TIMMOWAL_37695_ORI",
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO",
    "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO",
]

# Class definitions
CLASS_NAMES = {
    0: "Background",
    1: "Road",
    2: "Bridge",
    3: "Built-Up Area",
}

CLASS_IDS = {
    "road": 1,
    "bridge": 2,
    "built_up": 3,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: Path, device: str) -> Tuple[torch.nn.Module, Dict]:
    """Load trained model and checkpoint metadata."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = create_model(
        architecture=cfg["architecture"],
        encoder_name=cfg["encoder_name"],
        encoder_weights=None,
        in_channels=3,
        classes=cfg["classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded from {checkpoint_path.name}")
    print(f"  Epoch {ckpt.get('epoch', 'N/A')} | best mIoU {ckpt.get('best_iou', 'N/A'):.4f}")
    print(f"  Architecture: {cfg['architecture']} | Encoder: {cfg['encoder_name']}")
    print(f"  Classes: {cfg['classes']}")
    print()

    return model, cfg


def create_val_dataloader(config: Dict, batch_size: int, num_workers: int) -> DataLoader:
    """Create validation dataloader using VAL_TIFFS."""
    val_dataset = UnifiedMultiClassDataset(
        sources=DEFAULT_SOURCES,
        split="val",
        transform=get_val_transform(config.get("image_size", 768)),
        patch_size=config.get("image_size", 768),
        patches_per_image=config.get("patches_per_image", 50),
        train_tiffs=TRAIN_TIFFS,
        val_tiffs=VAL_TIFFS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return val_loader


# ─────────────────────────────────────────────────────────────────────────────
# METRICS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

class MetricsAccumulator:
    """Accumulate pixel-level predictions and ground truth for metrics computation."""

    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.total_pixels = 0
        self.correct_pixels = 0
        self.class_gt_pixels = np.zeros(num_classes, dtype=np.int64)
        self.class_pred_pixels = np.zeros(num_classes, dtype=np.int64)
        self.class_tp = np.zeros(num_classes, dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Update metrics with batch predictions.

        Args:
            pred: Predicted class indices (N, H, W)
            target: Ground truth class indices (N, H, W)
        """
        # Flatten predictions and targets
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        # Overall accuracy
        self.correct_pixels += np.sum(pred_flat == target_flat)
        self.total_pixels += len(pred_flat)

        # Confusion matrix
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                count = np.sum((target_flat == i) & (pred_flat == j))
                self.confusion_matrix[i, j] += count

        # Per-class statistics
        for class_id in range(self.num_classes):
            # Ground truth pixels for this class
            gt_pixels = np.sum(target_flat == class_id)
            self.class_gt_pixels[class_id] += gt_pixels

            # Predicted pixels for this class
            pred_pixels = np.sum(pred_flat == class_id)
            self.class_pred_pixels[class_id] += pred_pixels

            # True positives
            tp = np.sum((target_flat == class_id) & (pred_flat == class_id))
            self.class_tp[class_id] += tp

    def compute_metrics(self) -> Dict:
        """Compute final metrics."""
        metrics = {}

        # 1. PIXEL ACCURACY (Overall)
        pixel_accuracy = (
            self.correct_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
        )
        metrics["pixel_accuracy"] = float(pixel_accuracy)

        # 2. INFRASTRUCTURE ACCURACY (non-background)
        infra_gt = self.class_gt_pixels[1:].sum()  # Roads, Bridges, Built-Up
        infra_correct = self.confusion_matrix[1:, 1:].diagonal().sum()
        infra_accuracy = infra_correct / infra_gt if infra_gt > 0 else 0.0
        metrics["infrastructure_accuracy"] = float(infra_accuracy)

        # 3. PER-CLASS METRICS
        metrics["per_class_metrics"] = {}
        for class_id in range(self.num_classes):
            class_name = CLASS_NAMES[class_id]
            gt_pixels = self.class_gt_pixels[class_id]
            name_key = class_name.lower().replace("-", "").replace(" ", "_")

            # Per-class accuracy
            class_correct = self.confusion_matrix[class_id, class_id]
            class_accuracy = class_correct / gt_pixels if gt_pixels > 0 else 0.0

            # Precision: TP / (TP + FP)
            pred_pixels = self.class_pred_pixels[class_id]
            tp = self.class_tp[class_id]
            precision = tp / pred_pixels if pred_pixels > 0 else 0.0

            # Recall: TP / (TP + FN)
            recall = tp / gt_pixels if gt_pixels > 0 else 0.0

            # F1 Score
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            # Create consistent key names for easier access
            if class_id == 1:
                key = "road"
            elif class_id == 2:
                key = "bridge"
            elif class_id == 3:
                key = "built_up"
            else:
                key = name_key

            metrics["per_class_metrics"][key] = {
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth_pixels": int(gt_pixels),
                "accuracy": float(class_accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }

        # 4. CONFUSION MATRIX
        metrics["confusion_matrix"] = self.confusion_matrix.tolist()

        # 5. INFRASTRUCTURE DETECTION RATES (per infrastructure class)
        metrics["infrastructure_detection"] = {}
        for class_name, class_id in CLASS_IDS.items():
            gt_pixels = self.class_gt_pixels[class_id]
            tp = self.class_tp[class_id]
            detection_rate = tp / gt_pixels if gt_pixels > 0 else 0.0
            metrics["infrastructure_detection"][class_name] = float(detection_rate)

        # 6. OVERALL SUCCESS RATE
        metrics["overall_success_rate"] = float(pixel_accuracy)

        # 7. AVERAGE INFRASTRUCTURE RATE
        infra_rates = list(metrics["infrastructure_detection"].values())
        avg_infra_rate = sum(infra_rates) / len(infra_rates) if infra_rates else 0.0
        metrics["average_infrastructure_detection_rate"] = float(avg_infra_rate)

        # 8. CLASS DISTRIBUTION
        metrics["class_distribution"] = {
            CLASS_NAMES[i]: int(self.class_gt_pixels[i]) for i in range(self.num_classes)
        }

        return metrics


def run_inference(
    model: torch.nn.Module, val_loader: DataLoader, device: str
) -> MetricsAccumulator:
    """Run inference on validation dataset and accumulate metrics."""
    accumulator = MetricsAccumulator(num_classes=4)

    model.eval()
    num_batches = len(val_loader)

    print("Running inference on validation dataset...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, total=num_batches)):
            images, masks = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Forward pass
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(images)

            # Argmax to get predicted classes
            predictions = torch.argmax(logits, dim=1)  # (B, H, W)

            # Move to numpy
            pred_np = predictions.cpu().numpy().astype(np.uint8)
            mask_np = masks.cpu().numpy().astype(np.uint8)

            # Update metrics
            accumulator.update(pred_np, mask_np)

    print(f"✓ Inference complete. Processed {accumulator.total_pixels:,} pixels")
    print()
    return accumulator


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics_summary(metrics: Dict) -> str:
    """Format metrics into human-readable summary."""
    summary = []
    summary.append("\n" + "=" * 80)
    summary.append("MODEL PERFORMANCE SUMMARY — SVAMITVA Infrastructure Segmentation")
    summary.append("=" * 80)
    summary.append("")

    # Overall accuracy
    pixel_acc = metrics["pixel_accuracy"]
    infra_acc = metrics["infrastructure_accuracy"]

    summary.append(f"Overall Pixel Accuracy:         {pixel_acc * 100:.1f}%")
    summary.append(f"Infrastructure Accuracy:       {infra_acc * 100:.1f}%")
    summary.append("")

    # Per-class performance
    summary.append("─ PER-CLASS PERFORMANCE ─")
    summary.append("")

    for class_name in ["road", "bridge", "built_up"]:
        metrics_dict = metrics["per_class_metrics"][class_name]
        gt_pixels = metrics_dict["ground_truth_pixels"]
        accuracy = metrics_dict["accuracy"]
        precision = metrics_dict["precision"]
        recall = metrics_dict["recall"]
        f1 = metrics_dict["f1_score"]

        display_name = metrics_dict["class_name"]

        summary.append(f"{display_name} Detection:")
        summary.append(f"  Ground Truth Pixels:  {gt_pixels:,}")
        summary.append(f"  Detection Accuracy:   {accuracy * 100:.1f}%")
        summary.append(f"  Precision:            {precision * 100:.1f}%")
        summary.append(f"  Recall:               {recall * 100:.1f}%")
        summary.append(f"  F1 Score:             {f1:.3f}")
        summary.append("")

    # Infrastructure detection rates
    summary.append("─ INFRASTRUCTURE DETECTION RATES ─")
    summary.append("")
    for class_name, rate in metrics["infrastructure_detection"].items():
        display_name = class_name.replace("_", " ").title()
        summary.append(f"{display_name} Detection Success:     {rate * 100:.1f}%")
    summary.append("")

    # Summary statistics
    summary.append("─ FINAL SUCCESS METRICS ─")
    summary.append("")
    overall_rate = metrics["overall_success_rate"]
    avg_infra_rate = metrics["average_infrastructure_detection_rate"]

    summary.append(f"Overall System Success Rate:    {overall_rate * 100:.1f}%")
    summary.append(f"Average Infrastructure Rate:    {avg_infra_rate * 100:.1f}%")
    summary.append("")

    # Class distribution
    summary.append("─ CLASS DISTRIBUTION IN VALIDATION SET ─")
    summary.append("")
    for class_name, pixel_count in metrics["class_distribution"].items():
        pct = 100 * pixel_count / sum(metrics["class_distribution"].values())
        summary.append(f"{class_name:20s}: {pixel_count:12,} pixels ({pct:5.1f}%)")
    summary.append("")

    # Confusion matrix
    summary.append("─ CONFUSION MATRIX ─")
    summary.append("(rows=ground truth, cols=predictions)")
    summary.append("")
    cm = np.array(metrics["confusion_matrix"])
    header = ["GT \\ Pred"] + [CLASS_NAMES[i][:8] for i in range(4)]
    summary.append(" ".join(f"{h:>12}" for h in header))

    for i in range(4):
        row = [CLASS_NAMES[i][:8]]
        for j in range(4):
            row.append(str(cm[i, j]))
        summary.append(" ".join(f"{val:>12}" for val in row))

    summary.append("")
    summary.append("=" * 80)
    summary.append("INTERPRETATION FOR STAKEHOLDERS")
    summary.append("=" * 80)
    summary.append("")
    summary.append(f"✓ The AI system correctly classifies {overall_rate * 100:.1f}% of pixels in")
    summary.append("  the validation imagery.")
    summary.append("")
    summary.append(f"✓ Infrastructure features (roads, bridges, built-up areas) are detected")
    summary.append(f"  with an average success rate of {avg_infra_rate * 100:.1f}%.")
    summary.append("")
    summary.append(f"✓ Road detection achieved {metrics['infrastructure_detection']['road'] * 100:.1f}% accuracy.")
    summary.append(f"✓ Bridge detection achieved {metrics['infrastructure_detection']['bridge'] * 100:.1f}% accuracy.")
    summary.append(f"✓ Built-up area detection achieved {metrics['infrastructure_detection']['built_up'] * 100:.1f}% accuracy.")
    summary.append("")
    summary.append("=" * 80)
    summary.append("")

    return "\n".join(summary)


def save_json_report(metrics: Dict, output_path: Path) -> None:
    """Save metrics to JSON file."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ JSON report saved to {output_path}")


def save_txt_report(summary: str, output_path: Path) -> None:
    """Save formatted text report."""
    with open(output_path, "w") as f:
        f.write(summary)
    print(f"✓ Text report saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main evaluation pipeline."""
    print("=" * 80)
    print("SVAMITVA MODEL EVALUATION PIPELINE")
    print("=" * 80)
    print()

    # 1. Verify checkpoint exists
    if not CHECKPOINT_PATH.exists():
        print(f"❌ ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    # 2. Load model
    print(f"Loading model checkpoint from: {CHECKPOINT_PATH.absolute()}")
    print()
    model, config = load_checkpoint(CHECKPOINT_PATH, DEVICE)

    # 3. Create validation dataloader
    print("Creating validation dataset...")
    val_loader = create_val_dataloader(config, BATCH_SIZE, NUM_WORKERS)
    print(f"✓ Validation dataset created")
    print()

    # 4. Run inference
    start_time = time.time()
    accumulator = run_inference(model, val_loader, DEVICE)
    elapsed_time = time.time() - start_time
    print(f"Inference time: {elapsed_time:.2f}s")
    print()

    # 5. Compute metrics
    print("Computing evaluation metrics...")
    metrics = accumulator.compute_metrics()
    print("✓ Metrics computed")
    print()

    # 6. Generate and print summary
    summary = format_metrics_summary(metrics)
    print(summary)

    # 7. Save reports
    print("Saving evaluation reports...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_json_report(metrics, REPORT_JSON)
    save_txt_report(summary, REPORT_TXT)
    print()

    # 8. Final summary
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Overall Success Rate: {metrics['overall_success_rate'] * 100:.1f}%")
    print(f"Average Infrastructure Detection: {metrics['average_infrastructure_detection_rate'] * 100:.1f}%")
    print()
    print(f"Reports saved to:")
    print(f"  - {REPORT_JSON}")
    print(f"  - {REPORT_TXT}")
    print()


if __name__ == "__main__":
    main()
