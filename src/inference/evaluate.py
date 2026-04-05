"""Evaluate SVAMITVA model on validation data with per-class IoU."""

import argparse

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.datasets.unified_dataset import (
    UnifiedMultiClassDataset,
    DEFAULT_SOURCES,
    get_val_transform,
    CLASS_NAMES,
)
from src.models.model_factory import create_model


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load trained model from checkpoint, return (model, config)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = create_model(
        architecture=config.get("architecture", "DeepLabV3Plus"),
        encoder_name=config.get("encoder_name", "resnet50"),
        encoder_weights=None,
        in_channels=3,
        classes=config.get("classes", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
) -> dict[str, float]:
    """Evaluate model with global per-class IoU accumulation."""
    total_intersection = torch.zeros(num_classes, device=device, dtype=torch.float64)
    total_union = torch.zeros(num_classes, device=device, dtype=torch.float64)

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(device_type="cuda"):
            logits = model(images)
        preds = logits.argmax(dim=1)

        for c in range(num_classes):
            p = (preds == c).float()
            t = (masks == c).float()
            inter = (p * t).sum()
            total_intersection[c] += inter
            total_union[c] += p.sum() + t.sum() - inter

    smooth = 1e-6
    results: dict[str, float] = {}
    fg_ious: list[float] = []
    for c in range(1, num_classes):
        iou = (total_intersection[c] + smooth) / (total_union[c] + smooth)
        results[f"iou_class_{c}"] = iou.item()
        if total_union[c].item() > 0:
            fg_ious.append(iou.item())
    results["mean_iou"] = sum(fg_ious) / len(fg_ious) if fg_ious else 0.0
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SVAMITVA model")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    num_classes = config.get("classes", 4)

    print("Loading validation dataset...")
    val_dataset = UnifiedMultiClassDataset(
        sources=DEFAULT_SOURCES,
        split="val",
        transform=get_val_transform(config.get("image_size", 512)),
        patch_size=config.get("image_size", 512),
    )
    dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    print(f"Evaluating on {len(val_dataset)} patches...")
    metrics = evaluate(model, dataloader, device, num_classes)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for c in range(1, num_classes):
        name = CLASS_NAMES.get(c, f"Class {c}")
        iou = metrics.get(f"iou_class_{c}", 0.0)
        print(f"  {name:14s}:  IoU={iou:.4f}")
    print(f"  {'Mean FG IoU':14s}:  {metrics['mean_iou']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
