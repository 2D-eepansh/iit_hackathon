"""Evaluate model on test set."""

import argparse
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.datasets.building_dataset import BuildingDataset, get_val_transform
from src.models.model_factory import create_model


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get("config", {})
    model = create_model(
        architecture=config.get("architecture", "Unet"),
        encoder_name=config.get("encoder_name", "efficientnet-b4"),
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate on dataset."""
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0
    
    print("Evaluating...")
    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Predict
        with autocast(device_type="cuda"):
            logits = model(images)
        
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Compute metrics
        preds_flat = preds.flatten(1)
        masks_flat = masks.flatten(1)
        
        intersection = (preds_flat * masks_flat).sum(dim=1)
        union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1) - intersection
        cardinality = preds_flat.sum(dim=1) + masks_flat.sum(dim=1)
        
        iou = ((intersection + 1e-6) / (union + 1e-6)).mean()
        dice = ((2.0 * intersection + 1e-6) / (cardinality + 1e-6)).mean()
        
        total_iou += iou.item()
        total_dice += dice.item()
        num_samples += 1
    
    return {
        "iou": total_iou / num_samples,
        "dice": total_dice / num_samples,
    }


def main() -> None:
    """Main evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="Path to checkpoint",
    )
    parser.add_argument("--data-root", type=str, default="data/raw/AerialImageDataset", help="Data root")
    parser.add_argument("--split", type=str, default="test", help="Data split (train/test)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    
    print(f"Loading {args.split} dataset...")
    dataset = BuildingDataset(
        data_root=args.data_root,
        split=args.split,
        transform=get_val_transform(),
        use_processed=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    metrics = evaluate(model, dataloader, device, args.threshold)
    
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS ({args.split.upper()})")
    print("=" * 80)
    print(f"IoU:   {metrics['iou']:.4f}")
    print(f"Dice:  {metrics['dice']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
