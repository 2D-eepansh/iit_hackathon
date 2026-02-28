"""Inference pipeline for aerial building segmentation."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast

from src.models.model_factory import create_model


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get("config", {})
    model = create_model(
        architecture=config.get("architecture", "Unet"),
        encoder_name=config.get("encoder_name", "efficientnet-b4"),
        encoder_weights=None,  # Don't load pretrained
        in_channels=3,
        classes=1,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path: str, image_size: int = 512) -> tuple[torch.Tensor, tuple]:
    """Load and preprocess image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    
    # Resize
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # To tensor (C, H, W)
    image = torch.from_numpy(image.transpose(2, 0, 1))
    
    return image, original_shape


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run inference on image tensor."""
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dim
    
    with autocast(device_type="cuda"):
        logits = model(image_tensor)
    
    # Sigmoid + threshold
    probs = torch.sigmoid(logits)
    mask = (probs > threshold).squeeze().cpu().numpy().astype(np.uint8) * 255
    
    return mask


def predict_single(
    image_path: str,
    checkpoint_path: str,
    output_path: str | None = None,
    threshold: float = 0.5,
) -> None:
    """Predict segmentation for single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    
    # Preprocess
    print(f"Processing {image_path}...")
    image, original_shape = preprocess_image(image_path)
    
    # Predict
    mask = predict(model, image, device, threshold)
    
    # Resize back to original
    mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Save
    if output_path is None:
        output_path = Path(image_path).stem + "_mask.png"
    
    cv2.imwrite(output_path, mask_resized)
    print(f"✓ Saved mask to {output_path}")
    
    # Stats
    building_pixels = np.count_nonzero(mask_resized)
    total_pixels = mask_resized.size
    building_ratio = building_pixels / total_pixels * 100
    
    print(f"\nPrediction Stats:")
    print(f"  Building pixels: {building_pixels}")
    print(f"  Building ratio: {building_ratio:.2f}%")


def main() -> None:
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="Predict building segmentation")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="Path to checkpoint",
    )
    parser.add_argument("--output", type=str, default=None, help="Output mask path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    
    args = parser.parse_args()
    
    predict_single(args.image, args.checkpoint, args.output, args.threshold)


if __name__ == "__main__":
    main()
