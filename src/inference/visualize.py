"""Visualize predictions on images."""

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
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path: str, image_size: int = 512) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Resize
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # To tensor
    image = torch.from_numpy(image.transpose(2, 0, 1))
    
    return image, original_image


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Get probability map."""
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with autocast(device_type="cuda"):
        logits = model(image_tensor)
    
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs


def visualize_prediction(
    image_path: str,
    checkpoint_path: str,
    output_path: str | None = None,
    threshold: float = 0.5,
) -> None:
    """Visualize prediction with overlay."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model...")
    model = load_model(checkpoint_path, device)
    
    print(f"Processing {image_path}...")
    image, original_image = preprocess_image(image_path)
    h_orig, w_orig = original_image.shape[:2]
    
    # Predict
    probs = predict_probs(model, image, device)
    
    # Resize to original
    probs_resized = cv2.resize(probs, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    # Create mask
    mask = (probs_resized > threshold).astype(np.uint8) * 255
    
    # Create visualization
    fig = np.zeros((h_orig, w_orig * 3, 3), dtype=np.uint8)
    
    # Original image
    fig[:, :w_orig] = original_image
    
    # Probability heatmap
    prob_heatmap = (probs_resized * 255).astype(np.uint8)
    prob_heatmap_color = cv2.applyColorMap(prob_heatmap, cv2.COLORMAP_JET)
    fig[:, w_orig:w_orig*2] = prob_heatmap_color
    
    # Overlay
    overlay = original_image.copy()
    overlay[mask == 255] = [0, 255, 0]  # Green for buildings
    fig[:, w_orig*2:] = cv2.addWeighted(original_image, 0.6, overlay, 0.4, 0)
    
    # Save
    if output_path is None:
        output_path = Path(image_path).stem + "_visualization.png"
    
    # Convert RGB to BGR for saving
    fig_bgr = cv2.cvtColor(fig, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, fig_bgr)
    print(f"✓ Saved visualization to {output_path}")
    
    # Print stats
    building_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    building_ratio = building_pixels / total_pixels * 100
    
    print(f"\nPrediction Stats:")
    print(f"  Building pixels: {building_pixels}")
    print(f"  Building ratio: {building_ratio:.2f}%")
    print(f"  Confidence range: [{probs_resized.min():.3f}, {probs_resized.max():.3f}]")


def main() -> None:
    """Main visualization entry point."""
    parser = argparse.ArgumentParser(description="Visualize building predictions")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="Path to checkpoint",
    )
    parser.add_argument("--output", type=str, default=None, help="Output visualization path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    
    args = parser.parse_args()
    
    visualize_prediction(args.image, args.checkpoint, args.output, args.threshold)


if __name__ == "__main__":
    main()
