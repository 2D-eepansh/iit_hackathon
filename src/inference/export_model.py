"""Export model for deployment."""

import argparse
from pathlib import Path

import torch
import torch.onnx

from src.models.model_factory import create_model


def export_checkpoint(
    checkpoint_path: str,
    output_dir: str = "outputs/models",
) -> None:
    """Export to clean checkpoint format."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    
    # Create output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state only
    weight_path = output_dir / "model_weights.pth"
    torch.save(checkpoint["model_state_dict"], weight_path)
    print(f"✓ Saved weights to {weight_path}")
    
    # Save config
    import json
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        # Filter serializable config
        serializable_config = {}
        for k, v in config.items():
            if isinstance(v, (str, int, float, bool, list, dict)):
                serializable_config[k] = v
        json.dump(serializable_config, f, indent=2)
    print(f"✓ Saved config to {config_path}")
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    if "metrics" in checkpoint:
        with open(metrics_path, "w") as f:
            json.dump(checkpoint["metrics"], f, indent=2)
        print(f"✓ Saved metrics to {metrics_path}")
    
    print(f"\nExport complete!")
    print(f"  Output dir: {output_dir}")
    print(f"  Best IoU: {checkpoint.get('best_iou', 'N/A'):.4f}")


def export_onnx(
    checkpoint_path: str,
    output_path: str = "outputs/models/model.onnx",
    image_size: int = 512,
) -> None:
    """Export to ONNX format."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    
    # Create model
    model = create_model(
        architecture=config.get("architecture", "Unet"),
        encoder_name=config.get("encoder_name", "efficientnet-b4"),
        encoder_weights=None,
        in_channels=3,
        classes=config.get("classes", 4),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["segmentation"],
        opset_version=14,
        do_constant_folding=True,
    )
    
    print(f"✓ Exported ONNX to {output_path}")


def main() -> None:
    """Main export."""
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="Path to checkpoint",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/models", help="Output directory")
    parser.add_argument("--onnx", action="store_true", help="Also export to ONNX")
    
    args = parser.parse_args()
    
    export_checkpoint(args.checkpoint, args.output_dir)
    
    if args.onnx:
        export_onnx(args.checkpoint, f"{args.output_dir}/model.onnx")


if __name__ == "__main__":
    main()
