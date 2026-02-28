"""Analyze training results and metrics."""

import json
from pathlib import Path

import torch


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint and extract metrics."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint


def print_checkpoint_info(checkpoint: dict) -> None:
    """Print checkpoint information."""
    print("\n" + "=" * 80)
    print("CHECKPOINT INFORMATION")
    print("=" * 80)
    
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best IoU: {checkpoint.get('best_iou', 'N/A'):.4f}")
    
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        print(f"\nMetrics (Last Epoch):")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
    
    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"\nTraining Configuration:")
        print(f"  Image Size: {config.get('image_size', 'N/A')}")
        print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"  Encoder LR: {config.get('encoder_lr', 'N/A')}")
        print(f"  Num Epochs: {config.get('num_epochs', 'N/A')}")
        print(f"  Architecture: {config.get('architecture', 'N/A')}")
        print(f"  Encoder: {config.get('encoder_name', 'N/A')}")
    
    print("=" * 80 + "\n")


def extract_training_history() -> None:
    """Extract and display training history from checkpoint files."""
    checkpoint_dir = Path("outputs/checkpoints")
    
    if not checkpoint_dir.exists():
        print("✗ No checkpoints found in outputs/checkpoints/")
        return
    
    # Load latest checkpoint
    latest_path = checkpoint_dir / "latest_model.pth"
    best_path = checkpoint_dir / "best_model.pth"
    
    if not latest_path.exists():
        print("✗ latest_model.pth not found")
        return
    
    print("\n" + "=" * 80)
    print("TRAINING ANALYSIS")
    print("=" * 80)
    
    # Load latest
    latest = load_checkpoint(str(latest_path))
    print_checkpoint_info(latest)
    
    # Compare with best
    if best_path.exists():
        best = load_checkpoint(str(best_path))
        print("\n" + "=" * 80)
        print("BEST MODEL")
        print("=" * 80)
        print(f"Epoch: {best.get('epoch', 'N/A')}")
        print(f"Best IoU: {best.get('best_iou', 'N/A'):.4f}")
        if "metrics" in best:
            metrics = best["metrics"]
            print(f"\nMetrics at Best:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
        print("=" * 80 + "\n")


def print_summary() -> None:
    """Print training summary."""
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. CREATE INFERENCE PIPELINE
   python src/inference/predict.py --image <path> --checkpoint outputs/checkpoints/best_model.pth

2. EVALUATE ON TEST SET
   python src/inference/evaluate.py --checkpoint outputs/checkpoints/best_model.pth

3. VISUALIZE PREDICTIONS
   python src/inference/visualize.py --checkpoint outputs/checkpoints/best_model.pth

4. EXPORT MODEL
   python src/inference/export_model.py --checkpoint outputs/checkpoints/best_model.pth

5. FINE-TUNE OR RETRAIN
   - Adjust hyperparameters in train.py CONFIG
   - Run python train.py again
    """)
    print("=" * 80 + "\n")


def main() -> None:
    """Main analysis function."""
    extract_training_history()
    print_summary()


if __name__ == "__main__":
    main()
