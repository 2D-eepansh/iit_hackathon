# Building Segmentation - Phase 1: Foundation Training Pipeline

Production-grade semantic segmentation for building detection in aerial imagery.

## Architecture

- **Model**: U-Net with EfficientNet-B4 encoder (segmentation_models_pytorch)
- **Loss**: Composite BCE + Dice Loss
- **Optimizer**: AdamW with differential learning rates
- **Scheduler**: CosineAnnealingLR
- **Training**: Mixed Precision (AMP) with gradient clipping

## Project Structure

```
iit_hackathon/
├── configs/                      # Configuration files
├── data/
│   ├── raw/
│   │   └── AerialImageDataset/  # Place dataset here
│   │       ├── train/
│   │       │   ├── images/      # Training images (.tif/.png)
│   │       │   └── gt/          # Ground truth masks (255=building)
│   │       └── test/
│   ├── processed/               # Preprocessed data
│   └── splits/                  # Train/val split indices
├── src/
│   ├── datasets/
│   │   └── building_dataset.py  # Dataset loader with albumentations
│   ├── models/
│   │   └── model_factory.py     # Model instantiation
│   ├── losses/
│   │   └── composite_loss.py    # BCE + Dice loss
│   ├── training/
│   │   └── train_one_epoch.py   # Training/validation logic
│   └── inference/               # Inference scripts (Phase 2)
├── outputs/
│   └── checkpoints/             # Model checkpoints
├── notebooks/                    # Jupyter notebooks
├── train.py                      # Main training entry point
└── requirements.txt              # Python dependencies
```

## Dataset Format

Inria-style aerial imagery dataset:
- **Images**: RGB aerial photos in `train/images/`
- **Masks**: Binary masks in `train/gt/` (0=background, 255=building)
- Supported formats: `.tif`, `.png`

## Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (already done if requirements.txt used)
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

### Configuration (inline in train.py)

```python
data_root = "data/raw/AerialImageDataset"
image_size = 512
batch_size = 8
num_epochs = 5
learning_rate = 1e-4
encoder_lr = 1e-5
```

## Training Features

- ✅ Automatic train/val split (80/20)
- ✅ Data augmentation (flip, rotate, color jitter)
- ✅ Mixed precision training (AMP)
- ✅ Gradient clipping
- ✅ Best model checkpointing based on IoU
- ✅ ImageNet normalization
- ✅ GPU optimization (pin_memory, non_blocking transfers)

## Metrics

- **IoU** (Intersection over Union)
- **Dice Coefficient**
- **BCE + Dice Loss**

## Output

Checkpoints saved to `outputs/checkpoints/`:
- `best_model.pth` - Best validation IoU
- `latest_model.pth` - Latest epoch

## Hardware Requirements

- GPU: NVIDIA RTX 5070 (8GB VRAM) or equivalent
- CUDA 13+
- Ubuntu (WSL2 compatible)

## Phase 1 Completion Checklist

- [x] Dataset loader with albumentations
- [x] U-Net with EfficientNet-B4 encoder
- [x] Composite BCE + Dice loss
- [x] Training loop with AMP
- [x] Validation with IoU/Dice metrics
- [x] Checkpoint saving
- [x] Modular code structure
- [x] Type hints and docstrings

## Next Steps (Future Phases)

- Phase 2: Advanced training (patch sampling, TTA, boundary loss)
- Phase 3: Inference pipeline with tiling
- Phase 4: Experiment tracking and hyperparameter tuning
