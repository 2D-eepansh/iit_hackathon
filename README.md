# SVAMITVA Multi-Class Geospatial Feature Segmentation

Semantic segmentation pipeline for detecting **Road**, **Bridge**, and **Built-Up Area** features from high-resolution drone orthomosaics, developed for the **Survey of Villages Abadi and Mapping with Improvised Technology in Village Areas (SVAMITVA)** scheme.

## Problem Statement

The SVAMITVA scheme requires automated extraction of rural infrastructure features (roads, bridges, built-up areas) from high-resolution drone orthomosaics covering Indian villages. Manual digitization of these features across thousands of village orthomosaics is prohibitively expensive and slow. This project automates the process using deep learning-based semantic segmentation.

## Solution Approach

We train a multi-class segmentation model on georeferenced drone orthomosaics with vector (shapefile) annotations. The pipeline performs:

1. **On-the-fly rasterization** of vector geometries to pixel masks per patch
2. **Windowed raster reading** via rasterio (memory-safe, handles 235K x 120K TIFFs)
3. **Minority-aware centroid sampling** (90% of train patches centered on feature centroids)
4. **Sliding-window inference** on full-resolution test TIFFs with georeferenced output

## Performance

| Metric | Value |
|--------|-------|
| Best Validation mIoU (EMA + TTA + Multiscale) | **0.5880** |
| Best Validation mIoU (training, no TTA) | **0.5523** |
| Road IoU | 0.531 |
| Built-Up IoU | 0.645 |

> Results are from the best checkpoint at epoch 39/80. TTA (test-time augmentation) and multi-scale inference provide a +0.036 mIoU gain over single-scale evaluation.

## Model Architecture

| Component | Detail |
|-----------|--------|
| **Model** | DeepLabV3+ with ResNet-50 encoder (ImageNet pretrained) |
| **Loss** | 0.6 x Focal Loss (gamma=2.0, per-class alpha) + 0.4 x Class-weighted Dice (Road 2x weight) |
| **Optimizer** | AdamW with differential LR (encoder 1e-5, decoder 1e-4) |
| **Scheduler** | ReduceLROnPlateau (patience=5, factor=0.5, monitors val mIoU) |
| **EMA** | Exponential Moving Average (decay=0.99), updated per optimizer step |
| **Training** | Mixed-Precision (AMP) + gradient accumulation (effective batch=16) + gradient clipping (max norm 1.0) |
| **Validation** | Multi-scale inference (1.0x + 0.75x) + morphological road refinement |
| **Reproducibility** | Fixed seeds (seed=42), deterministic cuDNN |

## Classes

| ID | Class | Color (visualizations) | Focal Alpha | Dice Weight |
|----|-------|------------------------|-------------|-------------|
| 0 | Background | Black | 0.1 | -- |
| 1 | Road | Red | 1.0 | 2.0 |
| 2 | Bridge | Blue | 3.0 | 1.0 |
| 3 | Built-Up Area | Yellow | 1.2 | 1.0 |

## Dataset

The unified dataset merges two sources of SVAMITVA orthomosaic data:

| Source | Region | CRS | Train TIFFs | Val TIFFs | Key Features |
|--------|--------|-----|-------------|-----------|--------------|
| **PB** | Punjab | EPSG:32643 | 2 | 1 | Road, Built-Up (high density) |
| **CG** | Chhattisgarh | EPSG:32644 | 2 (+2 corrupt, auto-skipped) | 1 | Road, Bridge, Built-Up |

Key data pipeline features:
- **TIFF-level train/val split** (no spatial leakage between train and val)
- **Windowed raster reading** via rasterio (no full TIFF loading)
- **On-the-fly CRS reprojection** of SHP annotations to match each TIFF's native CRS
- **On-the-fly rasterization** of vector geometries to pixel masks per patch
- **Bridge copy-paste augmentation** (30 cached patches) to address extreme bridge scarcity
- **Deterministic validation grid** (500 patches) for stable val metrics

## Project Structure

```
train.py                            # Main training entry point (CONFIG dict + training loop)
test_inference.py                   # Full-TIFF sliding-window inference on test TIFFs
evaluate_model_statistics.py        # Post-training evaluation report for stakeholders
visualize_training.py               # 2D + 3D training curve visualization

src/
  datasets/
    unified_dataset.py              # Unified PB+CG dataset, transforms, bridge copy-paste
  losses/
    multiclass_loss.py              # Focal + class-weighted Dice composite loss
  models/
    model_factory.py                # smp model factory (DeepLabV3+, gradient checkpointing)
  training/
    train_one_epoch.py              # Training loop + multi-class validation with TTA
  inference/
    evaluate.py                     # Standalone CLI evaluation script
    export_model.py                 # Checkpoint cleanup + ONNX export

requirements.txt                    # Python dependencies
outputs/                            # Checkpoints, plots, reports (.gitignored)
data/                               # Raw TIFFs + shapefiles (.gitignored)
```

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 5070, CUDA 12.8)
- ~16 GB GPU memory recommended

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation

Place the SVAMITVA orthomosaic TIFFs and shapefiles under `data/` following this structure:

```
data/Raz/
  PB_training_dataSet_shp_file/       # Punjab TIFFs
    shp-file/                         # Punjab shapefiles
  Training_dataSet_2/                 # Chhattisgarh TIFFs (set 2)
  Training_dataSet_3/                 # Chhattisgarh TIFFs (set 3)
  CG_shp-file/
    shp-file/                         # Chhattisgarh shapefiles
```

### Training

```bash
python train.py
```

Training configuration is controlled by the `CONFIG` dict in `train.py`. Key settings:
- 80 epochs, batch size 4 with 4x gradient accumulation (effective batch 16)
- 768x768 patches, 150 patches per image per epoch
- EMA decay 0.99, ReduceLROnPlateau scheduler

### Evaluation

```bash
# Generate stakeholder evaluation report
python evaluate_model_statistics.py

# Visualize training curves
python visualize_training.py
```

### Inference on Test TIFFs

```bash
python test_inference.py
```

Outputs georeferenced predicted masks (GeoTIFF) and visualization PNGs to `outputs/test_predictions_live_demo/`.

### Export Model

```bash
python src/inference/export_model.py --checkpoint outputs/checkpoints/best_model.pth
```

## Training Details

- **Augmentations**: Horizontal/vertical flip, random rotate 90, affine (rotation +/-45 deg, scale 0.85-1.15), color jitter, Gaussian blur, CLAHE
- **Positive sampling**: 90% of train patches are centered on feature centroids (roads, bridges, built-up)
- **Bridge handling**: Copy-paste augmentation with 30 cached bridge patches (bridge has ~0 pixels in val set)
- **EMA strategy**: Shadow weights updated per optimizer step; actual model used for validation (clean LR scheduler signal); EMA weights saved as best checkpoint
- **Corrupt TIFF handling**: KUTRU and SAMLUR TIFFs have corrupted headers and are automatically skipped

## Known Limitations

1. **Bridge scarcity** -- only ~15 bridge features across all TIFFs; no bridge pixels in validation set
2. **Geographic bias** -- Punjab and Chhattisgarh have different terrain/architecture styles
3. **Corrupt TIFFs** -- 2 CG TIFFs (KUTRU, SAMLUR) have corrupted headers and are auto-skipped
4. **Built-Up collapse** -- occasional BU class collapse during training (~6/80 epochs) due to noisy validation, mitigated by EMA checkpointing

## License

This project was developed for the IIT Hackathon / SVAMITVA competition.
