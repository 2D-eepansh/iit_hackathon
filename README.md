# SVAMITVA Multi-Class Geospatial Feature Segmentation

Semantic segmentation pipeline for detecting **Road**, **Bridge**, and **Built-Up Area** features from high-resolution drone orthomosaics, developed for the **Survey of Villages Abadi and Mapping with Improvised Technology in Village Areas (SVAMITVA)** scheme under the Ministry of Panchayati Raj.

---

## Problem Statement

The SVAMITVA scheme requires automated extraction of rural infrastructure features (roads, bridges, built-up areas) from drone orthomosaics covering Indian villages. Manual digitization across thousands of village orthomosaics is prohibitively expensive and slow. This project automates the process using deep learning-based semantic segmentation on georeferenced imagery.

## Solution Approach

We train a multi-class segmentation model on georeferenced drone orthomosaics with vector (shapefile) annotations. The pipeline:

1. **Ingests** GeoTIFF rasters and shapefile vector annotations from Punjab and Chhattisgarh
2. **Rasterizes** vector geometries to pixel masks on-the-fly per patch
3. **Trains** a DeepLabV3+ model with Focal + Dice loss, EMA checkpointing, and minority-aware sampling
4. **Infers** on full-size test TIFFs via sliding-window with georeferenced output masks

---

## Performance

### Segmentation Metrics (Best Checkpoint -- Epoch 39/80)

| Metric | Value |
|--------|-------|
| Best Validation mIoU (EMA + TTA + Multiscale) | **0.5880** |
| Best Validation mIoU (training, no TTA) | **0.5523** |
| Road IoU | 0.531 |
| Built-Up IoU | 0.645 |

> TTA (test-time augmentation) and multi-scale inference provide a +0.036 mIoU gain over single-scale evaluation.

### Calibrated Pipeline (Phase 5)

Applying a coordinate-descent logit bias search over a weighted ensemble of two checkpoints (epochs 43 + 80) with morphological postprocessing:

| Metric | Baseline | Calibrated | Delta |
|--------|----------|------------|-------|
| Foreground mIoU | 0.4981 | **0.5208** | +0.022 |
| Road IoU | 0.532 | **0.556** | +0.024 |
| Built-Up IoU | 0.673 | **0.696** | +0.023 |

Bias vector: `[0.0, 0.5, 2.0, 0.5, -0.5]` (saved to `outputs/optimal_bias.json`).

### Evaluation Statistics

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Road** | 53.1% | 83.5% | **0.649** |
| **Built-Up Area** | 64.7% | 96.7% | **0.775** |
| **Bridge** | N/A | N/A | N/A* |

\* Bridge has 0 ground-truth pixels in the validation set.

| Summary Metric | Value |
|----------------|-------|
| **Mean Infrastructure F1** | **0.712** |
| Infrastructure Detection Accuracy | 94.4% |
| Validation Pixels Evaluated | ~295M |

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| **Model** | DeepLabV3+ with ResNet-50 encoder (ImageNet pretrained) |
| **Loss** | 0.6 x Focal Loss (gamma=2.0, per-class alpha) + 0.4 x Class-weighted Dice (Road 2x weight) |
| **Optimizer** | AdamW with differential LR (encoder 1e-5, decoder 1e-4) |
| **Scheduler** | ReduceLROnPlateau (patience=5, factor=0.5, monitors val mIoU) |
| **EMA** | Exponential Moving Average (decay=0.99), updated per optimizer step |
| **Training** | Mixed-Precision (AMP) + gradient accumulation (effective batch=16) + gradient clipping |
| **Validation** | Multi-scale inference (1.0x + 0.75x) + morphological road refinement |
| **Reproducibility** | Fixed seeds (seed=42), deterministic cuDNN |

---

## Classes

| ID | Class | Color | Focal Alpha | Dice Weight |
|----|-------|-------|-------------|-------------|
| 0 | Background | Black | 0.1 | -- |
| 1 | Road | Red | 1.0 | 2.0 |
| 2 | Bridge | Blue | 3.0 | 1.0 |
| 3 | Built-Up Area | Yellow | 1.2 | 1.0 |

---

## Dataset

The unified dataset merges two sources of SVAMITVA orthomosaic data:

| Source | Region | CRS | Train TIFFs | Val TIFFs |
|--------|--------|-----|-------------|-----------|
| **PB** | Punjab | EPSG:32643 | 2 | 1 |
| **CG** | Chhattisgarh | EPSG:32644 | 2 (+2 corrupt, auto-skipped) | 1 |

Key data pipeline features:
- **TIFF-level train/val split** (no spatial leakage)
- **Windowed raster reading** via rasterio (handles 235K x 120K TIFFs without loading full images)
- **On-the-fly CRS reprojection** of shapefile annotations to match each TIFF's native CRS
- **On-the-fly rasterization** of vector geometries to pixel masks per patch
- **Bridge copy-paste augmentation** (30 cached patches) for extreme bridge scarcity
- **Deterministic validation grid** (500 patches) for stable metrics

---

## Project Structure

```
train.py                            # Main training entry point (CONFIG dict)
test_inference.py                   # Full-TIFF sliding-window inference
evaluate_model_statistics.py        # Stakeholder evaluation report
visualize_training.py               # 2D + 3D training curve visualization
audit_model.py                      # Model audit: raw per-class metrics, confusion matrix
bias_search.py                      # Coordinate-descent logit bias calibration
run_calibrated_eval.py              # Evaluate calibrated pipeline vs baseline
submission_document.md              # Hackathon submission document

src/
  datasets/
    unified_dataset.py              # Unified PB+CG dataset with augmentations
  losses/
    multiclass_loss.py              # Focal + class-weighted Dice composite loss
  models/
    model_factory.py                # smp model factory (DeepLabV3+)
  training/
    train_one_epoch.py              # Training loop + validation with TTA
  inference/
    calibrated_engine.py            # Production inference: ensemble + bias + TTA + postproc
    village_stats.py                # Per-village statistics with physical units
    evaluate.py                     # Standalone CLI evaluation
    export_model.py                 # Checkpoint cleanup + ONNX export
  postprocessing.py                 # Road gap-fill + bridge recovery morphological ops

demo_ui/
  app.py                            # Streamlit demo interface
  inference_wrapper.py              # Model loading bridge for demo
  assets/
    sample_test_patch.png           # Sample input for demo

docs/                               # System overview, architecture, training, evaluation
requirements.txt                    # Python dependencies
outputs/                            # Checkpoints, plots, reports (.gitignored)
data/                               # Raw TIFFs + shapefiles (.gitignored)
```

---

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

Place the SVAMITVA orthomosaic TIFFs and shapefiles under `data/`:

```
data/Raz/
  PB_training_dataSet_shp_file/       # Punjab TIFFs + shapefiles
    shp-file/
  Training_dataSet_2/                 # Chhattisgarh TIFFs (set 2)
  Training_dataSet_3/                 # Chhattisgarh TIFFs (set 3)
  CG_shp-file/
    shp-file/                         # Chhattisgarh shapefiles
```

### Training

```bash
python train.py
```

Configuration is in the `CONFIG` dict at the top of `train.py`:
- 80 epochs, batch size 4 with 4x gradient accumulation (effective batch 16)
- 768x768 patches, 150 patches per image per epoch
- EMA decay 0.99, ReduceLROnPlateau scheduler

### Evaluation

```bash
# Stakeholder evaluation report
python evaluate_model_statistics.py

# Training curve visualization
python visualize_training.py

# Raw model audit (no TTA/postprocessing)
python audit_model.py

# Calibrated pipeline evaluation (ensemble + bias + postprocessing)
python run_calibrated_eval.py
```

### Logit Bias Calibration

To regenerate the optimal per-class bias vector from scratch:

```bash
python bias_search.py
```

This runs a coordinate-descent search over validation logits and writes the result to `outputs/optimal_bias.json`. The calibrated inference engine (`src/inference/calibrated_engine.py`) reads this file automatically.

### Inference on Test TIFFs

```bash
python test_inference.py
```

Outputs georeferenced predicted masks (GeoTIFF) and visualization PNGs to `outputs/test_predictions_live_demo/`.

### Demo UI

```bash
streamlit run demo_ui/app.py
```

Upload aerial images and get real-time segmentation results. See [docs/demo_instructions.md](docs/demo_instructions.md) for details.

### Export Model

```bash
python src/inference/export_model.py --checkpoint outputs/checkpoints/best_model.pth
```

---

## Training Details

- **Augmentations**: Horizontal/vertical flip, random rotate 90, affine (rotation +/-45 deg, scale 0.9-1.1), Gaussian noise/blur, brightness/contrast jitter
- **Positive sampling**: 90% of train patches centered on feature centroids
- **Bridge handling**: Copy-paste augmentation with 30 cached bridge patches
- **EMA strategy**: Shadow weights updated per optimizer step; actual model used for validation; EMA weights saved as best checkpoint
- **Corrupt TIFF handling**: KUTRU and SAMLUR TIFFs are automatically skipped

## Known Limitations

1. **Bridge scarcity** -- only ~15 bridge features across all TIFFs; no bridge pixels in validation set
2. **Geographic bias** -- Punjab and Chhattisgarh have different terrain/architecture styles
3. **Corrupt TIFFs** -- 2 CG TIFFs have corrupted headers and are auto-skipped
4. **Built-Up oscillation** -- occasional BU class collapse during training (~6/80 epochs), mitigated by EMA checkpointing

---

## Documentation

| Document | Description |
|----------|-------------|
| [System Overview](docs/SYSTEM_OVERVIEW.md) | Pipeline stages, key metrics, entry points |
| [Architecture](docs/ARCHITECTURE.md) | Model design, loss function, training/inference pipelines |
| [Training Guide](docs/TRAINING.md) | Configuration reference, data split, augmentation, checkpoints |
| [Evaluation Guide](docs/EVALUATION.md) | Evaluation scripts, metrics reference, validation strategy |
| [Demo Instructions](docs/demo_instructions.md) | How to run the Streamlit demo UI |

---

## Technology Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, segmentation-models-pytorch, timm |
| Geospatial | rasterio, geopandas, shapely, pyproj, fiona |
| Augmentation | Albumentations, OpenCV |
| Visualization | Matplotlib, Streamlit |
| Hardware | NVIDIA RTX 5070, CUDA 12.8 |

## License

Developed for the AI/ML Hackathon -- Ministry of Panchayati Raj, powered by Geo-Intel Lab, IIT Tirupati Navavishkar I-Hub Foundation.
