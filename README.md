# SVAMITVA — Multi-Class Geospatial Feature Segmentation

Semantic segmentation pipeline for detecting **Road**, **Bridge**, and **Built-Up Area**
features from high-resolution drone orthomosaics, developed for the
**Survey of Villages Abadi and Mapping with Improvised Technology in Village Areas (SVAMITVA)** scheme.

## Classes

| ID | Class | Colour (in visualisations) |
|----|-------|---------------------------|
| 0 | Background | Black |
| 1 | Road | Red |
| 2 | Bridge | Blue |
| 3 | Built-Up Area | Yellow |

## Architecture

| Component | Detail |
|-----------|--------|
| **Backbone** | U-Net with EfficientNet-B4 encoder (ImageNet pretrained) |
| **Loss** | Focal Loss (γ=2.0, per-class α) + Soft Dice, weighted 0.5 / 0.5 |
| **Optimiser** | AdamW with differential learning rates (encoder 3e-5, decoder 1e-4) |
| **Scheduler** | ReduceLROnPlateau (patience=3, factor=0.5, monitors val mIoU) |
| **Training** | Mixed-Precision (AMP) with gradient clipping (max norm 1.0) |
| **Reproducibility** | Fixed seeds (`torch`, `numpy`, `random`, CUDA), deterministic cuDNN |

## Dataset

The unified dataset merges **two sources** of SVAMITVA orthomosaic data:

| Source | Region | CRS | TIFFs | Key Features |
|--------|--------|-----|-------|--------------|
| **PB** | Punjab | EPSG:32643 | 3 | Road, Built-Up (high density) |
| **CG** | Chhattisgarh | EPSG:32644 | 3 valid | Road, Bridge, Built-Up |

### Data Handling

- **Windowed raster reading** via rasterio — memory-safe, no full TIFF loading
- **On-the-fly CRS reprojection** of SHP annotations to match each TIFF's native CRS
- **On-the-fly rasterization** of vector geometries to pixel masks per patch
- **Minority-aware centroid sampling** — 50% of train patches centred on feature centroids

### Train / Val Split

Split is at the **TIFF level** (no spatial leakage):

| Split | TIFFs |
|-------|-------|
| Train | `PINDORI MAYA SINGH-TUGALWAL_28456`, `TIMMOWAL_37695`, `BADETUMNAR_…`, `MURDANDA_…` |
| Val | `28996_NADALA`, `NAGUL_…` |

Validation patches are **deterministic random samples** (seeded by index) spread across
the entire TIFF extent — not limited to a single centre crop.

## Project Structure

```
├── train.py                           # Main training entry point
├── test_inference.py                  # Multi-class inference on test TIFFs
├── validate_svamvitva_dataset.py      # Pre-training dataset health check
├── audit_dataset.py                   # Dataset structure audit
├── analyze_training.py                # Post-training checkpoint analysis
├── src/
│   ├── datasets/
│   │   ├── unified_dataset.py         # Unified PB + CG dataset loader
│   │   └── multiclass_dataset.py      # Single-source dataset (shared transforms)
│   ├── losses/
│   │   └── multiclass_loss.py         # Focal + Dice composite loss
│   ├── models/
│   │   └── model_factory.py           # smp model factory
│   ├── training/
│   │   └── train_one_epoch.py         # Train / validate loops with per-class IoU
│   └── inference/
│       ├── evaluate.py                # Standalone evaluation script
│       ├── predict.py                 # Single-image inference
│       ├── visualize.py               # Overlay visualisation
│       └── export_model.py            # ONNX / weight export
├── _legacy/                           # Archived Phase-1 files (not used)
├── outputs/checkpoints/               # Saved model checkpoints (.gitignored)
├── data/                              # Raw + processed data (.gitignored)
└── requirements.txt
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# PyTorch — install the version matching your CUDA:
# https://pytorch.org/get-started/locally/
```

## Usage

### Training

```bash
python train.py
```

Configuration is centralised in the `CONFIG` dict inside `train.py`.
Key settings: `image_size=512`, `batch_size=4`, `num_epochs=50`, `seed=42`.

### Inference on Test TIFFs

```bash
python test_inference.py
```

Outputs GeoTIFF masks + PNG visualisations to `outputs/test_predictions_live_demo/`.

### Dataset Validation

```bash
python validate_svamvitva_dataset.py
```

## Evaluation Methodology

- **Per-class IoU** accumulated globally across all validation patches (not batch-averaged)
- **Macro FG mIoU** computed only over classes with >0 ground-truth pixels
  - Classes absent from the validation set are explicitly excluded and reported
- **Per-class Dice** reported alongside IoU for completeness

## Class Weights (Focal Loss α)

| Class | Weight | Rationale |
|-------|--------|-----------|
| Background | 0.1 | Down-weighted (dominant class) |
| Road | 1.0 | Baseline |
| Bridge | 3.0 | Rare class (~15 features total) |
| Built-Up | 1.2 | Slightly boosted |

## Known Dataset Limitations

1. **Bridge scarcity** — only ~15 bridge features across all TIFFs; per-class IoU may be volatile
2. **Geographic bias** — PB (Punjab) and CG (Chhattisgarh) have different terrain/architecture styles
3. **Corrupt TIFFs** — 2 CG TIFFs (KUTRU, SAMLUR) have corrupted headers and are auto-skipped
4. **Validation coverage** — 2 val TIFFs provide limited geographic diversity

## License

This project was developed for the IIT Hackathon / SVAMITVA competition.
