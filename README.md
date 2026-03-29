# SVAMITVA — Multi-Class Geospatial Feature Segmentation

Semantic segmentation pipeline for detecting **Road**, **Bridge**, and **Built-Up Area**
features from high-resolution drone orthomosaics, developed for the
**Survey of Villages Abadi and Mapping with Improvised Technology in Village Areas (SVAMITVA)** scheme.

## Performance

| Metric | Value |
|--------|-------|
| Pixel Accuracy | **93.9%** |
| Infrastructure Accuracy | **92.5%** |
| Road F1 / Recall | 0.612 / 77.0% |
| Built-Up F1 / Recall | 0.834 / 94.4% |
| Best Foreground mIoU | **0.6012** |

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
| **Model** | DeepLabV3+ with ResNet-50 encoder (ImageNet pretrained) |
| **Loss** | Focal Loss (gamma=2.0, per-class alpha) + Soft Dice, weighted 0.5/0.5 |
| **Optimiser** | AdamW with differential LR (encoder 1e-5, decoder 1e-4) |
| **Scheduler** | ReduceLROnPlateau (patience=3, factor=0.5, monitors val mIoU) |
| **Training** | Mixed-Precision (AMP) with gradient clipping (max norm 1.0) |
| **Validation** | Multi-scale inference (1.0x + 0.75x) + morphological road refinement |
| **Reproducibility** | Fixed seeds (seed=42), deterministic cuDNN |

## Dataset

The unified dataset merges **two sources** of SVAMITVA orthomosaic data:

| Source | Region | CRS | TIFFs | Key Features |
|--------|--------|-----|-------|--------------|
| **PB** | Punjab | EPSG:32643 | 3 | Road, Built-Up (high density) |
| **CG** | Chhattisgarh | EPSG:32644 | 3 valid | Road, Bridge, Built-Up |

Key data pipeline features:
- **Windowed raster reading** via rasterio (memory-safe, no full TIFF loading)
- **On-the-fly CRS reprojection** of SHP annotations to match each TIFF's native CRS
- **On-the-fly rasterization** of vector geometries to pixel masks per patch
- **Minority-aware centroid sampling** (50% of train patches centred on feature centroids)
- **TIFF-level train/val split** (no spatial leakage)

## Project Structure

```
train.py                            # Main training entry point (CONFIG + training loop)
evaluate_model_statistics.py        # Post-training evaluation + stakeholder reports
test_inference.py                   # Full-TIFF sliding-window inference

src/
  datasets/
    unified_dataset.py              # Unified PB+CG dataset + augmentation transforms
  losses/
    multiclass_loss.py              # Focal + Dice composite loss
  models/
    model_factory.py                # smp model factory (DeepLabV3+, ResNet-50)
  training/
    train_one_epoch.py              # Training loop + multi-class validation
  inference/
    evaluate.py                     # Standalone CLI evaluation
    export_model.py                 # Checkpoint cleanup + ONNX export

scripts/                            # Utility scripts
  validate_svamvitva_dataset.py     # Pre-training dataset health check
  audit_dataset.py                  # Dataset structure audit
  analyze_training.py               # Checkpoint metadata inspector

configs/
  train_config.yaml                 # Reference training configuration

docs/                               # Detailed documentation
  SYSTEM_OVERVIEW.md                # What the system does
  ARCHITECTURE.md                   # Model + pipeline architecture
  DATASET.md                        # Data sources, format, splits
  TRAINING.md                       # Training guide + configuration
  EVALUATION.md                     # Metrics + inference guide
  IMPROVEMENTS.md                   # Improvement roadmap

_legacy/                            # Archived Phase-1 files (binary era)
outputs/checkpoints/                # Model checkpoints (.gitignored)
data/                               # Raw data (.gitignored)
```

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Validate dataset (optional)
python scripts/validate_svamvitva_dataset.py

# Train
python train.py

# Evaluate
python evaluate_model_statistics.py

# Inference on test TIFFs
python test_inference.py
```

## Class Weights (Focal Loss alpha)

| Class | Weight | Rationale |
|-------|--------|-----------|
| Background | 0.1 | Down-weighted (dominant class) |
| Road | 1.0 | Baseline |
| Bridge | 3.0 | Rare class (~15 features total) |
| Built-Up | 1.2 | Slightly boosted |

## Known Limitations

1. **Bridge scarcity** — only ~15 bridge features across all TIFFs; no bridge pixels in validation set
2. **Geographic bias** — PB (Punjab) and CG (Chhattisgarh) have different terrain/architecture styles
3. **Corrupt TIFFs** — 2 CG TIFFs (KUTRU, SAMLUR) have corrupted headers and are auto-skipped
4. **Road precision** — 51% precision indicates false positives need post-processing refinement

## Documentation

See the `docs/` folder for detailed documentation:
- [System Overview](docs/SYSTEM_OVERVIEW.md) — What the system does and key metrics
- [Architecture](docs/ARCHITECTURE.md) — Model, loss, and pipeline design
- [Dataset](docs/DATASET.md) — Data sources, format, splits, and sampling
- [Training](docs/TRAINING.md) — Configuration and training guide
- [Evaluation](docs/EVALUATION.md) — Metrics, reports, and inference
- [Improvements](docs/IMPROVEMENTS.md) — Prioritised improvement roadmap

## License

This project was developed for the IIT Hackathon / SVAMITVA competition.
