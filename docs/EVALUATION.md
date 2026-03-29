# Evaluation Guide

## Evaluation Scripts

### 1. Stakeholder Evaluation Report

```bash
python evaluate_model_statistics.py
```

Generates:
- `outputs/evaluation_report.json` -- detailed metrics in JSON
- `outputs/evaluation_report.txt` -- formatted text report for stakeholders
- Console summary with per-class accuracy, precision, recall, F1

Metrics computed:
- Overall pixel accuracy
- Infrastructure accuracy (non-background classes only)
- Per-class accuracy, precision, recall, F1 score
- Confusion matrix
- Infrastructure detection rates
- Class distribution in validation set

### 2. Training Curve Visualization

```bash
python visualize_training.py
```

Generates plots in `outputs/plots/`:
- `loss_curves.png` -- train/val loss over epochs
- `iou_dice.png` -- validation mIoU and mDice
- `per_class_iou.png` -- per-class IoU trajectories
- `lr_schedule.png` -- learning rate schedule
- `summary_2d.png` -- 4-panel summary
- `3d_per_class_iou.png` -- 3D class IoU trajectories
- `3d_training_trajectory.png` -- 3D epoch-loss-IoU trajectory
- `3d_iou_surface.png` -- IoU surface across classes and epochs

### 3. Standalone CLI Evaluation

```bash
python src/inference/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```

Lightweight evaluation with TTA and multi-scale options.

## Inference on Test TIFFs

```bash
python test_inference.py
```

Configuration (edit constants at top of `test_inference.py`):
- `CHECKPOINT` -- path to model checkpoint
- `TEST_DIR` -- directory containing test GeoTIFFs
- `PATCH_SIZE` -- inference patch size (default: 512)
- `OVERLAP` -- overlap between patches (default: 64)
- `BATCH_SIZE` -- inference batch size (default: 8)

Outputs per test TIFF:
- `{name}_pred_mask.tif` -- georeferenced prediction mask (GeoTIFF, LZW compressed)
- `{name}_prediction.png` -- 3-panel visualization: input | mask | overlay

## Model Export

```bash
python src/inference/export_model.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output-dir outputs/models
```

Exports:
- Cleaned checkpoint (weights only, no optimizer state)
- ONNX model for deployment

## Metrics Reference

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union across all 4 classes |
| **mDice** | Mean Dice coefficient across all 4 classes |
| **Per-class IoU** | IoU for each individual class |
| **Pixel Accuracy** | Fraction of correctly classified pixels |
| **Infrastructure Accuracy** | Accuracy considering only non-background pixels |
| **F1 Score** | Harmonic mean of precision and recall per class |

## Validation Strategy

- **Deterministic grid**: 500 patches sampled on a fixed grid (no randomness between epochs)
- **Actual model validation**: The real model weights (not EMA) are used during training validation to give a clean signal to the LR scheduler
- **EMA for checkpointing**: EMA shadow weights are saved when a new best val mIoU is achieved
- **TTA disabled during training**: TTA is too slow for per-epoch validation; enable for final evaluation via `use_tta=True`
