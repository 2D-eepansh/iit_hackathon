# System Overview

## What the System Does

This is an automated geospatial feature extraction system for the SVAMITVA scheme. It takes high-resolution drone orthomosaics (GeoTIFF format) as input and produces pixel-level segmentation masks identifying three infrastructure classes:

- **Road** (class 1) -- paved and unpaved village roads
- **Bridge** (class 2) -- bridge structures over water bodies
- **Built-Up Area** (class 3) -- residential and commercial structures

## Key Metrics

| Metric | Value |
|--------|-------|
| Best mIoU (EMA + TTA + Multiscale) | 0.5880 |
| Road IoU | 0.531 |
| Built-Up IoU | 0.645 |
| Best checkpoint epoch | 39 / 80 |

## Pipeline Stages

```
[TIFF + SHP] --> [Patch Extraction] --> [Training] --> [Best Checkpoint]
                                                              |
                                                              v
[Test TIFF] --> [Sliding Window] --> [Predicted GeoTIFF Mask + Visualization]
```

### Stage 1: Data Ingestion
- Reads large GeoTIFFs (up to 235K x 120K pixels) via windowed rasterio access
- Reprojects shapefile annotations to match each TIFF's CRS on-the-fly
- Rasterizes vector geometries to pixel masks per patch

### Stage 2: Training
- Extracts 768x768 patches with minority-aware centroid sampling
- Trains DeepLabV3+ with Focal + Dice loss for 80 epochs
- Maintains EMA shadow weights for robust checkpointing

### Stage 3: Inference
- Sliding-window inference with 512x512 patches and 64px overlap
- Strip-based processing to bound memory usage on large TIFFs
- Outputs georeferenced GeoTIFF masks with original CRS preserved

### Stage 4: Evaluation
- Patch-based validation on deterministic grid (500 patches)
- Full stakeholder report with per-class accuracy, precision, recall, F1
- Training curve visualization (2D + 3D plots)

## Entry Points

| Script | Purpose |
|--------|---------|
| `train.py` | Train the segmentation model |
| `test_inference.py` | Run inference on test TIFFs |
| `evaluate_model_statistics.py` | Generate evaluation report |
| `visualize_training.py` | Plot training curves |
| `src/inference/evaluate.py` | Standalone CLI evaluation |
| `src/inference/export_model.py` | Export checkpoint / ONNX |
