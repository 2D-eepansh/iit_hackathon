# Architecture

## Model

**DeepLabV3+** with a **ResNet-50** encoder pretrained on ImageNet, implemented via the `segmentation-models-pytorch` (smp) library.

- 4 output classes: Background (0), Road (1), Bridge (2), Built-Up (3)
- Gradient checkpointing enabled to reduce GPU memory usage
- Input: 3-channel RGB patches (768x768 during training, 512x512 during inference)

### Why DeepLabV3+?
- Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale context
- Decoder module recovers fine boundary details
- ResNet-50 provides sufficient capacity without overfitting on this dataset size (ResNet-101 was tested and underperformed)

## Loss Function

**Composite loss**: 0.6 x Focal Loss + 0.4 x Class-weighted Dice Loss

### Focal Loss
- Gamma = 2.0 (focuses on hard examples)
- Per-class alpha weights: Background=0.1, Road=1.0, Bridge=3.0, Built-Up=1.2
- Addresses severe class imbalance (bridge < 0.1% of pixels)

### Class-Weighted Dice Loss
- Computed on foreground classes only (1, 2, 3)
- Per-class weights: Road=2.0, Bridge=1.0, Built-Up=1.0
- Road gets 2x weight as thin linear features are harder to segment
- Smooth factor: 1e-4

## Training Pipeline

```
Patch Extraction (768x768, centroid sampling)
    --> Augmentation (flip, rotate, affine, color jitter)
    --> Forward pass (AMP float16)
    --> Focal + Dice loss
    --> Gradient accumulation (4 steps, effective batch=16)
    --> Gradient clipping (max norm 1.0)
    --> AdamW optimizer step
    --> EMA shadow update (decay=0.99)
```

### Optimizer
- AdamW with weight decay 1e-4
- Differential learning rates: encoder 1e-5, decoder 1e-4
- ReduceLROnPlateau scheduler (patience=5, factor=0.5, monitors val mIoU)

### EMA (Exponential Moving Average)
- Decay = 0.99, updated per optimizer step (~38 steps/epoch)
- Actual model weights used for validation (clean signal for LR scheduler)
- EMA shadow weights saved as the best checkpoint
- Stabilizes against Built-Up class collapse episodes

## Inference Pipeline

```
Full TIFF --> Strip-based sliding window (512x512, overlap=64)
    --> Batch inference (batch=8, AMP)
    --> Logit accumulation + averaging
    --> Argmax --> Predicted mask
    --> Save as GeoTIFF (original CRS preserved)
```

## Data Pipeline

```
GeoTIFF (rasterio windowed read)
    + Shapefile (geopandas, reprojected to TIFF CRS)
    --> On-the-fly rasterization per patch
    --> 4-class pixel mask [0, 1, 2, 3]
```

### Class Priority (Rasterization Order)
Rasterized in reverse order [3, 2, 1] so that Road overwrites Built-Up where they overlap. This matches the real-world hierarchy where roads cut through built-up areas.
