# Training Guide

## Configuration

All training configuration is in the `CONFIG` dict at the top of `train.py`. There are no CLI arguments -- edit the dict directly.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 768 | Patch size in pixels |
| `patches_per_image` | 150 | Patches extracted per TIFF per epoch |
| `batch_size` | 4 | GPU batch size |
| `accumulation_steps` | 4 | Gradient accumulation (effective batch = 16) |
| `num_epochs` | 80 | Total training epochs |
| `learning_rate` | 1e-4 | Decoder learning rate |
| `encoder_lr` | 1e-5 | Encoder learning rate (10x lower) |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `ema_decay` | 0.99 | EMA shadow weight decay |
| `architecture` | DeepLabV3Plus | Model architecture |
| `encoder_name` | resnet50 | Encoder backbone |
| `use_tta` | False | TTA during training validation (slow) |
| `use_multiscale_val` | True | Multi-scale validation |
| `use_road_refinement` | True | Morphological road post-processing |

## Running Training

```bash
python train.py
```

For unbuffered output with logging:
```bash
PYTHONUNBUFFERED=1 python -u train.py 2>&1 | tee outputs/train.log
```

## Data Split

The train/val split is at the TIFF level (no spatial leakage):

**Train TIFFs** (4 valid + 2 auto-skipped due to corruption):
- PINDORI MAYA SINGH-TUGALWAL (Punjab)
- TIMMOWAL (Punjab)
- BADETUMNAR/BANGAPAL/CHHOTETUMAR/MOFALNAR (Chhattisgarh)
- MURDANDA/AWAPALLI/CHINTAKONTA (Chhattisgarh, has 5 bridges)

**Validation TIFFs** (2):
- NADALA (Punjab)
- NAGUL/MADASE/GHOTPAL (Chhattisgarh, has 1 bridge)

## Augmentation Pipeline

Training augmentations (via Albumentations):
1. Resize to `image_size` x `image_size`
2. Horizontal flip (p=0.5)
3. Vertical flip (p=0.5)
4. Random rotate 90 (p=0.5)
5. Affine: rotation +/-45 deg, scale 0.85-1.15, translate +/-6.25%
6. Color jitter: brightness/contrast +/-0.2, saturation/hue +/-0.1
7. Gaussian blur (kernel 3-5, p=0.1)
8. CLAHE (p=0.2)
9. ImageNet normalization
10. Bridge copy-paste (30 cached patches, applied to train patches)

## Checkpoints

Saved to `outputs/checkpoints/`:
- `best_model.pth` -- EMA weights from the epoch with highest val mIoU
- `latest_model.pth` -- actual model weights from the latest epoch

Checkpoint contents:
```python
{
    "epoch": int,
    "model_state_dict": ...,    # EMA weights (best) or actual weights (latest)
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "scaler_state_dict": ...,
    "ema_state_dict": ...,
    "best_iou": float,
    "config": dict,
    "metrics": dict,
}
```

## Resume Training

Set `resume_checkpoint` in `CONFIG`:
```python
"resume_checkpoint": "outputs/checkpoints/latest_model.pth",
```

## Training History

Epoch metrics are saved to `outputs/training_history.json` after every epoch. Use `visualize_training.py` to generate plots.

## Resource Requirements

- GPU: 8+ GB VRAM (tested on RTX 5070 Laptop, 16 GB)
- RAM: 16+ GB recommended
- Disk: ~2 GB for dataset, ~500 MB for checkpoints
- Epoch time: 60-180 seconds (varies with rasterio I/O on large TIFFs)
