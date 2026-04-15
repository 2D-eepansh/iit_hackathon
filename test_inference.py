"""
Inference script for SVAMITVA multi-class segmentation on test TIFFs.

Tiles each test raster into 512×512 patches, runs the trained model,
stitches predictions, and outputs:
  - Per-TIFF predicted mask (GeoTIFF with original CRS/transform)
  - Per-TIFF visualization: image | mask | overlay
  - Summary statistics per class per TIFF

Usage:
    python test_inference.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

import rasterio
from rasterio.windows import Window

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from src.models.model_factory import create_model
from src.postprocessing import postprocess_mask

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BEST_CKPT    = Path("outputs/checkpoints/best_model.pth")
LATEST_CKPT  = Path("outputs/checkpoints/latest_model.pth")
BIAS_JSON    = Path("outputs/optimal_bias.json")
TEST_DIR     = Path("Test/live-demo")
OUTPUT_DIR   = Path("outputs/test_predictions_live_demo")
PATCH_SIZE   = 512
OVERLAP      = 64
BATCH_SIZE   = 8
USE_TTA      = True
W_BEST       = 0.65
W_LATEST     = 0.35
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "Background",
    1: "Road",
    2: "Bridge",
    3: "Built-Up Area",
    4: "Water Body",
}

COLORS = np.array([
    [  0,   0,   0],      # 0 background   — black
    [255,   0,   0],      # 1 road         — red
    [  0,   0, 255],      # 2 bridge       — blue
    [255, 255,   0],      # 3 built-up     — yellow
    [  0, 200, 255],      # 4 water body   — cyan
], dtype=np.uint8)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_ensemble(device: str) -> tuple:
    """Load both checkpoints and optimal bias. Returns (model_best, model_latest, bias_t)."""
    import json

    def _load(path, key):
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        cfg  = ckpt["config"]
        m = create_model(
            architecture=cfg["architecture"],
            encoder_name=cfg["encoder_name"],
            encoder_weights=None,
            in_channels=3,
            classes=cfg["classes"],
        )
        m.load_state_dict(ckpt[key])
        m.to(device).eval()
        return m, ckpt

    model_best,   ckpt_b = _load(BEST_CKPT,   "model_state_dict")
    model_latest, _      = _load(LATEST_CKPT, "ema_state_dict")

    # Load optimised bias
    bias = [0.0, 1.5, 4.0, 0.0, 0.0]  # fallback
    if BIAS_JSON.exists():
        with open(BIAS_JSON) as f:
            data = json.load(f)
        bias = data.get("optimal_bias", bias)
        print(f"  Loaded optimal bias from {BIAS_JSON.name}")
    else:
        print(f"  Using default bias (run bias_search.py to tune)")
    bias_t = torch.tensor(bias, dtype=torch.float32, device=device).view(1, len(CLASS_NAMES), 1, 1)

    print(f"  Ensemble: {BEST_CKPT.name} (ep{ckpt_b['epoch']}, w={W_BEST}) + "
          f"{LATEST_CKPT.name} (w={W_LATEST})")
    print(f"  Best mIoU: {ckpt_b['best_iou']:.4f}  Classes: {ckpt_b['config']['classes']}")
    return model_best, model_latest, bias_t


def normalize_patch(patch_rgb: np.ndarray) -> torch.Tensor:
    """Normalise uint8 HWC → CHW float32 tensor (ImageNet stats)."""
    x = patch_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))   # (3, H, W)


def predict_tiff(
    model_best: torch.nn.Module,
    model_latest: torch.nn.Module,
    bias_t: torch.Tensor,
    tif_path: Path,
    device: str,
) -> tuple:
    """
    Run sliding-window inference on a full TIFF using strip-based processing
    to keep memory usage bounded.

    Returns:
        pred_mask  (H, W) uint8   — predicted class per pixel
        rgb_thumb  (th, tw, 3)     — downsampled RGB for visualisation
    """
    with rasterio.open(str(tif_path)) as src:
        h, w   = src.height, src.width
        bands  = src.count
        crs    = src.crs
        profile = src.profile.copy()

        print(f"\n  Raster: {tif_path.name}")
        print(f"  Size: {h}×{w}  Bands: {bands}  CRS: {crs}")

        num_classes = len(CLASS_NAMES)
        stride = PATCH_SIZE - OVERLAP

        # Process in horizontal strips of PATCH_SIZE height
        # Only keep one strip of logit accumulators in memory at a time
        pred_mask = np.zeros((h, w), dtype=np.uint8)

        # Generate row positions
        row_positions = list(range(0, h - PATCH_SIZE + 1, stride))
        if row_positions[-1] + PATCH_SIZE < h:
            row_positions.append(h - PATCH_SIZE)

        col_positions = list(range(0, w - PATCH_SIZE + 1, stride))
        if col_positions[-1] + PATCH_SIZE < w:
            col_positions.append(w - PATCH_SIZE)

        total_patches = len(row_positions) * len(col_positions)
        print(f"  Patches: {total_patches}  (stride={stride})")

        patch_count = 0

        for row_idx, r in enumerate(row_positions):
            # Accumulation buffer for this strip only
            strip_logits = np.zeros((num_classes, PATCH_SIZE, w), dtype=np.float32)
            strip_counts = np.zeros((PATCH_SIZE, w), dtype=np.float32)

            batch_imgs = []
            batch_cols = []

            for c in col_positions:
                window = Window(c, r, PATCH_SIZE, PATCH_SIZE)
                patch = src.read([1, 2, 3], window=window)
                patch_hwc = patch.transpose(1, 2, 0).astype(np.uint8)
                tensor = normalize_patch(patch_hwc)
                batch_imgs.append(tensor)
                batch_cols.append(c)

                if len(batch_imgs) == BATCH_SIZE or c == col_positions[-1]:
                    batch = torch.stack(batch_imgs).to(device)
                    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                        logits = (W_BEST * model_best(batch).float() +
                                  W_LATEST * model_latest(batch).float())
                        if USE_TTA:
                            b_h = torch.flip(batch, [3])
                            l_h = torch.flip(W_BEST * model_best(b_h).float() +
                                             W_LATEST * model_latest(b_h).float(), [3])
                            b_v = torch.flip(batch, [2])
                            l_v = torch.flip(W_BEST * model_best(b_v).float() +
                                             W_LATEST * model_latest(b_v).float(), [2])
                            logits = (logits + l_h + l_v) / 3.0
                        # Apply calibrated bias
                        logits = logits + bias_t
                    logits_np = logits.cpu().numpy()

                    for j, pc in enumerate(batch_cols):
                        strip_logits[:, :, pc:pc+PATCH_SIZE] += logits_np[j]
                        strip_counts[:, pc:pc+PATCH_SIZE] += 1.0

                    batch_imgs.clear()
                    batch_cols.clear()

                patch_count += 1

            # Average and argmax this strip
            strip_counts = np.maximum(strip_counts, 1.0)
            for cls in range(num_classes):
                strip_logits[cls] /= strip_counts

            strip_pred = strip_logits.argmax(axis=0).astype(np.uint8)  # (PATCH_SIZE, w)

            # Write into the non-overlapping portion of this strip
            if row_idx == 0:
                write_start = 0
                write_end = stride if len(row_positions) > 1 else PATCH_SIZE
            elif row_idx == len(row_positions) - 1:
                write_start = 0  # take entire last strip's unique part
                write_end = PATCH_SIZE
            else:
                write_start = 0
                write_end = stride

            # For first strip, write from row 0; for others, only write the new rows
            if row_idx == 0:
                pred_mask[r:r+write_end, :] = strip_pred[:write_end, :]
            else:
                # Only overwrite rows not yet finalized
                actual_start = r + OVERLAP // 2  # blend boundary
                actual_end   = r + PATCH_SIZE
                if row_idx == len(row_positions) - 1:
                    actual_end = h
                local_start = actual_start - r
                local_end   = actual_end - r
                pred_mask[actual_start:actual_end, :] = strip_pred[local_start:local_end, :]

            if (row_idx + 1) % 5 == 0 or row_idx == len(row_positions) - 1:
                print(f"    Strip {row_idx+1}/{len(row_positions)}  "
                      f"({patch_count}/{total_patches} patches)")

            del strip_logits, strip_counts

        # Read a downsampled thumbnail for visualisation (max 2048 on long side)
        scale = min(1.0, 2048.0 / max(h, w))
        th, tw = int(h * scale), int(w * scale)
        thumb = src.read(
            [1, 2, 3],
            out_shape=(3, th, tw),
        ).transpose(1, 2, 0).astype(np.uint8)

    return pred_mask, thumb, profile, (h, w)


def save_geotiff_mask(pred_mask: np.ndarray, profile: dict, output_path: Path) -> None:
    """Save prediction as single-band GeoTIFF with original CRS."""
    profile.update(
        dtype="uint8",
        count=1,
        compress="lzw",
    )
    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(pred_mask, 1)
    print(f"    ✓ GeoTIFF mask  → {output_path.name}")


def save_visualisation(
    pred_mask: np.ndarray, thumb: np.ndarray, shape: tuple,
    output_path: Path, tif_name: str,
) -> None:
    """Save a 3-panel PNG: image thumbnail | predicted mask | overlay."""
    h, w = shape
    scale = min(1.0, 2048.0 / max(h, w))
    th, tw = int(h * scale), int(w * scale)

    # Downsample mask to thumbnail size
    mask_small = np.array(
        torch.nn.functional.interpolate(
            torch.from_numpy(pred_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
            size=(th, tw),
            mode="nearest",
        ).squeeze().numpy()
    ).astype(np.uint8)

    # Colorize mask
    mask_rgb = COLORS[mask_small]                 # (th, tw, 3)

    # Overlay
    img_f = thumb.astype(np.float32) / 255.0
    msk_f = mask_rgb.astype(np.float32) / 255.0
    alpha = np.where(mask_small[..., None] > 0, 0.45, 0.0)
    overlay = ((1 - alpha) * img_f + alpha * msk_f)
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    axes[0].imshow(thumb)
    axes[0].set_title("Input Orthomosaic", fontsize=13)
    axes[0].axis("off")

    im = axes[1].imshow(mask_small, cmap="tab10", vmin=0, vmax=len(CLASS_NAMES)-1)
    axes[1].set_title("Predicted Segmentation Mask", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Image + Prediction)", fontsize=13)
    axes[2].axis("off")

    legend_elements = [
        Patch(facecolor=np.array(COLORS[i]) / 255.0, label=f"{i} — {CLASS_NAMES[i]}")
        for i in range(len(CLASS_NAMES))
    ]
    axes[2].legend(handles=legend_elements, loc="lower right", fontsize=9,
                   framealpha=0.8)

    fig.suptitle(
        f"SVAMITVA Test Prediction — {tif_name}\n{h}×{w} px  |  {PATCH_SIZE}px patches  |  {OVERLAP}px overlap",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Visualisation → {output_path.name}")


def print_class_stats(pred_mask: np.ndarray, tif_name: str) -> None:
    """Print per-class pixel statistics."""
    total = pred_mask.size
    print(f"\n    {'Class':<6} {'Name':<18} {'Pixels':>12}  {'%':>7}")
    print(f"    {'─'*6} {'─'*18} {'─'*12}  {'─'*7}")
    for cid in range(len(CLASS_NAMES)):
        count = int((pred_mask == cid).sum())
        pct   = 100.0 * count / total
        bar   = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"    {cid:<6} {CLASS_NAMES[cid]:<18} {count:>12,}  {pct:>6.2f}%  {bar[:40]}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("  SVAMITVA MULTI-CLASS SEGMENTATION — TEST INFERENCE")
    print("=" * 72)

    # Find test TIFFs
    test_tifs = sorted(TEST_DIR.rglob("*.tif"))
    if not test_tifs:
        print(f"No TIFF files found in {TEST_DIR}")
        sys.exit(1)

    print(f"\nTest TIFFs found: {len(test_tifs)}")
    for t in test_tifs:
        print(f"  • {t.relative_to(TEST_DIR)}")

    # Load ensemble
    print()
    model_best, model_latest, bias_t = load_ensemble(DEVICE)

    # Output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    for tif_path in test_tifs:
        tif_name = tif_path.stem
        print(f"\n{'─' * 72}")
        print(f"  Processing: {tif_name}")
        print(f"{'─' * 72}")

        t0 = time.time()
        pred_mask, thumb, profile, shape = predict_tiff(
            model_best, model_latest, bias_t, tif_path, DEVICE
        )
        elapsed = time.time() - t0
        print(f"    Inference time: {elapsed:.1f}s")

        # Morphological post-processing (road/water/bridge refinement)
        pred_mask = postprocess_mask(pred_mask)

        # Stats
        print_class_stats(pred_mask, tif_name)

        # Save outputs
        mask_out = OUTPUT_DIR / f"{tif_name}_pred_mask.tif"
        vis_out  = OUTPUT_DIR / f"{tif_name}_prediction.png"

        save_geotiff_mask(pred_mask, profile, mask_out)
        save_visualisation(pred_mask, thumb, shape, vis_out, tif_name)

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 72}")
    print(f"  INFERENCE COMPLETE")
    print(f"  Total time:  {total_elapsed:.1f}s")
    print(f"  Outputs in:  {OUTPUT_DIR.resolve()}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
