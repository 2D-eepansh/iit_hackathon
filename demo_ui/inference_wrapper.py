"""
inference_wrapper.py — Demo UI inference bridge.

Wraps the existing trained model (best_model.pth) for image-level inference.
Accepts numpy RGB arrays (from PIL/rasterio) and returns segmentation masks.

This file is DEMO ONLY — delete /demo_ui to remove entirely.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_factory import create_model  # noqa: E402 (after sys.path)

# ── Constants ─────────────────────────────────────────────────────────────────

CHECKPOINT = ROOT / "outputs" / "checkpoints" / "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SIZE = 512
OVERLAP    = 64
BATCH_SIZE = 4

CLASS_NAMES = {
    0: "Background",
    1: "Road",
    2: "Bridge",
    3: "Built-Up Area",
}

# Same colors as test_inference.py
CLASS_COLORS = np.array([
    [  0,   0,   0],   # 0 Background — black
    [255,   0,   0],   # 1 Road       — red
    [  0,   0, 255],   # 2 Bridge     — blue
    [255, 255,   0],   # 3 Built-Up   — yellow
], dtype=np.uint8)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Cached model (loaded once per session)
_model_cache: dict = {}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model() -> tuple[torch.nn.Module, int, float]:
    """Load model from checkpoint. Returns (model, epoch, best_iou)."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["epoch"], _model_cache["best_iou"]

    ckpt = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]

    model = create_model(
        architecture=cfg["architecture"],
        encoder_name=cfg["encoder_name"],
        encoder_weights=None,
        in_channels=3,
        classes=cfg["classes"],
        use_gradient_checkpointing=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    _model_cache["model"]    = model
    _model_cache["epoch"]    = ckpt["epoch"]
    _model_cache["best_iou"] = ckpt["best_iou"]

    return model, ckpt["epoch"], ckpt["best_iou"]


# ── Core inference ────────────────────────────────────────────────────────────

def _normalize(patch_rgb: np.ndarray) -> torch.Tensor:
    """uint8 HWC → CHW float32 tensor (ImageNet normalization)."""
    x = patch_rgb.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def predict_image(image_rgb: np.ndarray) -> np.ndarray:
    """
    Run sliding-window inference on an arbitrary-size RGB image.

    Args:
        image_rgb: (H, W, 3) uint8 numpy array.

    Returns:
        pred_mask: (H, W) uint8 array with class indices 0-3.
    """
    model, _, _ = load_model()

    # Drop alpha if present
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        image_rgb = image_rgb[:, :, :3]

    h, w = image_rgb.shape[:2]
    stride = PATCH_SIZE - OVERLAP

    logit_acc = np.zeros((4, h, w), dtype=np.float32)
    count_acc = np.zeros((h, w),    dtype=np.float32)

    # Generate tile positions (cover full image, clamp to edges)
    def _positions(size, patch, s):
        pos = list(range(0, max(1, size - patch + 1), s))
        if not pos or pos[-1] + patch < size:
            pos.append(max(0, size - patch))
        return pos

    rows = _positions(h, PATCH_SIZE, stride)
    cols = _positions(w, PATCH_SIZE, stride)

    batch_tensors   = []
    batch_positions = []

    def _flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", enabled=(DEVICE == "cuda")
        ):
            logits = model(batch)
        logits_np = logits.cpu().float().numpy()  # (B, 4, PH, PW)

        for j, (r, c) in enumerate(batch_positions):
            ph = min(PATCH_SIZE, h - r)
            pw = min(PATCH_SIZE, w - c)
            logit_acc[:, r : r + ph, c : c + pw] += logits_np[j, :, :ph, :pw]
            count_acc[r : r + ph, c : c + pw]     += 1.0

        batch_tensors.clear()
        batch_positions.clear()

    for r in rows:
        for c in cols:
            r_end = min(r + PATCH_SIZE, h)
            c_end = min(c + PATCH_SIZE, w)
            patch = image_rgb[r:r_end, c:c_end]

            ph, pw = patch.shape[:2]
            if ph < PATCH_SIZE or pw < PATCH_SIZE:
                padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded

            batch_tensors.append(_normalize(patch))
            batch_positions.append((r, c))

            if len(batch_tensors) >= BATCH_SIZE:
                _flush()

    _flush()

    count_acc = np.maximum(count_acc, 1.0)
    for cls in range(4):
        logit_acc[cls] /= count_acc

    return logit_acc.argmax(axis=0).astype(np.uint8)


# ── Post-processing ───────────────────────────────────────────────────────────

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map class-index mask (H, W) → RGB image (H, W, 3)."""
    return CLASS_COLORS[mask]


def create_overlay(
    image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Blend segmentation colors over the original image (non-background only)."""
    if image_rgb.shape[2] == 4:
        image_rgb = image_rgb[:, :, :3]
    img_f = image_rgb.astype(np.float32) / 255.0
    msk_f = CLASS_COLORS[mask].astype(np.float32) / 255.0
    a     = np.where(mask[..., None] > 0, alpha, 0.0)
    return np.clip((1 - a) * img_f + a * msk_f, 0, 1)


def get_class_stats(mask: np.ndarray) -> dict:
    """Return per-class pixel count and percentage."""
    total = mask.size
    return {
        CLASS_NAMES[i]: {
            "pixels": int((mask == i).sum()),
            "pct":    float(100.0 * (mask == i).sum() / total),
            "color":  CLASS_COLORS[i].tolist(),
        }
        for i in range(4)
    }


def model_info() -> dict:
    """Return metadata about the loaded checkpoint."""
    _, epoch, best_iou = load_model()
    return {
        "checkpoint": CHECKPOINT.name,
        "epoch":      epoch,
        "best_miou":  best_iou,
        "device":     DEVICE,
        "classes":    list(CLASS_NAMES.values()),
    }
