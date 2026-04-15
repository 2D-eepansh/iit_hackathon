"""
inference_wrapper.py — Demo UI inference bridge (Calibrated Engine v2).

Uses ensemble of EMA(ep43) + EMA(ep80) with optimised per-class logit biases,
TTA, road gap-fill, and bridge spatial recovery.

This file is DEMO ONLY — delete /demo_ui to remove entirely.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_factory import create_model  # noqa: E402
from src.postprocessing import (  # noqa: E402
    classify_rooftops,
    get_infrastructure_summary,
    postprocess_mask,
    road_gap_fill,
    bridge_recovery_from_builtup,
)

# Re-export so app.py can import from one place
__all__ = [
    "CLASS_NAMES", "CLASS_COLORS", "ROOFTOP_COLOR",
    "colorize_mask", "colorize_with_rooftops", "create_overlay",
    "get_class_stats", "get_infrastructure_summary",
    "classify_rooftops", "model_info", "predict_image",
]

# ── Constants ─────────────────────────────────────────────────────────────────

BEST_CKPT   = ROOT / "outputs" / "checkpoints" / "best_model.pth"
LATEST_CKPT = ROOT / "outputs" / "checkpoints" / "latest_model.pth"
BIAS_JSON   = ROOT / "outputs" / "optimal_bias.json"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SIZE  = 512
OVERLAP     = 64
BATCH_SIZE  = 4

# Ensemble weights (best checkpoint gets higher weight)
W_BEST   = 0.65
W_LATEST = 0.35

CLASS_NAMES = {
    0: "Background",
    1: "Road",
    2: "Bridge",
    3: "Built-Up Area",
    4: "Water Body",
}

CLASS_COLORS = np.array([
    [  0,   0,   0],   # 0 Background — black
    [255,   0,   0],   # 1 Road       — red
    [  0,   0, 255],   # 2 Bridge     — blue
    [255, 255,   0],   # 3 Built-Up   — yellow
    [  0, 200, 255],   # 4 Water Body — cyan
], dtype=np.uint8)

ROOFTOP_COLOR = np.array([255, 140, 0], dtype=np.uint8)
_DISPLAY_COLORS = np.vstack([CLASS_COLORS, ROOFTOP_COLOR[None, :]])

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Fallback bias if json not found
_DEFAULT_BIAS = [0.0, 1.5, 4.0, 0.0, 0.0]

# Cached models
_model_cache: dict = {}


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(path: Path, state_key: str) -> torch.nn.Module:
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    m = create_model(
        architecture=cfg["architecture"],
        encoder_name=cfg["encoder_name"],
        encoder_weights=None,
        in_channels=3,
        classes=cfg["classes"],
        use_gradient_checkpointing=False,
    )
    m.load_state_dict(ckpt[state_key])
    m.to(DEVICE).eval()
    return m, ckpt


def _get_models():
    if "model_best" in _model_cache:
        return (
            _model_cache["model_best"],
            _model_cache["model_latest"],
            _model_cache["bias"],
            _model_cache["epoch"],
            _model_cache["best_iou"],
        )

    model_best,   ckpt_b = _load_model(BEST_CKPT,   "model_state_dict")
    model_latest, _      = _load_model(LATEST_CKPT, "ema_state_dict")

    # Load optimised bias
    import json
    bias = _DEFAULT_BIAS
    if BIAS_JSON.exists():
        with open(BIAS_JSON) as f:
            data = json.load(f)
        bias = data.get("optimal_bias", _DEFAULT_BIAS)

    _model_cache["model_best"]   = model_best
    _model_cache["model_latest"] = model_latest
    _model_cache["bias"]         = bias
    _model_cache["epoch"]        = ckpt_b.get("epoch", "?")
    _model_cache["best_iou"]     = ckpt_b.get("best_iou", 0.0)

    return model_best, model_latest, bias, ckpt_b.get("epoch", "?"), ckpt_b.get("best_iou", 0.0)


# ── Core inference ────────────────────────────────────────────────────────────

def _normalize(patch_rgb: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> CHW float32 tensor (ImageNet normalization)."""
    x = patch_rgb.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def predict_image(
    image_rgb: np.ndarray,
    use_tta: bool = True,
    return_confidence: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    """
    Run calibrated ensemble sliding-window inference on an arbitrary-size RGB image.

    Args:
        image_rgb:         (H, W, 3) uint8 numpy array.
        use_tta:           Average predictions with h-flip and v-flip.
        return_confidence: If True, also return (H, W) float32 max-confidence map.

    Returns:
        pred_mask:  (H, W) uint8 class indices 0-4, post-processed.
        conf_map:   (H, W) float32, max softmax probability per pixel (only if return_confidence).
    """
    model_best, model_latest, bias_list, _, _ = _get_models()
    bias_t = torch.tensor(bias_list, dtype=torch.float32, device=DEVICE).view(1, len(CLASS_NAMES), 1, 1)

    if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        image_rgb = image_rgb[:, :, :3]

    h, w   = image_rgb.shape[:2]
    stride = PATCH_SIZE - OVERLAP

    logit_acc = np.zeros((len(CLASS_NAMES), h, w), dtype=np.float32)
    count_acc = np.zeros((h, w), dtype=np.float32)

    def _positions(size, patch, s):
        pos = list(range(0, max(1, size - patch + 1), s))
        if not pos or pos[-1] + patch < size:
            pos.append(max(0, size - patch))
        return pos

    rows = _positions(h, PATCH_SIZE, stride)
    cols = _positions(w, PATCH_SIZE, stride)

    batch_tensors   = []
    batch_positions = []

    def _ensemble_forward(batch_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", enabled=(DEVICE == "cuda")
        ):
            logits = W_BEST * model_best(batch_t).float() + W_LATEST * model_latest(batch_t).float()
            if use_tta:
                b_h = torch.flip(batch_t, [3])
                l_h = torch.flip(
                    W_BEST * model_best(b_h).float() + W_LATEST * model_latest(b_h).float(),
                    [3]
                )
                b_v = torch.flip(batch_t, [2])
                l_v = torch.flip(
                    W_BEST * model_best(b_v).float() + W_LATEST * model_latest(b_v).float(),
                    [2]
                )
                logits = (logits + l_h + l_v) / 3.0
        return logits

    def _flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(DEVICE)
        logits_t = _ensemble_forward(batch)
        # Apply bias before accumulating
        logits_t = logits_t + bias_t
        logits_np = logits_t.cpu().numpy()  # (B, C, PH, PW)

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
    for cls in range(len(CLASS_NAMES)):
        logit_acc[cls] /= count_acc

    mask = logit_acc.argmax(axis=0).astype(np.uint8)
    mask = postprocess_mask(mask, logit_acc)

    if return_confidence:
        exp_logits = np.exp(logit_acc - logit_acc.max(axis=0, keepdims=True))
        probs = exp_logits / (exp_logits.sum(axis=0, keepdims=True) + 1e-9)
        conf_map = probs.max(axis=0).astype(np.float32)
        return mask, conf_map

    return mask


# ── Colorization helpers ──────────────────────────────────────────────────────

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask]


def colorize_with_rooftops(
    mask: np.ndarray,
    rooftop_mask: "np.ndarray | None" = None,
) -> np.ndarray:
    rgb = CLASS_COLORS[mask].copy()
    if rooftop_mask is not None and rooftop_mask.any():
        rgb[rooftop_mask] = ROOFTOP_COLOR
    return rgb


def colorize_confidence(conf_map: np.ndarray) -> np.ndarray:
    """Map (H,W) float32 confidence [0,1] -> RGB heatmap (H,W,3) uint8."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap("RdYlGn")
    rgba = cmap(conf_map)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def create_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    rooftop_mask: "np.ndarray | None" = None,
) -> np.ndarray:
    if image_rgb.shape[2] == 4:
        image_rgb = image_rgb[:, :, :3]
    img_f = image_rgb.astype(np.float32) / 255.0
    msk_f = colorize_with_rooftops(mask, rooftop_mask).astype(np.float32) / 255.0
    a     = np.where(mask[..., None] > 0, alpha, 0.0)
    return np.clip((1 - a) * img_f + a * msk_f, 0, 1)


def get_class_stats(
    mask: np.ndarray,
    rooftop_mask: "np.ndarray | None" = None,
) -> dict:
    total = mask.size
    stats = {
        CLASS_NAMES[i]: {
            "pixels": int((mask == i).sum()),
            "pct":    float(100.0 * (mask == i).sum() / total),
            "color":  CLASS_COLORS[i].tolist(),
        }
        for i in range(len(CLASS_NAMES))
    }
    if rooftop_mask is not None:
        rooftop_px = int(rooftop_mask.sum())
        stats["Rooftop (est.)"] = {
            "pixels": rooftop_px,
            "pct":    float(100.0 * rooftop_px / total),
            "color":  ROOFTOP_COLOR.tolist(),
        }
    return stats


def model_info() -> dict:
    """Return metadata about the loaded ensemble."""
    import json
    _, _, bias, epoch, best_iou = _get_models()
    bias_loaded = BIAS_JSON.exists()
    return {
        "checkpoint":    BEST_CKPT.name,
        "epoch":         epoch,
        "best_miou":     best_iou,
        "device":        DEVICE,
        "classes":       list(CLASS_NAMES.values()),
        "ensemble":      True,
        "bias_tuned":    bias_loaded,
        "pipeline":      "Ensemble(ep43+ep80) + CalibBias + TTA + RoadGapFill + BridgeRecovery",
    }
