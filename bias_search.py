"""
Logit Bias Grid Search — find optimal per-class decision boundaries.

Strategy:
  1. Run ONE inference pass: ensemble of EMA(ep43) + EMA(ep80), save logits as float16 memmap.
  2. Coordinate-descent grid search over per-class biases on cached logits (ms per config).
  3. Report optimal biases and expected mIoU improvement.

Usage:
    venv/bin/python bias_search.py
"""

import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.datasets.unified_dataset import (
    UnifiedMultiClassDataset, DEFAULT_SOURCES, get_val_transform, CLASS_NAMES
)
from src.models.model_factory import create_model

# ── Config ────────────────────────────────────────────────────────────────────
BEST_CKPT   = Path("outputs/checkpoints/best_model.pth")      # EMA ep43
LATEST_CKPT = Path("outputs/checkpoints/latest_model.pth")    # EMA ep80
CACHE_DIR   = Path("outputs/bias_cache")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 8
NUM_WORKERS = 4
NUM_CLASSES = 5
VAL_TIFFS   = [
    "28996_NADALA_ORTHO",
    "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO",
]
TRAIN_TIFFS = [
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho",
    "TIMMOWAL_37695_ORI",
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO",
    "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO",
]
ENSEMBLE_WEIGHT_BEST = 0.65   # higher weight to epoch-43 best model
ENSEMBLE_WEIGHT_LATEST = 0.35


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model_from_ckpt(path: Path, state_key: str = "model_state_dict") -> torch.nn.Module:
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    model = create_model(
        architecture=cfg["architecture"],
        encoder_name=cfg["encoder_name"],
        encoder_weights=None,
        in_channels=3,
        classes=cfg["classes"],
    )
    model.load_state_dict(ckpt[state_key])
    model.to(DEVICE).eval()
    return model, cfg


def create_val_loader(cfg):
    ds = UnifiedMultiClassDataset(
        sources=DEFAULT_SOURCES, split="val",
        transform=get_val_transform(cfg.get("image_size", 768)),
        patch_size=cfg.get("image_size", 768),
        patches_per_image=cfg.get("patches_per_image", 50),
        train_tiffs=TRAIN_TIFFS, val_tiffs=VAL_TIFFS,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


# ── IoU from flat arrays ───────────────────────────────────────────────────────

def compute_fg_miou(preds_flat: np.ndarray, gt_flat: np.ndarray,
                    num_classes: int = NUM_CLASSES) -> tuple[float, dict]:
    """Return (fg_mIoU, per_class_iou_dict). Excludes BG and absent classes."""
    fg_ious = {}
    for c in range(1, num_classes):
        gt_c = (gt_flat == c)
        if gt_c.sum() == 0:
            continue
        pr_c = (preds_flat == c)
        inter = (gt_c & pr_c).sum()
        union = (gt_c | pr_c).sum()
        fg_ious[c] = float(inter) / float(union + 1e-10)
    miou = float(np.mean(list(fg_ious.values()))) if fg_ious else 0.0
    return miou, fg_ious


# ── Inference + Cache ─────────────────────────────────────────────────────────

def run_and_cache(model_best, model_latest, val_loader) -> tuple:
    """
    Run ensemble inference, return (logits_flat_f16, gt_flat_u8).
    Caches to CACHE_DIR so subsequent runs skip inference.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logit_cache = CACHE_DIR / "ensemble_logits.npy"
    gt_cache    = CACHE_DIR / "gt_flat.npy"

    if logit_cache.exists() and gt_cache.exists():
        print(f"Loading cached logits from {CACHE_DIR}/")
        logits_all = np.load(logit_cache)
        gt_all     = np.load(gt_cache)
        print(f"  Loaded: logits {logits_all.shape}, gt {gt_all.shape}")
        return logits_all, gt_all

    all_logits = []
    all_gt     = []

    print("Running ensemble inference (will be cached afterwards)...")
    model_best.eval()
    model_latest.eval()
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Caching logits"):
            images = images.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type="cuda"):
                logits_b = model_best(images).float()
                logits_l = model_latest(images).float()
            # Weighted ensemble at logit level
            logits_ens = ENSEMBLE_WEIGHT_BEST * logits_b + ENSEMBLE_WEIGHT_LATEST * logits_l

            B, C, H, W = logits_ens.shape
            logits_flat = logits_ens.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            gt_flat     = masks.numpy().reshape(-1).astype(np.uint8)                      # (B*H*W,)
            all_logits.append(logits_flat.astype(np.float16))
            all_gt.append(gt_flat)

    logits_all = np.concatenate(all_logits, axis=0)   # (N_total, C)
    gt_all     = np.concatenate(all_gt, axis=0)       # (N_total,)

    print(f"Saving cache: logits {logits_all.shape} ({logits_all.nbytes/1e9:.2f} GB)...")
    np.save(logit_cache, logits_all)
    np.save(gt_cache, gt_all)
    print("Cached.")
    return logits_all, gt_all


# ── Grid Search ───────────────────────────────────────────────────────────────

def apply_bias_and_iou(logits_f16: np.ndarray, gt: np.ndarray, bias: np.ndarray) -> tuple[float, dict]:
    """Apply per-class logit bias and compute fg mIoU. Fast numpy path."""
    # logits_f16: (N, C) float16; bias: (C,) float32
    biased = logits_f16.astype(np.float32) + bias[np.newaxis, :]  # broadcast
    preds  = biased.argmax(axis=1).astype(np.uint8)
    return compute_fg_miou(preds, gt)


def coordinate_descent_search(logits: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    1D coordinate descent: optimise one class bias at a time, fix others.
    Converges in 1-2 rounds.
    """
    # Search grids per class
    grids = {
        0: np.array([-2.0, -1.0, -0.5, 0.0]),          # Background: slight penalty helps Road
        1: np.arange(0.0, 3.5, 0.25),                   # Road: boost to recover recall
        2: np.arange(0.0, 6.5, 0.5),                    # Bridge: large boost needed
        3: np.array([-0.5, -0.25, 0.0, 0.25, 0.5]),    # Built-Up: small adjustment
        4: np.array([-0.5, 0.0, 0.5]),                  # Water: already good, minor tweak
    }

    best_bias  = np.zeros(NUM_CLASSES, dtype=np.float32)
    best_miou, best_per = apply_bias_and_iou(logits, gt, best_bias)
    print(f"\nBaseline mIoU (bias=0): {best_miou:.4f}")
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES.get(c, f"C{c}")
        iou  = best_per.get(c, None)
        print(f"  {name}: {iou:.4f}" if iou is not None else f"  {name}: N/A")

    for _round in range(3):
        changed = False
        for c in range(NUM_CLASSES):
            best_c = best_bias[c]
            for val in grids[c]:
                trial = best_bias.copy()
                trial[c] = float(val)
                miou_t, _ = apply_bias_and_iou(logits, gt, trial)
                if miou_t > best_miou + 1e-5:
                    best_miou = miou_t
                    best_bias = trial.copy()
                    best_c    = float(val)
                    changed   = True
            best_bias[c] = best_c
        if not changed:
            break

    print(f"\nOptimised mIoU:        {best_miou:.4f}")
    _, best_per = apply_bias_and_iou(logits, gt, best_bias)
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES.get(c, f"C{c}")
        iou  = best_per.get(c, None)
        print(f"  {name}: {iou:.4f}" if iou is not None else f"  {name}: N/A")

    return best_bias, best_miou


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  LOGIT BIAS GRID SEARCH  (Ensemble ep43 + ep80)")
    print("=" * 70)

    # 1. Load models
    print("\n[1/3] Loading models...")
    t0 = time.time()
    model_best,   cfg  = load_model_from_ckpt(BEST_CKPT,   "model_state_dict")
    model_latest, _    = load_model_from_ckpt(LATEST_CKPT, "ema_state_dict")
    print(f"  Both models loaded in {time.time()-t0:.1f}s")

    # 2. Create val loader & run/load cached inference
    print("\n[2/3] Caching ensemble logits...")
    val_loader = create_val_loader(cfg)
    t0 = time.time()
    logits, gt = run_and_cache(model_best, model_latest, val_loader)
    print(f"  Ready in {time.time()-t0:.1f}s  | pixels: {len(gt):,}")

    # 3. Grid search
    print("\n[3/3] Running coordinate-descent bias search...")
    t0 = time.time()
    best_bias, best_miou = coordinate_descent_search(logits, gt)
    print(f"\n  Search complete in {time.time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print("  OPTIMAL BIAS VECTOR")
    print("=" * 70)
    for c, v in enumerate(best_bias):
        print(f"  {CLASS_NAMES.get(c, f'C{c}'):15s}: {v:+.3f}")
    print(f"\n  SAVE THIS for calibrated_engine.py:")
    bias_str = "[" + ", ".join(f"{v:.4f}" for v in best_bias) + "]"
    print(f"  OPTIMAL_BIAS = {bias_str}")
    print(f"  Optimised FG mIoU: {best_miou:.4f}")

    # Save results
    import json
    result = {
        "optimal_bias": best_bias.tolist(),
        "best_miou": float(best_miou),
        "ensemble_weights": {"best_ep43": ENSEMBLE_WEIGHT_BEST, "latest_ep80": ENSEMBLE_WEIGHT_LATEST},
    }
    out = Path("outputs/optimal_bias.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
