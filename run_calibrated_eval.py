"""
Calibrated Pipeline Evaluation — measures improvement over raw baseline.

Runs the full calibrated pipeline (ensemble + bias + road gap fill +
bridge recovery) on the validation set and compares to baseline.

Usage:
    venv/bin/python run_calibrated_eval.py
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
from src.inference.calibrated_engine import CalibratedEngine

NUM_CLASSES = 5
VAL_TIFFS   = ["28996_NADALA_ORTHO", "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO"]
TRAIN_TIFFS = [
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho", "TIMMOWAL_37695_ORI",
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO",
    "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_eval(engine, val_loader, postprocess=True):
    tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    gt_px = np.zeros(NUM_CLASSES, dtype=np.int64)
    pr_px = np.zeros(NUM_CLASSES, dtype=np.int64)

    for images, masks in tqdm(val_loader, desc=f"Eval (postproc={postprocess})"):
        preds_np, _ = engine.predict_batch(images, postprocess=postprocess)
        masks_np = masks.numpy()
        for b in range(preds_np.shape[0]):
            p = preds_np[b].flatten()
            t = masks_np[b].flatten()
            for c in range(NUM_CLASSES):
                tp[c]   += ((t == c) & (p == c)).sum()
                gt_px[c] += (t == c).sum()
                pr_px[c] += (p == c).sum()

    results = {}
    fg_ious = []
    for c in range(1, NUM_CLASSES):
        if gt_px[c] == 0:
            continue
        iou = float(tp[c]) / float(gt_px[c] + pr_px[c] - tp[c] + 1e-10)
        prec = float(tp[c]) / float(pr_px[c] + 1e-10)
        rec  = float(tp[c]) / float(gt_px[c] + 1e-10)
        f1   = 2 * prec * rec / (prec + rec + 1e-10)
        results[CLASS_NAMES.get(c)] = {
            "iou": round(iou, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "gt_pixels": int(gt_px[c])
        }
        fg_ious.append(iou)
    results["fg_miou"] = round(float(np.mean(fg_ious)) if fg_ious else 0.0, 4)
    return results


def main():
    print("=" * 70)
    print("  CALIBRATED PIPELINE EVALUATION")
    print("  Ensemble + OptimalBias + RoadGapFill + BridgeRecovery")
    print("=" * 70)

    # Load engine
    print("\n[1/3] Loading calibrated engine...")
    engine = CalibratedEngine.from_checkpoints(
        Path("outputs/checkpoints/best_model.pth"),
        Path("outputs/checkpoints/latest_model.pth"),
        device=DEVICE,
        bias_path=Path("outputs/optimal_bias.json"),
        use_tta=True,
    )

    # Create val loader
    print("\n[2/3] Creating val loader...")
    import torch
    ckpt = torch.load("outputs/checkpoints/best_model.pth", map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    val_ds = UnifiedMultiClassDataset(
        sources=DEFAULT_SOURCES, split="val",
        transform=get_val_transform(cfg.get("image_size", 768)),
        patch_size=cfg.get("image_size", 768),
        patches_per_image=cfg.get("patches_per_image", 50),
        train_tiffs=TRAIN_TIFFS, val_tiffs=VAL_TIFFS,
    )
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"  {len(val_ds)} patches")

    # Run eval: with postprocessing
    print("\n[3/3] Evaluating calibrated pipeline...")
    t0 = time.time()
    results = run_eval(engine, val_loader, postprocess=True)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print("  CALIBRATED PIPELINE RESULTS")
    print(f"{'='*70}")
    print(f"  FG mIoU:  {results['fg_miou']:.4f}   (baseline clean: 0.4981)")
    print(f"  {'Class':20s} {'IoU':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}")
    print(f"  {'-'*64}")
    baseline = {"Road": 0.4894, "Bridge": 0.0000, "Built-Up Area": 0.6530, "Water Body": 0.8500}
    for name, m in results.items():
        if name == "fg_miou":
            continue
        delta = m['iou'] - baseline.get(name, 0)
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:20s} {m['iou']:8.4f} {m['precision']:8.4f} {m['recall']:8.4f} {m['f1']:8.4f}   ({sign}{delta:.4f})")
    print(f"\n  Total eval time: {elapsed:.1f}s")
    print(f"{'='*70}")

    import json
    out = Path("outputs/calibrated_eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()
