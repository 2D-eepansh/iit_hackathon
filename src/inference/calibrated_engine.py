"""
Calibrated Inference Engine — Phase 5 Production Pipeline.

Combines:
  1. Two-model EMA ensemble (ep43 best + ep80 final),  weighted 65/35
  2. Per-class logit bias (tuned by bias_search.py coordinate descent)
  3. TTA: horizontal + vertical flip averaging
  4. Enhanced postprocessing (road gap fill, bridge spatial recovery)
  5. Village-level quantitative statistics

Usage:
    from src.inference.calibrated_engine import CalibratedEngine
    engine = CalibratedEngine.from_checkpoints(
        best_ckpt, latest_ckpt, device, bias_path="outputs/optimal_bias.json"
    )
    pred_mask, stats = engine.predict_patch(image_tensor)
    pred_mask, stats = engine.predict_tiff(tiff_path, output_path)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.model_factory import create_model
from src.postprocessing import (
    postprocess_mask,
    bridge_recovery_from_builtup,
    road_gap_fill,
    classify_rooftops,
    get_infrastructure_summary,
)

# ── Class IDs ─────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Background", 1: "Road", 2: "Bridge", 3: "Built-Up Area", 4: "Water Body"}
NUM_CLASSES  = 5

# ── Fallback bias if no JSON available (from search results) ──────────────────
_DEFAULT_BIAS = [0.0, 1.5, 4.0, 0.0, 0.0]


class CalibratedEngine:
    """
    Two-model ensemble inference + calibrated decision + enhanced postprocessing.
    """

    def __init__(
        self,
        model_best:    torch.nn.Module,
        model_latest:  torch.nn.Module,
        device:        str,
        image_size:    int   = 768,
        bias:          list  = None,
        w_best:        float = 0.65,
        w_latest:      float = 0.35,
        use_tta:       bool  = True,
    ):
        self.model_best   = model_best
        self.model_latest = model_latest
        self.device       = device
        self.image_size   = image_size
        self.w_best       = w_best
        self.w_latest     = w_latest
        self.use_tta      = use_tta

        b = bias if bias is not None else _DEFAULT_BIAS
        self.bias = torch.tensor(b, dtype=torch.float32, device=device).view(1, NUM_CLASSES, 1, 1)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoints(
        cls,
        best_ckpt:   Path,
        latest_ckpt: Path,
        device:      str,
        bias_path:   Optional[Path] = None,
        use_tta:     bool = True,
    ) -> "CalibratedEngine":
        best_ckpt   = Path(best_ckpt)
        latest_ckpt = Path(latest_ckpt)

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
            return m, cfg

        model_best,   cfg  = _load(best_ckpt,   "model_state_dict")   # EMA ep43
        model_latest, _    = _load(latest_ckpt, "ema_state_dict")      # EMA ep80

        # Load or use default bias
        bias = _DEFAULT_BIAS
        bp   = Path(bias_path) if bias_path else Path("outputs/optimal_bias.json")
        if bp.exists():
            with open(bp) as f:
                data = json.load(f)
            bias = data.get("optimal_bias", _DEFAULT_BIAS)
            print(f"  Loaded optimal bias from {bp}")
        else:
            print(f"  Using default bias (run bias_search.py to tune): {_DEFAULT_BIAS}")

        return cls(
            model_best=model_best,
            model_latest=model_latest,
            device=device,
            image_size=cfg.get("image_size", 768),
            bias=bias,
            use_tta=use_tta,
        )

    # ── Core Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _forward_ensemble(self, x: torch.Tensor) -> torch.Tensor:
        """
        Weighted ensemble of two models, optionally averaged across TTA flips.
        Returns raw logits (B, C, H, W) float32 before bias.
        """
        def _pred(model, imgs):
            with torch.amp.autocast(device_type="cuda", enabled=(self.device == "cuda")):
                return model(imgs).float()

        logits = self.w_best * _pred(self.model_best, x) + self.w_latest * _pred(self.model_latest, x)

        if self.use_tta:
            # Horizontal flip
            x_h    = torch.flip(x, [3])
            l_h    = torch.flip(
                self.w_best * _pred(self.model_best, x_h) + self.w_latest * _pred(self.model_latest, x_h),
                [3]
            )
            # Vertical flip
            x_v    = torch.flip(x, [2])
            l_v    = torch.flip(
                self.w_best * _pred(self.model_best, x_v) + self.w_latest * _pred(self.model_latest, x_v),
                [2]
            )
            logits = (logits + l_h + l_v) / 3.0

        return logits  # (B, C, H, W)

    def _calibrated_predict(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply bias → argmax. Returns (preds, probs)."""
        biased = logits + self.bias
        probs  = F.softmax(biased, dim=1)
        preds  = biased.argmax(dim=1)
        return preds, probs

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,      # (B, 3, H, W) normalised
        postprocess: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          preds:  (B, H, W) uint8  class indices
          probs:  (B, C, H, W) float32 softmax probabilities
        """
        images = images.to(self.device, non_blocking=True)
        logits = self._forward_ensemble(images)
        preds, probs = self._calibrated_predict(logits)

        preds_np = preds.cpu().numpy().astype(np.uint8)
        probs_np = probs.cpu().numpy()

        if postprocess:
            for b in range(preds_np.shape[0]):
                mask  = preds_np[b]
                logit_acc = probs_np[b]  # (C, H, W) probs serve as logit_acc proxy
                mask = road_gap_fill(mask)
                mask = postprocess_mask(mask, logit_acc)
                mask = bridge_recovery_from_builtup(mask)
                preds_np[b] = mask

        return preds_np, probs_np

    def patch_stats(self, mask: np.ndarray, pixel_size_m: float = 0.2) -> dict:
        """Compute quantitative infrastructure stats for a single patch."""
        rooftop_mask = classify_rooftops(mask)
        stats = get_infrastructure_summary(mask, rooftop_mask)
        total_px = mask.size
        px_area  = pixel_size_m * pixel_size_m  # m² per pixel

        # Convert pixel counts to physical units
        road_px   = int((mask == 1).sum())
        water_px  = int((mask == 4).sum())
        bu_px     = int((mask == 3).sum())

        stats["road_length_m"]  = round(road_px  * pixel_size_m, 1)   # approx (1px wide road)
        stats["water_area_m2"]  = round(water_px * px_area, 1)
        stats["builtup_area_m2"] = round(bu_px   * px_area, 1)
        return stats
