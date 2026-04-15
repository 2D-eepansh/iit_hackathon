"""Multi-class composite loss for SVAMITVA 5-class segmentation.

V1: Focal + class-weighted Dice  (kept for backward compat)
V2: OHEM-CE + label-smoothed Dice with conditional bridge
    — fixes the oscillation root cause identified in Phase 4 training:
      (a) Bridge Focal alpha=5.0 on near-zero GT caused gradient noise
      (b) Focal+Dice double-compensate imbalance → competing gradients
      (c) Low confidence from logit over-smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── V1 weights (Focal+Dice) — kept for compat ────────────────────────────────
SVAMITVA_CLASS_WEIGHTS = torch.tensor([0.1, 1.5, 5.0, 1.2, 2.5])
SVAMITVA_DICE_WEIGHTS  = torch.tensor([2.5, 1.5, 1.0, 2.0])

# ── V2 weights (OHEM+SmoothedDice) ───────────────────────────────────────────
# Bridge alpha removed from OHEM — bridge gradient noise is now suppressed
# by the conditional Dice (skipped when bridge GT < threshold).
# Moderate water weight (2.0 not 2.5) — let OHEM drive hard-pixel focus.
SVAMITVA_CLASS_WEIGHTS_V2 = torch.tensor([0.1, 1.5, 1.5, 1.2, 2.0])
SVAMITVA_DICE_WEIGHTS_V2  = torch.tensor([2.5, 2.0, 1.0, 2.0])


class MultiClassDiceLoss(nn.Module):
    """Soft Dice loss with per-class weighting across foreground classes (1..N)."""

    def __init__(
        self,
        num_classes: int = 4,
        smooth: float = 1e-4,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        # Weights for foreground classes 1..C-1
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", torch.ones(num_classes - 1))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp(0), num_classes=self.num_classes)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()

        dice_losses = []
        for i, c in enumerate(range(1, self.num_classes)):
            p = probs[:, c].reshape(probs.shape[0], -1)
            t = targets_oh[:, c].reshape(probs.shape[0], -1)
            intersection = (p * t).sum(dim=1)
            cardinality = p.sum(dim=1) + t.sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            w = self.class_weights[i]
            dice_losses.append(w * (1.0 - dice.mean()))

        return torch.stack(dice_losses).sum() / self.class_weights.sum()


class FocalLoss(nn.Module):
    """Multi-class Focal Loss: ``alpha * (1 - p_t)^gamma * CE``."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MultiClassCompositeLoss(nn.Module):
    """Focal + class-weighted Dice composite loss."""

    def __init__(
        self,
        num_classes: int = 4,
        ce_weight: float = 0.6,
        dice_weight: float = 0.4,
        smooth: float = 1e-4,
        class_weights: torch.Tensor | None = None,
        dice_class_weights: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.focal_weight = ce_weight
        self.dice_weight = dice_weight

        if class_weights is None:
            class_weights = SVAMITVA_CLASS_WEIGHTS[:num_classes].clone()
        if dice_class_weights is None:
            dice_class_weights = SVAMITVA_DICE_WEIGHTS[: num_classes - 1].clone()

        self.focal = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        self.dice = MultiClassDiceLoss(
            num_classes=num_classes, smooth=smooth, class_weights=dice_class_weights,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


# ─────────────────────────────────────────────────────────────────────────────
# V2 LOSSES  —  OHEM-CE + label-smoothed conditional Dice
# ─────────────────────────────────────────────────────────────────────────────


class OHEMLoss(nn.Module):
    """Online Hard Example Mining Cross-Entropy.

    Selects the ``top_ratio`` fraction of pixels with the highest CE loss
    and computes the mean CE only over those.  No per-class alpha weighting
    is used — hard examples are selected purely by prediction confidence,
    giving focused gradients without the noise from Focal's soft reweighting
    on near-absent classes (e.g. Bridge with ~0 GT pixels).
    """

    def __init__(self, top_ratio: float = 0.30) -> None:
        super().__init__()
        self.top_ratio = top_ratio

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ce_map = F.cross_entropy(logits, targets, reduction="none")  # (B, H, W)
            n_hard = max(1, int(self.top_ratio * ce_map.numel()))
            # topk on flattened — min of topk gives the threshold
            threshold = ce_map.flatten().topk(n_hard, largest=True, sorted=False).values.min()

        hard_mask = ce_map.detach() >= threshold
        return F.cross_entropy(logits, targets, reduction="none")[hard_mask].mean()


class SmoothedDiceLoss(nn.Module):
    """Soft Dice with label smoothing and conditional bridge skip.

    Label smoothing (eps=0.05): prevents logit over-confidence on dominant
    classes (Background/Built-Up), which was the root cause of the collapse
    oscillation — the model aggressively drove logits to ±∞ attempting to
    satisfy the hard one-hot targets.

    Conditional bridge: when the batch has fewer than ``bridge_min_pixels``
    bridge GT pixels, the bridge Dice term is zero.  This eliminates the
    gradient noise from the 5× bridge weight on effectively-empty batches.
    """

    def __init__(
        self,
        num_classes: int = 5,
        smooth: float = 1e-4,
        label_eps: float = 0.05,
        class_weights: torch.Tensor | None = None,
        bridge_min_pixels: int = 100,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.label_eps = label_eps
        self.bridge_min_pixels = bridge_min_pixels
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", torch.ones(num_classes - 1))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)

        # Label-smoothed one-hot targets
        targets_oh = F.one_hot(targets.clamp(0), num_classes=self.num_classes)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()
        targets_smooth = (
            targets_oh * (1.0 - self.label_eps)
            + self.label_eps / self.num_classes
        )

        dice_losses: list[torch.Tensor] = []
        total_weight = 0.0

        for i, c in enumerate(range(1, self.num_classes)):
            # Conditional bridge: skip when near-absent in this batch
            if c == 2 and (targets == 2).sum().item() < self.bridge_min_pixels:
                continue

            p = probs[:, c].reshape(probs.shape[0], -1)
            t = targets_smooth[:, c].reshape(probs.shape[0], -1)
            intersection = (p * t).sum(dim=1)
            cardinality  = p.sum(dim=1) + t.sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            w = float(self.class_weights[i].item())
            dice_losses.append(w * (1.0 - dice.mean()))
            total_weight += w

        if not dice_losses:
            return logits.sum() * 0.0   # zero but autograd-safe

        return torch.stack(dice_losses).sum() / max(total_weight, 1e-9)


class MultiClassCompositeLossV2(nn.Module):
    """OHEM-CE + label-smoothed conditional Dice.

    Drop-in replacement for MultiClassCompositeLoss that fixes the three
    failure modes identified in Phase 4:
      1. Bridge gradient noise       → conditional Dice skip
      2. Double class-balance compen → OHEM uses no alpha weights
      3. Logit over-smoothing        → label smoothing (eps=0.05)
    """

    def __init__(
        self,
        num_classes: int = 5,
        ohem_ratio: float = 0.30,
        ohem_weight: float = 0.5,
        dice_weight: float = 0.5,
        label_eps: float = 0.05,
        bridge_min_pixels: int = 100,
        # kept for compat with callers using the old signature
        class_weights: torch.Tensor | None = None,
        dice_class_weights: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ohem_weight = ohem_weight
        self.dice_weight = dice_weight

        if dice_class_weights is None:
            dice_class_weights = SVAMITVA_DICE_WEIGHTS_V2[: num_classes - 1].clone()

        self.ohem = OHEMLoss(top_ratio=ohem_ratio)
        self.dice = SmoothedDiceLoss(
            num_classes=num_classes,
            label_eps=label_eps,
            class_weights=dice_class_weights,
            bridge_min_pixels=bridge_min_pixels,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.ohem_weight * self.ohem(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )
