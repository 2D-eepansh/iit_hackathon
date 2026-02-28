"""Multi-class composite loss: Focal + Dice for 4-class segmentation.

Upgraded from plain CrossEntropy to class-weighted Focal Loss to combat
severe class imbalance (Bridge is < 0.1% of pixels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassDiceLoss(nn.Module):
    """Soft Dice loss averaged across all foreground classes (1–N)."""

    def __init__(self, num_classes: int = 4, smooth: float = 1.0, ignore_index: int = -1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) — raw model outputs
            targets: (B, H, W)    — long class indices 0..C-1

        Returns:
            Scalar mean Dice loss over classes 1..C-1 (foreground only).
        """
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets → (B, C, H, W)
        targets_one_hot = F.one_hot(targets.clamp(0), num_classes=self.num_classes)  # (B,H,W,C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()              # (B,C,H,W)

        # Compute per-class Dice (skip class 0 background to focus on features)
        dice_losses = []
        for c in range(1, self.num_classes):
            p = probs[:, c].reshape(probs.shape[0], -1)       # (B, H*W)
            t = targets_one_hot[:, c].reshape(probs.shape[0], -1)
            intersection = (p * t).sum(dim=1)
            cardinality = p.sum(dim=1) + t.sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_losses.append(1.0 - dice.mean())

        return torch.stack(dice_losses).mean()


# ── Default class weights for SVAMITVA imbalance ────────────────────────────
# [Background, Road, Bridge, Built-Up]
# Background is heavily down-weighted; Bridge boosted 30×.
SVAMITVA_CLASS_WEIGHTS = torch.tensor([0.1, 1.0, 3.0, 1.2])


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for handling extreme class imbalance.

    Applies per-pixel focal scaling  ``alpha * (1 - p_t)^gamma * CE``
    so that well-classified background pixels are down-weighted and rare
    classes like Bridge receive stronger gradients.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            gamma:     Focusing parameter (higher → harder example focus).
            alpha:     Per-class weight tensor of shape (C,).  If *None* all
                       classes are weighted equally.
            reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
        """
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)       # moves with .to(device)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) — raw model outputs.
            targets: (B, H, W)    — long class indices 0..C-1.

        Returns:
            Scalar focal loss (when reduction='mean').
        """
        # Per-pixel CE without reduction
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (B, H, W)
        pt = torch.exp(-ce_loss)                                      # p(correct class)
        focal_weight = (1.0 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # Apply per-class alpha weighting
        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            alpha_t = alpha[targets]                   # (B, H, W)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MultiClassCompositeLoss(nn.Module):
    """Focal + Dice composite loss for multi-class segmentation.

    Drop-in replacement for the previous CE + Dice loss — the constructor
    accepts the same ``ce_weight`` / ``dice_weight`` parameters so that
    ``train.py`` CONFIG does not need changes.  Internally ``ce_weight``
    now controls the Focal Loss component.
    """

    def __init__(
        self,
        num_classes: int = 4,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        class_weights: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
    ) -> None:
        """
        Args:
            num_classes:   Number of output classes (including background).
            ce_weight:     Weight for Focal Loss component (was CE before).
            dice_weight:   Weight for Dice component.
            smooth:        Dice smoothing constant.
            class_weights: Per-class alpha for Focal Loss.  If *None*,
                           ``SVAMITVA_CLASS_WEIGHTS`` are used.
            focal_gamma:   Focal Loss focusing parameter.
        """
        super().__init__()
        self.focal_weight = ce_weight          # renamed internally for clarity
        self.dice_weight = dice_weight

        if class_weights is None:
            class_weights = SVAMITVA_CLASS_WEIGHTS[:num_classes].clone()

        self.focal = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        self.dice = MultiClassDiceLoss(num_classes=num_classes, smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) — raw model outputs.
            targets: (B, H, W)    — long class indices.

        Returns:
            Weighted sum of Focal and Dice losses.
        """
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
