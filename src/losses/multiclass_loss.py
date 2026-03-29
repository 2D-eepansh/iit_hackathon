"""Multi-class composite loss: Focal + class-weighted Dice for 4-class segmentation.

Class-weighted Focal Loss to combat severe class imbalance
(Bridge < 0.1% of pixels) plus per-class weighted Dice that
boosts Road (thin linear features) relative to Built-Up.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Default class weights for SVAMITVA imbalance ────────────────────────────
# [Background, Road, Bridge, Built-Up]
SVAMITVA_CLASS_WEIGHTS = torch.tensor([0.1, 1.0, 3.0, 1.2])

# Per-class Dice weights for foreground classes [Road, Bridge, Built-Up]
# Road gets 2× weight because it's a thin linear feature harder to segment.
SVAMITVA_DICE_WEIGHTS = torch.tensor([2.0, 1.0, 1.0])


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
