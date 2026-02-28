"""Composite loss combining BCE and Dice for segmentation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Vectorized Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        """
        Initialize DiceLoss.

        Args:
            smooth: Smoothing constant to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            logits: Raw model outputs (B, C, H, W).
            targets: Ground truth masks (B, C, H, W) with values in {0, 1}.

        Returns:
            Dice loss scalar.
        """
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions only, keep batch
        probs_flat = probs.flatten(1)
        targets_flat = targets.flatten(1)

        # Compute Dice per sample
        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Return mean Dice loss
        return 1.0 - dice_score.mean()


class CompositeLoss(nn.Module):
    """Composite loss combining BCE and Dice."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        """
        Initialize CompositeLoss.

        Args:
            bce_weight: Weight for BCE loss component.
            dice_weight: Weight for Dice loss component.
            smooth: Smoothing constant for Dice loss.
            pos_weight: Positive class weight for imbalanced data.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute composite loss.

        Args:
            logits: Raw model outputs (B, C, H, W).
            targets: Ground truth masks (B, C, H, W) with values in {0, 1}.

        Returns:
            Weighted sum of BCE and Dice losses.
        """
        if self.bce.pos_weight is not None:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
