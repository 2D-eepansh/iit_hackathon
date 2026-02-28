"""Model factory for creating segmentation models."""

import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


def create_model(
    architecture: str = "Unet",
    encoder_name: str = "efficientnet-b4",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    use_gradient_checkpointing: bool = False,
) -> nn.Module:
    """
    Create segmentation model using segmentation_models_pytorch.

    Args:
        architecture: Model architecture ('Unet', 'UnetPlusPlus', etc.).
        encoder_name: Encoder backbone name.
        encoder_weights: Pretrained weights for encoder.
        in_channels: Number of input channels.
        classes: Number of output classes.
        use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency.

    Returns:
        Segmentation model.
    """
    model_class = getattr(smp, architecture)
    
    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # No activation for BCEWithLogitsLoss
    )
    
    # Optional gradient checkpointing
    if use_gradient_checkpointing and hasattr(model.encoder, "set_gradient_checkpointing"):
        model.encoder.set_gradient_checkpointing(enable=True)
    
    return model
