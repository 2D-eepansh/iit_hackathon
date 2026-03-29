"""Model factory for creating segmentation models."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def create_model(
    architecture: str = "DeepLabV3Plus",
    encoder_name: str = "resnet50",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 4,
    use_gradient_checkpointing: bool = False,
) -> nn.Module:
    """
    Create segmentation model using segmentation_models_pytorch.

    Args:
        architecture: Model architecture ('Unet', 'DeepLabV3Plus', etc.).
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
        activation=None,  # Raw logits (softmax applied in loss / argmax at inference)
    )

    # Gradient checkpointing — try multiple APIs (timm / torchvision)
    if use_gradient_checkpointing:
        enabled = False
        if hasattr(model.encoder, "set_gradient_checkpointing"):
            model.encoder.set_gradient_checkpointing(enable=True)
            enabled = True
        elif hasattr(model.encoder, "gradient_checkpointing"):
            model.encoder.gradient_checkpointing = True
            enabled = True
        else:
            # Manual: wrap encoder forward with checkpoint
            _orig_forward = model.encoder.forward
            def _ckpt_forward(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(_orig_forward, *args, use_reentrant=False, **kwargs)
            model.encoder.forward = _ckpt_forward
            enabled = True
        if enabled:
            print("  Gradient checkpointing: ENABLED")

    return model
