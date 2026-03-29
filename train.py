"""Main training script for SVAMITVA multi-class feature extraction."""

import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.unified_dataset import (
    UnifiedMultiClassDataset,
    DEFAULT_SOURCES,
    get_train_transform,
    get_val_transform,
)
from src.losses.multiclass_loss import MultiClassCompositeLoss
from src.models.model_factory import create_model
from src.training.train_one_epoch import train_one_epoch, validate_multiclass


# ── TIFF-level train / val split (no spatial leakage) ─────────────────────
TRAIN_TIFFS = [
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho",          # PB
    "TIMMOWAL_37695_ORI",                                # PB
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO",  # CG-2
    "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO",        # CG-3  (has 5 bridges)
    "KUTRU_451189_AAKLANKA_451163_ORTHO",                # CG-2  (corrupt, auto-skipped)
    "SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO",  # CG-3  (corrupt, auto-skipped)
]
VAL_TIFFS = [
    "28996_NADALA_ORTHO",                                # PB
    "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO",   # CG-3  (has 1 bridge)
]

# Training configuration
CONFIG = {
    # Data
    "image_size": 768,
    "patches_per_image": 150,   # ↑ from 100 for more diversity per epoch

    # Training
    "batch_size": 4,
    "num_epochs": 80,
    "learning_rate": 1e-4,
    "encoder_lr": 1e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "accumulation_steps": 4,    # effective batch 16

    # Model
    "architecture": "DeepLabV3Plus",
    "encoder_name": "resnet50",
    "encoder_weights": "imagenet",
    "classes": 4,               # 0=Background, 1=Road, 2=Bridge, 3=Built-Up
    "seed": 42,
    "use_gradient_checkpointing": True,

    # Validation enhancements (TTA disabled during training — too slow)
    "use_multiscale_val": True,
    "use_road_refinement": True,
    "use_tta": False,           # disabled: saves 57% val time per epoch

    # Loss  (Focal + class-weighted Dice)
    "ce_weight": 0.6,
    "dice_weight": 0.4,

    # EMA
    "ema_decay": 0.999,

    # Scheduler
    "scheduler_type": "plateau",

    # DataLoader
    "num_workers": min(4, os.cpu_count() or 4),
    "persistent_workers": True,
    "prefetch_factor": 2,

    # Output
    "output_dir": "outputs",
    "checkpoint_dir": "outputs/checkpoints",

    # Resume
    "resume_checkpoint": None,
}


# ── EMA (Exponential Moving Average) ──────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model weights that smoothly tracks the
    training weights.  Eliminates epoch-to-epoch variance and prevents
    catastrophic forgetting on minority classes (Built-Up collapse fix).
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        self.backup: dict | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.state_dict().items():
            if self.shadow[name].is_floating_point():
                self.shadow[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(param)  # integer buffers (e.g. num_batches_tracked)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Swap model weights with EMA weights (for validation)."""
        self.backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model: torch.nn.Module) -> None:
        """Restore original model weights after validation."""
        if self.backup is not None:
            model.load_state_dict(self.backup)
            self.backup = None

    def state_dict(self) -> dict:
        return self.shadow


# ── Helpers ────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def setup_cuda_optimizations() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create train and val dataloaders with TIFF-level split."""
    train_dataset = UnifiedMultiClassDataset(
        sources=DEFAULT_SOURCES,
        split="train",
        transform=get_train_transform(config["image_size"]),
        patch_size=config["image_size"],
        patches_per_image=config.get("patches_per_image", 100),
        positive_sampling_prob=0.9,    # ↑ from 0.7 — forces more retries for positive patches
        train_tiffs=TRAIN_TIFFS,
        val_tiffs=VAL_TIFFS,
    )

    val_dataset = UnifiedMultiClassDataset(
        sources=DEFAULT_SOURCES,
        split="val",
        transform=get_val_transform(config["image_size"]),
        patch_size=config["image_size"],
        patches_per_image=config.get("patches_per_image", 100),
        train_tiffs=TRAIN_TIFFS,
        val_tiffs=VAL_TIFFS,
    )

    nw = config["num_workers"]
    pw = config["persistent_workers"] if nw > 0 else False
    pf = config["prefetch_factor"] if nw > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=pf,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=pf,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    config = CONFIG

    set_seed(config["seed"])
    setup_cuda_optimizations()
    Path(config["output_dir"]).mkdir(exist_ok=True)
    Path(config["checkpoint_dir"]).mkdir(exist_ok=True)

    if config["batch_size"] < 2:
        raise ValueError("batch_size must be >= 2 (BatchNorm constraint)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("TRAINING CONFIGURATION  —  SVAMITVA Multi-Class Pipeline")
    print("=" * 80)
    print(f"Device:        {device}")
    if torch.cuda.is_available():
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
        print(f"CUDA:          {torch.version.cuda}")
    print(f"Dataset:       Unified PB + CG ({len(TRAIN_TIFFS)} train, {len(VAL_TIFFS)} val TIFFs)")
    print(f"Classes:       {config['classes']}  "
          "(0=BG, 1=Road, 2=Bridge, 3=Built-Up Area)")
    print(f"Reproducibility active: seed={config['seed']}")
    print(f"Model:         {config['architecture']} ({config['encoder_name']} backbone)")
    print(f"Image size:    {config['image_size']}")
    print(f"Batch size:    {config['batch_size']}  (effective {config['batch_size'] * config['accumulation_steps']})")
    print(f"Epochs:        {config['num_epochs']}")
    print(f"EMA decay:     {config['ema_decay']}")
    if config.get("use_road_refinement"):
        print("Road structural refinement: ENABLED")
    if config.get("use_multiscale_val"):
        print("Multi-scale validation: ENABLED")
    print(f"TTA during training: {'ENABLED' if config.get('use_tta') else 'DISABLED (fast val)'}")
    print("=" * 80)

    # Create dataloaders
    print("\nCreating SVAMITVA dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        architecture=config["architecture"],
        encoder_name=config["encoder_name"],
        encoder_weights=config["encoder_weights"],
        in_channels=3,
        classes=config["classes"],
        use_gradient_checkpointing=config["use_gradient_checkpointing"],
    )
    model = model.to(device)

    # EMA
    ema = EMA(model, decay=config["ema_decay"])
    print(f"  EMA tracking: ENABLED (decay={config['ema_decay']})")

    # Loss
    criterion = MultiClassCompositeLoss(
        num_classes=config["classes"],
        ce_weight=config["ce_weight"],
        dice_weight=config["dice_weight"],
    )

    # Optimizer with differential learning rates
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config["encoder_lr"]},
            {"params": decoder_params, "lr": config["learning_rate"]},
        ],
        weight_decay=config["weight_decay"],
    )

    # Scheduler
    scheduler_type = config.get("scheduler_type", "plateau")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["num_epochs"], eta_min=1e-6,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
            threshold=1e-4, min_lr=1e-6,
        )

    scaler = GradScaler("cuda")

    print(f"Initial LR (encoder): {optimizer.param_groups[0]['lr']}")
    print(f"Initial LR (decoder): {optimizer.param_groups[1]['lr']}")
    print(f"Scheduler: {type(scheduler).__name__}")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_iou = 0.0
    training_history: list[dict] = []
    history_path = Path(config["output_dir"]) / "training_history.json"

    resume_path = config.get("resume_checkpoint")
    if resume_path and Path(resume_path).exists():
        print(f"\nResuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", 0.0)
        if "ema_state_dict" in ckpt:
            ema.shadow = ckpt["ema_state_dict"]
        if history_path.exists():
            with open(history_path) as f:
                training_history = json.load(f)
        print(f"Resumed at epoch {start_epoch}, best_iou={best_iou:.4f}")

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    epoch_times = []

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        epoch_start = time.time()

        print(f"\n{'─' * 80}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'─' * 80}")

        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            max_grad_norm=config["max_grad_norm"],
            accumulation_steps=config["accumulation_steps"],
        )

        # Update EMA after each training epoch
        ema.update(model)

        # Validate using EMA weights (smoother, more stable)
        ema.apply_shadow(model)
        val_metrics = validate_multiclass(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=config["classes"],
            use_multiscale=config.get("use_multiscale_val", False),
            use_road_refinement=config.get("use_road_refinement", False),
            use_tta=config.get("use_tta", False),
        )
        ema.restore(model)

        # Update scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["val_iou"])
        else:
            scheduler.step()

        # GPU memory
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        else:
            mem_allocated = mem_reserved = 0.0

        # ETA
        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = config["num_epochs"] - epoch
        eta_seconds = int(avg_epoch_time * remaining_epochs)
        h, m, s = eta_seconds // 3600, (eta_seconds % 3600) // 60, eta_seconds % 60
        eta_str = f"{h:02d}:{m:02d}:{s:02d}"

        # Persist epoch metrics
        epoch_record = {
            "epoch":         epoch,
            "train_loss":    train_metrics["train_loss"],
            "train_time":    train_metrics["train_time"],
            "val_loss":      val_metrics["val_loss"],
            "val_iou":       val_metrics["val_iou"],
            "val_dice":      val_metrics["val_dice"],
            "per_class_iou": {str(k): v for k, v in val_metrics.get("per_class_iou", {}).items()},
            "per_class_dice":{str(k): v for k, v in val_metrics.get("per_class_dice", {}).items()},
            "lr_encoder":    optimizer.param_groups[0]["lr"],
            "lr_decoder":    optimizer.param_groups[1]["lr"],
            "epoch_time":    epoch_elapsed,
        }
        training_history.append(epoch_record)
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)

        # Print metrics
        print(f"\nTrain Loss:  {train_metrics['train_loss']:.4f}")
        print(f"Val Loss:    {val_metrics['val_loss']:.4f}")
        print(f"Val mIoU:    {val_metrics['val_iou']:.4f}  (EMA)")
        print(f"Val mDice:   {val_metrics['val_dice']:.4f}")
        print(f"Epoch Time:  {epoch_elapsed:.1f}s (train={train_metrics['train_time']:.1f}s)")
        print(f"GPU Memory:  {mem_allocated:.2f}GB / {mem_reserved:.2f}GB")
        print(f"LR enc/dec:  {optimizer.param_groups[0]['lr']:.2e} / {optimizer.param_groups[1]['lr']:.2e}")
        print(f"ETA:         {eta_str}")

        # Save best model (EMA weights)
        if val_metrics["val_iou"] > best_iou:
            best_iou = val_metrics["val_iou"]
            checkpoint_path = Path(config["checkpoint_dir"]) / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "best_iou": best_iou,
                    "config": config,
                    "metrics": {**train_metrics, **val_metrics},
                },
                checkpoint_path,
            )
            print(f"\n✓ Saved best EMA model (mIoU: {best_iou:.4f})")

        # Save latest checkpoint (training weights + EMA)
        checkpoint_path = Path(config["checkpoint_dir"]) / "latest_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "best_iou": best_iou,
                "config": config,
                "metrics": {**train_metrics, **val_metrics},
            },
            checkpoint_path,
        )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation mIoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")
    print(f"Training history:     {history_path}")
    print("=" * 80 + "\n")

    # Generate training visualizations
    print("Generating training plots...")
    try:
        from visualize_training import main as visualize
        visualize(history_path)
    except Exception as exc:
        print(f"[WARN] Visualization failed: {exc}")


if __name__ == "__main__":
    main()
