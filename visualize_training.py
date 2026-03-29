"""Training visualization: 2D and 3D plots from training_history.json."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HISTORY_PATH = Path("outputs/training_history.json")
PLOTS_DIR    = Path("outputs/plots")

CLASS_NAMES  = {0: "Background", 1: "Road", 2: "Bridge", 3: "Built-Up"}
COLORS       = {"train_loss": "#e74c3c", "val_loss": "#3498db",
                "val_iou":    "#2ecc71", "val_dice": "#f39c12",
                1: "#3498db", 2: "#e74c3c", 3: "#2ecc71"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_history(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _add_best_marker(ax, epochs, values, label="best"):
    """Mark the epoch with the highest value."""
    best_idx = int(np.argmax(values))
    ax.axvline(epochs[best_idx], color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.scatter([epochs[best_idx]], [values[best_idx]],
               zorder=5, s=60, color="gold", edgecolors="k", linewidths=0.7)
    ax.annotate(f"{label}\n{values[best_idx]:.3f}",
                xy=(epochs[best_idx], values[best_idx]),
                xytext=(6, -14), textcoords="offset points",
                fontsize=7, color="gray")


# ──────────────────────────────────────────────────────────────────────────────
# 2-D  plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(history: list[dict], out_dir: Path) -> None:
    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, train_loss, color=COLORS["train_loss"], lw=2, label="Train Loss")
    ax.plot(epochs, val_loss,   color=COLORS["val_loss"],   lw=2, label="Val Loss", linestyle="--")

    best_idx = int(np.argmin(val_loss))
    ax.axvline(epochs[best_idx], color="gray", linestyle=":", alpha=0.6, linewidth=0.9)
    ax.scatter([epochs[best_idx]], [val_loss[best_idx]], zorder=5, s=70,
               color="gold", edgecolors="k", linewidths=0.8, label=f"Best val={val_loss[best_idx]:.4f}")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'loss_curves.png'}")


def plot_iou_dice(history: list[dict], out_dir: Path) -> None:
    epochs   = [h["epoch"]   for h in history]
    val_iou  = [h["val_iou"] for h in history]
    val_dice = [h["val_dice"] for h in history]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, val_iou,  color=COLORS["val_iou"],  lw=2, label="Val mIoU")
    ax.plot(epochs, val_dice, color=COLORS["val_dice"], lw=2, label="Val mDice", linestyle="--")
    _add_best_marker(ax, epochs, val_iou, "best IoU")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Validation mIoU & mDice")
    ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "iou_dice.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'iou_dice.png'}")


def plot_per_class_iou(history: list[dict], out_dir: Path) -> None:
    if "per_class_iou" not in history[0]:
        return
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 5))
    for cls_id, cls_name in CLASS_NAMES.items():
        key = str(cls_id)
        vals = [h["per_class_iou"].get(key, 0.0) for h in history]
        ax.plot(epochs, vals, lw=2, label=f"Class {cls_id}: {cls_name}",
                color=COLORS.get(cls_id))

    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU")
    ax.set_title("Per-Class IoU over Epochs")
    ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_iou.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'per_class_iou.png'}")


def plot_lr_schedule(history: list[dict], out_dir: Path) -> None:
    if "lr_encoder" not in history[0]:
        return
    epochs  = [h["epoch"]     for h in history]
    lr_enc  = [h["lr_encoder"] for h in history]
    lr_dec  = [h["lr_decoder"] for h in history]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.semilogy(epochs, lr_enc, lw=2, label="Encoder LR", color="#9b59b6")
    ax.semilogy(epochs, lr_dec, lw=2, label="Decoder LR", color="#e67e22", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate (log)")
    ax.set_title("Learning Rate Schedule")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "lr_schedule.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'lr_schedule.png'}")


def plot_summary_grid(history: list[dict], out_dir: Path) -> None:
    """4-panel 2D summary in a single figure."""
    epochs     = [h["epoch"]     for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_iou    = [h["val_iou"]    for h in history]
    val_dice   = [h["val_dice"]   for h in history]

    has_lr     = "lr_encoder" in history[0]
    has_cls    = "per_class_iou" in history[0]
    n_panels   = 2 + int(has_lr) + int(has_cls)
    ncols      = 2
    nrows      = (n_panels + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    idx  = 0

    # Panel 0 – Loss
    ax = axes[idx]; idx += 1
    ax.plot(epochs, train_loss, color=COLORS["train_loss"], lw=2, label="Train")
    ax.plot(epochs, val_loss,   color=COLORS["val_loss"],   lw=2, label="Val", linestyle="--")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 1 – IoU/Dice
    ax = axes[idx]; idx += 1
    ax.plot(epochs, val_iou,  color=COLORS["val_iou"],  lw=2, label="mIoU")
    ax.plot(epochs, val_dice, color=COLORS["val_dice"], lw=2, label="mDice")
    ax.set_ylim(0, 1.05); ax.set_title("Val mIoU & mDice"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(True, alpha=0.3)

    if has_cls:
        ax = axes[idx]; idx += 1
        for cls_id, cls_name in CLASS_NAMES.items():
            vals = [h["per_class_iou"].get(str(cls_id), 0.0) for h in history]
            ax.plot(epochs, vals, lw=2, label=f"C{cls_id}:{cls_name}", color=COLORS.get(cls_id))
        ax.set_ylim(0, 1.05); ax.set_title("Per-Class IoU"); ax.set_xlabel("Epoch")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    if has_lr:
        ax = axes[idx]; idx += 1
        ax.semilogy(epochs, [h["lr_encoder"] for h in history], lw=2, label="Encoder", color="#9b59b6")
        ax.semilogy(epochs, [h["lr_decoder"] for h in history], lw=2, label="Decoder", color="#e67e22")
        ax.set_title("LR Schedule"); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)

    # hide unused panels
    for i in range(idx, len(axes)):
        axes[i].set_visible(False)

    for ax in axes[:idx]:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("Training Summary", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'summary_2d.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# 3-D  plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_3d_per_class_trajectory(history: list[dict], out_dir: Path) -> None:
    """3D surface: X=epoch, Y=class, Z=IoU (one colored bar/line per class)."""
    if "per_class_iou" not in history[0]:
        return

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    epochs      = np.array([h["epoch"] for h in history])
    class_ids   = [1, 2, 3]  # foreground only
    class_names = [CLASS_NAMES[c] for c in class_ids]

    # Build Z matrix: shape (n_classes, n_epochs)
    Z = np.array([
        [h["per_class_iou"].get(str(c), 0.0) for h in history]
        for c in class_ids
    ])

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection="3d")

    cmap_names = ["Blues", "Reds", "Greens"]
    for i, (cid, cname, cmap_n) in enumerate(zip(class_ids, class_names, cmap_names)):
        y_pos   = np.full_like(epochs, fill_value=i, dtype=float)
        z_vals  = Z[i]
        cmap    = plt.get_cmap(cmap_n)
        norm_z  = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)
        colors  = cmap(0.4 + 0.5 * norm_z)

        # Filled ribbon via polygon
        verts_x = np.concatenate([[epochs[0]], epochs, [epochs[-1]]])
        verts_z = np.concatenate([[0], z_vals, [0]])
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        poly_xy = list(zip(verts_x, np.full_like(verts_x, i), verts_z))
        poly    = Poly3DCollection([poly_xy], alpha=0.35, facecolor=cmap(0.55), edgecolor="none")
        ax.add_collection3d(poly)

        # Line on top
        ax.plot(epochs, y_pos, z_vals, lw=2, color=cmap(0.7), label=f"C{cid}: {cname}")

    ax.set_xlabel("Epoch", labelpad=8)
    ax.set_ylabel("Class", labelpad=8)
    ax.set_zlabel("IoU",   labelpad=8)
    ax.set_yticks(range(len(class_ids)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_zlim(0, 1)
    ax.set_title("Per-Class IoU Trajectory (3D)")
    ax.view_init(elev=25, azim=-55)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "3d_per_class_iou.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / '3d_per_class_iou.png'}")


def plot_3d_loss_iou_surface(history: list[dict], out_dir: Path) -> None:
    """3D line: epoch on X, train_loss on Y, val_iou on Z — training trajectory."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    epochs     = np.array([h["epoch"]     for h in history], dtype=float)
    train_loss = np.array([h["train_loss"] for h in history])
    val_iou    = np.array([h["val_iou"]    for h in history])

    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # Color each segment by val_iou progress
    norm    = plt.Normalize(val_iou.min(), val_iou.max())
    cmap    = plt.get_cmap("viridis")
    sm      = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i in range(len(epochs) - 1):
        mid_iou = (val_iou[i] + val_iou[i + 1]) / 2
        color   = cmap(norm(mid_iou))
        ax.plot(epochs[i:i+2], train_loss[i:i+2], val_iou[i:i+2],
                color=color, lw=2.5, alpha=0.9)

    # Start / end markers
    ax.scatter(*[epochs[[0]], train_loss[[0]], val_iou[[0]]],
               color="red",  s=60, zorder=6, label="Start")
    ax.scatter(*[epochs[[-1]], train_loss[[-1]], val_iou[[-1]]],
               color="lime", s=60, zorder=6, label="End")

    # Best epoch
    best_idx = int(np.argmax(val_iou))
    ax.scatter(*[epochs[[best_idx]], train_loss[[best_idx]], val_iou[[best_idx]]],
               color="gold", s=80, zorder=7, edgecolors="k", linewidths=0.8, label="Best IoU")

    plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.6, label="Val IoU")
    ax.set_xlabel("Epoch",      labelpad=8)
    ax.set_ylabel("Train Loss", labelpad=8)
    ax.set_zlabel("Val IoU",    labelpad=8)
    ax.set_title("Training Trajectory (3D)\nEpoch → Train Loss → Val IoU")
    ax.view_init(elev=20, azim=-60)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "3d_training_trajectory.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / '3d_training_trajectory.png'}")


def plot_3d_metric_surface(history: list[dict], out_dir: Path) -> None:
    """3D surface: epoch × class → IoU heat surface (filled mesh)."""
    if "per_class_iou" not in history[0]:
        return
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    epochs    = np.array([h["epoch"] for h in history], dtype=float)
    class_ids = [0, 1, 2, 3]

    Z   = np.array([[h["per_class_iou"].get(str(c), 0.0) for h in history] for c in class_ids])
    X, Y = np.meshgrid(epochs, np.arange(len(class_ids)))

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.82, edgecolor="none")

    plt.colorbar(surf, ax=ax, pad=0.1, shrink=0.6, label="IoU")
    ax.set_xlabel("Epoch",  labelpad=8)
    ax.set_ylabel("Class",  labelpad=8)
    ax.set_zlabel("IoU",    labelpad=8)
    ax.set_yticks(range(len(class_ids)))
    ax.set_yticklabels([CLASS_NAMES[c] for c in class_ids], fontsize=8)
    ax.set_zlim(0, 1)
    ax.set_title("IoU Surface: All Classes over Epochs")
    ax.view_init(elev=30, azim=-50)
    fig.tight_layout()
    fig.savefig(out_dir / "3d_iou_surface.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / '3d_iou_surface.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Combined summary  (2D + 3D mosaic)
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_table(history: list[dict]) -> None:
    best_iou_idx = int(np.argmax([h["val_iou"] for h in history]))
    best         = history[best_iou_idx]
    last         = history[-1]

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total epochs:      {last['epoch']}")
    print(f"  Best Val mIoU:     {best['val_iou']:.4f}  @ epoch {best['epoch']}")
    print(f"  Best Val mDice:    {best['val_dice']:.4f}")
    print(f"  Final Val mIoU:    {last['val_iou']:.4f}")
    print(f"  Final Train Loss:  {last['train_loss']:.4f}")
    if "per_class_iou" in best:
        print("\n  Per-class IoU at best epoch:")
        for c, name in CLASS_NAMES.items():
            v = best["per_class_iou"].get(str(c), 0.0)
            bar = "█" * int(v * 20)
            print(f"    {name:<12} {v:.4f}  {bar}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(history_path: Path = HISTORY_PATH) -> None:
    if not history_path.exists():
        print(f"[ERROR] History file not found: {history_path}")
        print("        Run train.py first to generate training_history.json")
        sys.exit(1)

    history = load_history(history_path)
    if not history:
        print("[ERROR] History file is empty.")
        sys.exit(1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(history)} epoch(s) from {history_path}")
    print(f"Saving plots to {PLOTS_DIR}/\n")

    print("── 2D Plots ──────────────────────────────────")
    plot_loss_curves(history, PLOTS_DIR)
    plot_iou_dice(history, PLOTS_DIR)
    plot_per_class_iou(history, PLOTS_DIR)
    plot_lr_schedule(history, PLOTS_DIR)
    plot_summary_grid(history, PLOTS_DIR)

    print("\n── 3D Plots ──────────────────────────────────")
    plot_3d_per_class_trajectory(history, PLOTS_DIR)
    plot_3d_loss_iou_surface(history, PLOTS_DIR)
    plot_3d_metric_surface(history, PLOTS_DIR)

    print_summary_table(history)
    print(f"All plots saved to: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HISTORY_PATH
    main(path)
