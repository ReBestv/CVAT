"""
Plan B: U-Net + EfficientNet-b3 fine-tuning via segmentation-models-pytorch.
所有参数在 config.py 修改。直接运行: python src/train_unet.py
"""
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import segmentation_models_pytorch as smp
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    EVAL_EVERY_N_EPOCHS,
    EVAL_BATCH_TIME_LOG_INTERVAL,
    LEARNING_RATE,
    LOG_EVAL_BATCH_TIMES,
    LOSS_ALPHA,
    NUM_CLASSES,
    NUM_WORKERS,
    PHASE1_EPOCHS,
    PHASE1_LR,
    SAVE_TOP_K,
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    UNET_CHECKPOINT_DIR,
    UNET_ENCODER,
    UNET_ENCODER_LOCAL_WEIGHTS,
    UNET_ENCODER_WEIGHTS,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    VIZ_EVERY_N_EPOCHS,
    VIZ_NUM_SAMPLES,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    ID_TO_COLOR,
    IGNORE_INDEX,
    get_device,
)
from dataset import create_train_dataset, create_val_dataset
from losses import ComboLoss


# ============================================================================
# Metrics
# ============================================================================


def compute_iou(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> tuple[float, dict[int, float], np.ndarray]:
    """
    Compute mIoU and per-class IoU.

    Args:
        pred: (B, C, H, W) logits
        target: (B, H, W) class indices
        num_classes: number of classes

    Returns:
        miou: mean IoU
        per_class_iou: dict[class_id → IoU]
        confusion: (C, C) confusion matrix
    """
    pred_cls = pred.argmax(dim=1).to(torch.int64)
    target = target.to(torch.int64)
    mask = (target != IGNORE_INDEX) & (target >= 0) & (target < num_classes)
    bins = target[mask] * num_classes + pred_cls[mask]
    confusion = torch.bincount(
        bins, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).cpu().numpy()

    ious = {}
    for c in range(num_classes):
        intersection = confusion[c, c]
        union = confusion[c, :].sum() + confusion[:, c].sum() - intersection
        ious[c] = intersection / max(union, 1)

    return float(np.mean(list(ious.values()))), ious, confusion


def compute_iou_from_confusion(
    confusion: torch.Tensor, num_classes: int
) -> tuple[float, dict[int, float]]:
    """Compute mIoU from pre-accumulated confusion matrix (memory-efficient)."""
    confusion = confusion.float()
    ious = {}
    for c in range(num_classes):
        intersection = confusion[c, c].item()
        union = confusion[c, :].sum().item() + confusion[:, c].sum().item() - intersection
        ious[c] = intersection / max(union, 1)
    miou = sum(ious.values()) / len(ious)
    return miou, ious


def update_confusion_matrix_(
    confusion: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> None:
    """Accumulate confusion counts without per-pixel Python loops."""
    valid = (target != IGNORE_INDEX) & (target >= 0) & (target < num_classes)
    if not torch.any(valid):
        return

    pred = pred[valid].to(torch.int64)
    target = target[valid].to(torch.int64)
    bins = target * num_classes + pred
    confusion += torch.bincount(
        bins, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


# ============================================================================
# Visualization
# ============================================================================


def visualize_predictions(
    images: torch.Tensor,
    masks_true: torch.Tensor,
    masks_pred: torch.Tensor,
    class_names: list[str],
    save_dir: Path,
    epoch: int,
) -> None:
    """Save side-by-side visualization: image | ground truth | prediction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    num_samples = min(len(images), VIZ_NUM_SAMPLES)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = np.array([axes])

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    cmap = {i: (c[0] / 255, c[1] / 255, c[2] / 255) for i, c in ID_TO_COLOR.items()}

    for i in range(num_samples):
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        true = masks_true[i].cpu().numpy()
        pred_cls = masks_pred.argmax(dim=1)[i].cpu().numpy()

        true_rgb = np.full((*true.shape, 3), 0.75, dtype=np.float32)
        pred_rgb = np.zeros((*pred_cls.shape, 3), dtype=np.float32)
        for c, color in cmap.items():
            true_rgb[true == c] = color
            pred_rgb[pred_cls == c] = color

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(true_rgb)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    legend_patches = [
        Patch(color=cmap[c], label=f"{c}: {name}") for c, name in enumerate(class_names)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=min(6, len(class_names)),
        fontsize=7,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = save_dir / f"viz_epoch_{epoch:03d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved: {save_path}")


# ============================================================================
# Training Utilities
# ============================================================================


def get_linear_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-7,
) -> LambdaLR:
    """Linear warmup → cosine decay."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1.0, float(warmup_steps))
        progress = float(step - warmup_steps) / max(1.0, float(total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    miou: float,
    save_dir: Path,
    is_best: bool = False,
) -> Path:
    """Save model checkpoint."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"epoch_{epoch:03d}_miou_{miou:.4f}.pt" if not is_best else "best.pt"
    path = save_dir / fname
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "miou": miou,
        },
        path,
    )
    return path


def cleanup_old_checkpoints(save_dir: Path, keep: int = SAVE_TOP_K) -> None:
    """Keep only the top N most recent checkpoints."""
    checkpoints = sorted(
        save_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime
    )
    while len(checkpoints) > keep:
        oldest = checkpoints.pop(0)
        oldest.unlink()
        print(f"  Removed old checkpoint: {oldest.name}")


def set_encoder_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    """Freeze/unfreeze the SMP U-Net encoder."""
    for param in model.encoder.parameters():
        param.requires_grad = requires_grad


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Return a plain state dict from common checkpoint layouts."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "encoder_state_dict"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Local encoder weights must be a PyTorch state dict or checkpoint dict.")

    state_dict = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue

        clean_key = key
        for prefix in ("module.", "model.", "encoder."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        state_dict[clean_key] = value

    if not state_dict:
        raise ValueError("No tensor weights found in local encoder checkpoint.")

    return state_dict


def load_local_encoder_weights(
    model: torch.nn.Module, weights_path: Path, device: str
) -> None:
    """Load local ImageNet weights into the SMP encoder."""
    if not weights_path.exists():
        print(f"Local encoder weights not found: {weights_path}")
        print(f"Falling back to SMP encoder_weights={UNET_ENCODER_WEIGHTS!r}")
        return

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    load_result = model.encoder.load_state_dict(state_dict, strict=False)
    if load_result is None:
        missing, unexpected = [], []
    else:
        missing, unexpected = load_result

    encoder_keys = set(model.encoder.state_dict().keys())
    loaded_keys = encoder_keys.intersection(state_dict.keys())
    if not loaded_keys:
        raise RuntimeError(
            f"Local weights did not match encoder keys: {weights_path}"
        )

    print(f"Loaded local encoder weights: {weights_path}")
    if missing:
        print(f"  Missing encoder keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected checkpoint keys: {len(unexpected)}")


# ============================================================================
# Main Training
# ============================================================================


def train() -> None:
    device = get_device()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    encoder_slug = UNET_ENCODER.replace("-", "").replace("_", "")
    run_name = f"unet_{encoder_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = UNET_CHECKPOINT_DIR / run_name
    log_dir = Path("logs") / run_name
    viz_dir = checkpoint_dir / "viz"

    for d in [checkpoint_dir, log_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"Training: {run_name}")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_dataset = create_train_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_dataset = create_val_dataset(VAL_IMG_DIR, VAL_MASK_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    print(f"Train tiles: {len(train_dataset)}, Val tiles: {len(val_dataset)}")
    print(f"Batches per epoch: {len(train_loader)} train, {len(val_loader)} val")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    use_local_encoder_weights = UNET_ENCODER_LOCAL_WEIGHTS.exists()
    encoder_weights = None if use_local_encoder_weights else UNET_ENCODER_WEIGHTS

    print(f"Loading model: U-Net + {UNET_ENCODER} (weights={encoder_weights})")
    model = smp.Unet(
        encoder_name=UNET_ENCODER,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    model.to(device)
    if use_local_encoder_weights:
        load_local_encoder_weights(model, UNET_ENCODER_LOCAL_WEIGHTS, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    criterion = ComboLoss(
        alpha=LOSS_ALPHA, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX
    )

    # ------------------------------------------------------------------
    # Phase 1: Train decoder only (encoder frozen)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 1: Freeze encoder, train decoder ({PHASE1_EPOCHS} epochs)")
    print(f"{'='*60}")

    set_encoder_grad(model, requires_grad=False)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR,
        weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = len(train_loader)
    total_steps_p1 = PHASE1_EPOCHS * steps_per_epoch
    scheduler = get_linear_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=min(WARMUP_STEPS, total_steps_p1 // 4),
        total_steps=total_steps_p1,
    )

    best_miou = 0.0
    best_epoch = 0

    for epoch in range(1, PHASE1_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{PHASE1_EPOCHS} [train]")
        for batch in pbar:
            images = batch["pixel_values"].to(device)
            masks = batch["labels"].to(device)

            logits = model(images)  # SMP output is already (B, C, H, W)
            loss, loss_dict = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_ce += loss_dict["ce"]
            train_dice += loss_dict["dice"]

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{loss_dict['ce']:.4f}",
                dice=f"{loss_dict['dice']:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        n_batches = len(train_loader)
        writer.add_scalar("train/loss", train_loss / n_batches, epoch)
        writer.add_scalar("train/ce_loss", train_ce / n_batches, epoch)
        writer.add_scalar("train/dice_loss", train_dice / n_batches, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if epoch % EVAL_EVERY_N_EPOCHS == 0 or epoch == PHASE1_EPOCHS:
            miou = validate(model, val_loader, criterion, writer, epoch, device)
            print(f"  Val mIoU: {miou:.4f}")

            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, miou, checkpoint_dir, is_best=True)
                print(f"  New best mIoU: {best_miou:.4f} ✓")

            save_checkpoint(model, optimizer, epoch, miou, checkpoint_dir)
            cleanup_old_checkpoints(checkpoint_dir)

            if epoch % VIZ_EVERY_N_EPOCHS == 0:
                _viz_val_samples(model, val_dataset, viz_dir, epoch, device)

    # ------------------------------------------------------------------
    # Phase 2: Unfreeze all, fine-tune
    # ------------------------------------------------------------------
    remaining = EPOCHS - PHASE1_EPOCHS
    print(f"\n{'='*60}")
    print(f"PHASE 2: Unfreeze all, fine-tune ({remaining} epochs)")
    print(f"{'='*60}")

    set_encoder_grad(model, requires_grad=True)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps_p2 = remaining * steps_per_epoch
    scheduler = get_linear_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=min(WARMUP_STEPS, total_steps_p2 // 4),
        total_steps=total_steps_p2,
    )

    patience_counter = 0

    for epoch in range(PHASE1_EPOCHS + 1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for batch in pbar:
            images = batch["pixel_values"].to(device)
            masks = batch["labels"].to(device)

            logits = model(images)
            loss, loss_dict = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_ce += loss_dict["ce"]
            train_dice += loss_dict["dice"]

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{loss_dict['ce']:.4f}",
                dice=f"{loss_dict['dice']:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_loss = train_loss / n_batches
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/ce_loss", train_ce / n_batches, epoch)
        writer.add_scalar("train/dice_loss", train_dice / n_batches, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if epoch % EVAL_EVERY_N_EPOCHS == 0 or epoch == EPOCHS:
            miou = validate(model, val_loader, criterion, writer, epoch, device)
            print(f"  Val mIoU: {miou:.4f} (best: {best_miou:.4f} @ epoch {best_epoch})")

            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, miou, checkpoint_dir, is_best=True)
                print(f"  New best mIoU: {best_miou:.4f} ✓")
            else:
                patience_counter += 1

            save_checkpoint(model, optimizer, epoch, miou, checkpoint_dir)
            cleanup_old_checkpoints(checkpoint_dir)

            if epoch % VIZ_EVERY_N_EPOCHS == 0:
                _viz_val_samples(model, val_dataset, viz_dir, epoch, device)

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} evals)")
            break

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {checkpoint_dir / 'best.pt'}")
    writer.close()


# ============================================================================
# Validation
# ============================================================================


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: ComboLoss,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> float:
    """Run validation and return mIoU (memory-efficient)."""
    model.eval()
    val_loss = 0.0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
    batch_times = []
    data_times = []
    infer_times = []
    pbar = tqdm(range(len(val_loader)), desc="  Validating", leave=False)
    val_iter = iter(val_loader)

    for batch_idx in pbar:
        fetch_start = time.perf_counter()
        batch = next(val_iter)
        data_time = time.perf_counter() - fetch_start

        compute_start = time.perf_counter()
        images = batch["pixel_values"].to(device)
        masks = batch["labels"].to(device)

        logits = model(images)
        loss, _ = criterion(logits, masks)
        val_loss += loss.item()

        # Incremental confusion matrix without per-pixel Python loops.
        pred = logits.argmax(dim=1)
        update_confusion_matrix_(confusion, pred, masks, NUM_CLASSES)
        infer_time = time.perf_counter() - compute_start
        batch_time = data_time + infer_time

        data_times.append(data_time)
        infer_times.append(infer_time)
        batch_times.append(batch_time)
        avg_batch_time = sum(batch_times) / len(batch_times)

        pbar.set_postfix(
            batch_s=f"{batch_time:.2f}",
            data_s=f"{data_time:.2f}",
            infer_s=f"{infer_time:.2f}",
            avg_s=f"{avg_batch_time:.2f}",
        )
        if (
            LOG_EVAL_BATCH_TIMES
            and (
                (batch_idx + 1) % EVAL_BATCH_TIME_LOG_INTERVAL == 0
                or batch_idx == 0
                or batch_idx + 1 == len(val_loader)
            )
        ):
            tqdm.write(
                f"  [val] batch {batch_idx + 1}/{len(val_loader)}: "
                f"data={data_time:.2f}s infer={infer_time:.2f}s total={batch_time:.2f}s "
                f"avg={avg_batch_time:.2f}s"
            )

    avg_loss = val_loss / len(val_loader)
    writer.add_scalar("val/loss", avg_loss, epoch)

    miou, per_class_iou = compute_iou_from_confusion(confusion, NUM_CLASSES)

    writer.add_scalar("val/mIoU", miou, epoch)
    for cls_id, iou in per_class_iou.items():
        writer.add_scalar(f"val/IoU_{CLASS_NAMES[cls_id]}", iou, epoch)

    iou_str = " | ".join(
        f"{name}: {iou:.3f}"
        for name, (_, iou) in zip(CLASS_NAMES, per_class_iou.items())
    )
    if batch_times:
        print(
            "  Validation timing: "
            f"avg_total={sum(batch_times) / len(batch_times):.2f}s "
            f"avg_data={sum(data_times) / len(data_times):.2f}s "
            f"avg_infer={sum(infer_times) / len(infer_times):.2f}s"
        )
    print(f"  Per-class IoU: {iou_str}")

    return miou


def _viz_val_samples(
    model: torch.nn.Module,
    val_dataset,
    viz_dir: Path,
    epoch: int,
    device: torch.device,
) -> None:
    """Generate visualization from random validation samples."""
    model.eval()
    indices = random.sample(
        range(len(val_dataset)), min(VIZ_NUM_SAMPLES, len(val_dataset))
    )

    images_list = []
    masks_true_list = []
    masks_pred_list = []

    for idx in indices:
        sample = val_dataset[idx]
        image = sample["pixel_values"].unsqueeze(0).to(device)
        mask_true = sample["labels"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image)

        images_list.append(image.squeeze(0).cpu())
        masks_true_list.append(mask_true.squeeze(0).cpu())
        masks_pred_list.append(logits.squeeze(0).cpu())

    visualize_predictions(
        torch.stack(images_list),
        torch.stack(masks_true_list),
        torch.stack(masks_pred_list),
        CLASS_NAMES,
        viz_dir,
        epoch,
    )


# ============================================================================
# Entry
# ============================================================================

if __name__ == "__main__":
    train()
