"""
Plan C: DeepLabV3+ + MobileNetV3 fine-tuning via segmentation-models-pytorch.
All parameters live in config.py. Run: python src/train_deeplab.py
"""
from datetime import datetime
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    DEEPLAB_CHECKPOINT_DIR,
    DEEPLAB_ENCODER,
    DEEPLAB_ENCODER_WEIGHTS,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    EVAL_EVERY_N_EPOCHS,
    IGNORE_INDEX,
    LEARNING_RATE,
    LOSS_ALPHA,
    NUM_CLASSES,
    NUM_WORKERS,
    PHASE1_EPOCHS,
    PHASE1_LR,
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    VIZ_EVERY_N_EPOCHS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    get_device,
)
from dataset import create_train_dataset, create_val_dataset
from losses import ComboLoss
from train_unet import (
    _viz_val_samples,
    cleanup_old_checkpoints,
    get_linear_warmup_cosine_scheduler,
    save_checkpoint,
    set_encoder_grad,
    validate,
)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: ComboLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    writer: SummaryWriter,
    epoch: int,
    device: str,
) -> None:
    """Run one training epoch and log losses."""
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

    n_batches = len(train_loader)
    writer.add_scalar("train/loss", train_loss / n_batches, epoch)
    writer.add_scalar("train/ce_loss", train_ce / n_batches, epoch)
    writer.add_scalar("train/dice_loss", train_dice / n_batches, epoch)
    writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)


def train() -> None:
    device = get_device()

    encoder_slug = DEEPLAB_ENCODER.replace("-", "").replace("_", "")
    run_name = f"deeplab_{encoder_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = DEEPLAB_CHECKPOINT_DIR / run_name
    log_dir = Path("logs") / run_name
    viz_dir = checkpoint_dir / "viz"

    for d in [checkpoint_dir, log_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"Training: {run_name}")
    print(f"Device: {device}")

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

    print(
        f"Loading model: DeepLabV3+ + {DEEPLAB_ENCODER} "
        f"(weights={DEEPLAB_ENCODER_WEIGHTS})"
    )
    model = smp.DeepLabV3Plus(
        encoder_name=DEEPLAB_ENCODER,
        encoder_weights=DEEPLAB_ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    criterion = ComboLoss(
        alpha=LOSS_ALPHA, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX
    )

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
        train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, writer, epoch, device
        )

        if epoch % EVAL_EVERY_N_EPOCHS == 0 or epoch == PHASE1_EPOCHS:
            miou = validate(model, val_loader, criterion, writer, epoch, device)
            print(f"  Val mIoU: {miou:.4f}")

            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, miou, checkpoint_dir, is_best=True)
                print(f"  New best mIoU: {best_miou:.4f}")

            save_checkpoint(model, optimizer, epoch, miou, checkpoint_dir)
            cleanup_old_checkpoints(checkpoint_dir)

            if epoch % VIZ_EVERY_N_EPOCHS == 0:
                _viz_val_samples(model, val_dataset, viz_dir, epoch, device)

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
        train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, writer, epoch, device
        )

        if epoch % EVAL_EVERY_N_EPOCHS == 0 or epoch == EPOCHS:
            miou = validate(model, val_loader, criterion, writer, epoch, device)
            print(f"  Val mIoU: {miou:.4f} (best: {best_miou:.4f} @ epoch {best_epoch})")

            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, miou, checkpoint_dir, is_best=True)
                print(f"  New best mIoU: {best_miou:.4f}")
            else:
                patience_counter += 1

            save_checkpoint(model, optimizer, epoch, miou, checkpoint_dir)
            cleanup_old_checkpoints(checkpoint_dir)

            if epoch % VIZ_EVERY_N_EPOCHS == 0:
                _viz_val_samples(model, val_dataset, viz_dir, epoch, device)

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {EARLY_STOP_PATIENCE} evals)"
            )
            break

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {checkpoint_dir / 'best.pt'}")
    writer.close()


if __name__ == "__main__":
    train()
