from __future__ import annotations

"""
Shared evaluation script for SegFormer, U-Net, and DeepLabV3+.
在 config.py 修改 EVAL_MODEL_TYPE 和 EVAL_CHECKPOINT。
直接运行: python src/evaluate.py
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EVAL_CHECKPOINT,
    EVAL_BATCH_TIME_LOG_INTERVAL,
    EVAL_MODEL_TYPE,
    LOG_EVAL_BATCH_TIMES,
    NUM_CLASSES,
    NUM_WORKERS,
    DEEPLAB_ENCODER,
    DEEPLAB_ENCODER_WEIGHTS,
    SEGFORMER_MODEL_NAME,
    UNET_ENCODER,
    UNET_ENCODER_WEIGHTS,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    VIZ_NUM_SAMPLES,
    ID_TO_COLOR,
    IGNORE_INDEX,
    auto_find_checkpoint,
    get_device,
)
from dataset import create_val_dataset
from train_unet import update_confusion_matrix_


# ============================================================================
# Metrics
# ============================================================================


def compute_metrics(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> dict:
    """
    Compute comprehensive metrics: mIoU, per-class IoU, pixel accuracy, Dice.

    Args:
        pred: (B, C, H, W) logits
        target: (B, H, W) class indices
        num_classes: number of classes

    Returns:
        dict with miou, per_class_iou, pixel_accuracy, per_class_dice, confusion_matrix
    """
    pred_cls = pred.argmax(dim=1).to(torch.int64)
    target = target.to(torch.int64)
    mask = (target != IGNORE_INDEX) & (target >= 0) & (target < num_classes)

    bins = target[mask] * num_classes + pred_cls[mask]
    confusion = torch.bincount(
        bins, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).cpu().numpy()

    # Per-class IoU
    ious = {}
    dices = {}
    for c in range(num_classes):
        intersection = confusion[c, c]
        union = confusion[c, :].sum() + confusion[:, c].sum() - intersection
        ious[c] = float(intersection / max(union, 1))

        # Dice = 2*TP / (2*TP + FP + FN)
        dice_denom = confusion[c, :].sum() + confusion[:, c].sum()
        dices[c] = float(2 * intersection / max(dice_denom, 1))

    # Pixel accuracy
    correct = confusion.diagonal().sum()
    total = confusion.sum()
    pixel_acc = float(correct / max(total, 1))

    # Mean metrics
    miou = float(np.mean(list(ious.values())))
    mdice = float(np.mean(list(dices.values())))

    return {
        "miou": miou,
        "mdice": mdice,
        "pixel_accuracy": pixel_acc,
        "per_class_iou": ious,
        "per_class_dice": dices,
        "confusion_matrix": confusion.tolist(),
    }


def compute_metrics_from_confusion(confusion: torch.Tensor | np.ndarray) -> dict:
    """Compute evaluation metrics from a pre-accumulated confusion matrix."""
    if isinstance(confusion, torch.Tensor):
        confusion = confusion.detach().cpu().numpy()

    ious = {}
    dices = {}
    for c in range(NUM_CLASSES):
        intersection = confusion[c, c]
        union = confusion[c, :].sum() + confusion[:, c].sum() - intersection
        ious[c] = float(intersection / max(union, 1))

        dice_denom = confusion[c, :].sum() + confusion[:, c].sum()
        dices[c] = float(2 * intersection / max(dice_denom, 1))

    correct = confusion.diagonal().sum()
    total = confusion.sum()
    pixel_acc = float(correct / max(total, 1))

    return {
        "miou": float(np.mean(list(ious.values()))),
        "mdice": float(np.mean(list(dices.values()))),
        "pixel_accuracy": pixel_acc,
        "per_class_iou": ious,
        "per_class_dice": dices,
        "confusion_matrix": confusion.tolist(),
    }


def forward_for_eval(
    model: torch.nn.Module,
    model_type: str,
    images: torch.Tensor,
    mask_shape: tuple[int, int],
) -> torch.Tensor:
    """Run one forward pass and resize logits when required."""
    if model_type == "segformer":
        outputs = model(pixel_values=images)
        logits = outputs.logits
        return torch.nn.functional.interpolate(
            logits,
            size=mask_shape,
            mode="bilinear",
            align_corners=False,
        )
    return model(images)


# ============================================================================
# Visualization
# ============================================================================


def generate_visualizations(
    model: torch.nn.Module,
    val_dataset,
    model_type: str,
    output_dir: Path,
    device: str,
    num_samples: int = VIZ_NUM_SAMPLES,
) -> None:
    """Generate and save prediction visualizations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    model.eval()

    import random
    random.seed(42)
    indices = random.sample(
        range(len(val_dataset)), min(num_samples, len(val_dataset))
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    cmap = {i: (c[0] / 255, c[1] / 255, c[2] / 255) for i, c in ID_TO_COLOR.items()}

    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))
    if len(indices) == 1:
        axes = np.array([axes])

    for row, idx in enumerate(indices):
        sample = val_dataset[idx]
        image_t = sample["pixel_values"].unsqueeze(0).to(device)
        mask_true = sample["labels"].numpy()

        with torch.no_grad():
            logits = forward_for_eval(
                model,
                model_type,
                image_t,
                (sample["labels"].shape[0], sample["labels"].shape[1]),
            )

        pred_cls = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Denormalize image
        img = image_t.squeeze(0).cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # Colorize masks
        true_rgb = np.full((*mask_true.shape, 3), 0.75, dtype=np.float32)
        pred_rgb = np.zeros((*pred_cls.shape, 3), dtype=np.float32)
        for c, color in cmap.items():
            true_rgb[mask_true == c] = color
            pred_rgb[pred_cls == c] = color

        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Original")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(true_rgb)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(pred_rgb)
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis("off")

    legend_patches = [
        Patch(color=cmap[c], label=f"{c}: {name}")
        for c, name in enumerate(CLASS_NAMES)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=min(6, NUM_CLASSES),
        fontsize=7,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    save_path = output_dir / "predictions.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualizations saved: {save_path}")


# ============================================================================
# Model Loading
# ============================================================================


def load_segformer_model(checkpoint_path: Path, device: str):
    """Load SegFormer model from checkpoint."""
    from transformers import SegformerForSemanticSegmentation

    print(f"Loading SegFormer: {SEGFORMER_MODEL_NAME}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Checkpoint epoch: {state.get('epoch', 'unknown')}")
    print(f"  Checkpoint mIoU:  {state.get('miou', 'unknown'):.4f}")
    return model


def load_unet_model(checkpoint_path: Path, device: str):
    """Load U-Net model from SMP checkpoint."""
    import segmentation_models_pytorch as smp

    print(f"Loading U-Net: {UNET_ENCODER} (weights={UNET_ENCODER_WEIGHTS})")
    model = smp.Unet(
        encoder_name=UNET_ENCODER,
        encoder_weights=None,  # Load from checkpoint, not pretrained
        in_channels=3,
        classes=NUM_CLASSES,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Checkpoint epoch: {state.get('epoch', 'unknown')}")
    print(f"  Checkpoint mIoU:  {state.get('miou', 'unknown'):.4f}")
    return model


def load_deeplab_model(checkpoint_path: Path, device: str):
    """Load DeepLabV3+ model from SMP checkpoint."""
    import segmentation_models_pytorch as smp

    print(f"Loading DeepLabV3+: {DEEPLAB_ENCODER} (weights={DEEPLAB_ENCODER_WEIGHTS})")
    model = smp.DeepLabV3Plus(
        encoder_name=DEEPLAB_ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Checkpoint epoch: {state.get('epoch', 'unknown')}")
    print(f"  Checkpoint mIoU:  {state.get('miou', 'unknown'):.4f}")
    return model


# ============================================================================
# Main
# ============================================================================


def evaluate(
    model_type: str,
    checkpoint_path: Path,
    output_dir: Path,
    device: str = "cpu",
) -> None:
    """Run full evaluation and save results."""

    assert model_type in ("segformer", "unet", "deeplab"), \
        f"model_type must be 'segformer', 'unet', or 'deeplab', got '{model_type}'"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if model_type == "segformer":
        model = load_segformer_model(checkpoint_path, device)
    elif model_type == "unet":
        model = load_unet_model(checkpoint_path, device)
    else:
        model = load_deeplab_model(checkpoint_path, device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    val_dataset = create_val_dataset(VAL_IMG_DIR, VAL_MASK_DIR)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    print(f"Validation tiles: {len(val_dataset)}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("\nRunning inference...")
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
    batch_times = []
    data_times = []
    infer_times = []

    with torch.no_grad():
        pbar = tqdm(range(len(val_loader)), desc="  Evaluating")
        val_iter = iter(val_loader)
        for batch_idx in pbar:
            fetch_start = time.perf_counter()
            batch = next(val_iter)
            data_time = time.perf_counter() - fetch_start

            infer_start = time.perf_counter()
            images = batch["pixel_values"].to(device, non_blocking=True)
            masks = batch["labels"].to(device, non_blocking=True)
            logits = forward_for_eval(model, model_type, images, masks.shape[-2:])
            pred = logits.argmax(dim=1)
            update_confusion_matrix_(confusion, pred, masks, NUM_CLASSES)
            infer_time = time.perf_counter() - infer_start
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
                    f"  [eval] batch {batch_idx + 1}/{len(val_loader)}: "
                    f"data={data_time:.2f}s infer={infer_time:.2f}s total={batch_time:.2f}s "
                    f"avg={avg_batch_time:.2f}s"
                )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("\nComputing metrics...")
    metrics = compute_metrics_from_confusion(confusion)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — {model_type.upper()}")
    print(f"{'='*60}")
    print(f"  mIoU:           {metrics['miou']:.4f}")
    print(f"  mDice:          {metrics['mdice']:.4f}")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"\n  Per-class IoU & Dice:")
    print(f"  {'Class':<25s} {'IoU':>8s}  {'Dice':>8s}")
    print(f"  {'-'*25} {'-'*8}  {'-'*8}")
    for cls_id, name in enumerate(CLASS_NAMES):
        iou = metrics["per_class_iou"][cls_id]
        dice = metrics["per_class_dice"][cls_id]
        print(f"  {name:<25s} {iou:8.4f}  {dice:8.4f}")
    if batch_times:
        print(
            "\n  Evaluation timing:"
            f" avg_total={sum(batch_times) / len(batch_times):.2f}s"
            f" avg_data={sum(data_times) / len(data_times):.2f}s"
            f" avg_infer={sum(infer_times) / len(infer_times):.2f}s"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    # JSON report
    report = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
    }
    report_path = output_dir / "metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved: {report_path}")

    # Visualization
    print("\nGenerating visualizations...")
    generate_visualizations(model, val_dataset, model_type, output_dir, device)

    print(f"\nAll results saved to: {output_dir}")


# ============================================================================
# Entry
# ============================================================================

if __name__ == "__main__":
    model_type = EVAL_MODEL_TYPE
    device = get_device()

    # 自动找 checkpoint 或用手工指定的
    if EVAL_CHECKPOINT is not None:
        checkpoint_path = Path(EVAL_CHECKPOINT).resolve()
    else:
        found = auto_find_checkpoint(model_type)
        if found is None:
            print(f"ERROR: 没有找到 {model_type} 的 checkpoint")
            print(f"       请先训练，或在 config.py 设置 EVAL_CHECKPOINT 手工指定路径")
            exit(1)
        checkpoint_path = found
        print(f"自动找到 checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        exit(1)

    output_dir = f"eval_{checkpoint_path.parent.name}"

    evaluate(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        output_dir=Path(output_dir),
        device=device,
    )
