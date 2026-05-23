"""
Central configuration for CVAT AOI semantic segmentation pipeline.
All hyperparameters, paths, and model settings in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

ROOT = Path(r"E:\CVAT")
ORIGINAL_IMG_DIR = ROOT / "南原图"
ANNOTATION_DIR = ROOT / "南新"
LABELMAP_PATH = ANNOTATION_DIR / "labelmap.txt"
SEGCLASS_DIR = ANNOTATION_DIR / "SegmentationClass"
IMAGESET_PATH = ANNOTATION_DIR / "ImageSets" / "Segmentation" / "default.txt"

DATA_DIR = ROOT / "data"
TRAIN_IMG_DIR = DATA_DIR / "train" / "images"
TRAIN_MASK_DIR = DATA_DIR / "train" / "masks"
VAL_IMG_DIR = DATA_DIR / "val" / "images"
VAL_MASK_DIR = DATA_DIR / "val" / "masks"

CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"

# ============================================================================
# Preprocessing
# ============================================================================

IMAGE_SIZE = 512              # tile size (square)
TILE_STRIDE = 384             # stride (overlap = IMAGE_SIZE - TILE_STRIDE = 128)
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# ============================================================================
# Dataset
# ============================================================================

IGNORE_INDEX = 255
RAW_BACKGROUND_ID = 0
NUM_CLASSES = 9

CLASS_NAMES = [
    "crowd", "public_facilities", "commercial", "ground_paving",
    "sky", "building_interface", "cultural_decoration",
    "green_landscape", "vehicles",
]

# Training RGB encoding from labelmap.txt.
# CVAT background / unannotated pixels are ignored during training and metrics.
# Foreground CVAT ids 1-9 are remapped to training ids 0-8.
COLOR_TO_ID = {
    (0, 0, 0): IGNORE_INDEX,
    (12, 54, 138): 0,
    (131, 214, 41): 1,
    (180, 40, 5): 2,
    (103, 22, 84): 3,
    (97, 196, 248): 4,
    (233, 161, 3): 5,
    (60, 5, 5): 6,
    (12, 170, 34): 7,
    (78, 78, 78): 8,
}

ID_TO_COLOR = {
    cls_id: color for color, cls_id in COLOR_TO_ID.items() if cls_id != IGNORE_INDEX
}

# ============================================================================
# Training (shared)
# ============================================================================

BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 200
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
LR_SCHEDULER = "cosine"
DROPOUT = 0.1
LABEL_SMOOTHING = 0.0

# Combo Loss
LOSS_ALPHA = 0.5           # Dice weight in combo loss (0.5 = equal weight)

# Two-Phase Training
PHASE1_EPOCHS = 15
PHASE1_LR = 1e-4

# Checkpointing
SAVE_TOP_K = 3
EARLY_STOP_PATIENCE = 10

# Evaluation frequency
EVAL_EVERY_N_EPOCHS = 5
VIZ_EVERY_N_EPOCHS = 20
VIZ_NUM_SAMPLES = 4
LOG_EVAL_BATCH_TIMES = True
EVAL_BATCH_TIME_LOG_INTERVAL = 1

# ============================================================================
# Plan A: SegFormer (HuggingFace Transformers)
# ============================================================================

SEGFORMER_MODEL_NAME = "E:/CVAT/models/segformer-b1"
SEGFORMER_IMAGE_SIZE = 512
SEGFORMER_CHECKPOINT_DIR = CHECKPOINT_DIR / "segformer"

# ============================================================================
# Plan B: U-Net (segmentation-models-pytorch)
# ============================================================================

UNET_ENCODER = "efficientnet-b0"
UNET_ENCODER_WEIGHTS = "imagenet"
UNET_ENCODER_LOCAL_WEIGHTS = ROOT / "models" / "efficientnet-b0" / "efficientnet-b0-355c32eb.pth"
UNET_IMAGE_SIZE = 512
UNET_CHECKPOINT_DIR = CHECKPOINT_DIR / "unet"

# ============================================================================
# Plan C: DeepLabV3+ (segmentation-models-pytorch)
# ============================================================================

DEEPLAB_ENCODER = "timm-mobilenetv3_large_100"
DEEPLAB_ENCODER_WEIGHTS = "imagenet"
DEEPLAB_IMAGE_SIZE = 512
DEEPLAB_CHECKPOINT_DIR = CHECKPOINT_DIR / "deeplab"

# ============================================================================
# Runtime
# ============================================================================


def get_device() -> str:
    """Lazy device detection (avoids importing torch in preprocessing)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# Module-level default (train scripts will call get_device() directly)
DEVICE = "cpu"  # Placeholder; training scripts use get_device()


# ============================================================================
# Evaluation — 改这里切换评估的模型和checkpoint
# ============================================================================

# 要评估哪个模型: "segformer" 或 "unet"
EVAL_MODEL_TYPE = "segformer"

# checkpoint 路径（不填则自动找最新一个 best.pt）
# 手工指定示例: EVAL_CHECKPOINT = CHECKPOINT_DIR / "segformer" / "20260522_143000" / "best.pt"
EVAL_CHECKPOINT = None  # None = 自动扫描


def auto_find_checkpoint(model_type: str) -> Path | None:
    """自动找到模型目录下最新的 best.pt."""
    base = CHECKPOINT_DIR / model_type
    if not base.exists():
        return None
    runs = [d for d in base.iterdir() if d.is_dir()]
    bests = []
    for run in runs:
        pt = run / "best.pt"
        if pt.exists():
            bests.append((pt.stat().st_mtime, pt))
    if not bests:
        return None
    bests.sort(reverse=True)
    return bests[0][1]
