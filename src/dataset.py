"""
PyTorch Dataset for semantic segmentation tiles.
Supports on-the-fly Albumentations augmentation (train) and clean loading (val).
"""
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import IMAGE_SIZE


# ============================================================================
# Augmentation pipelines
# ============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """
    Aggressive augmentation for small-dataset training.

    Spatial transforms (applied to image+mask):
      - RandomResizedCrop: scale 0.5-1.0 → effective 5-10× dataset expansion
      - HorizontalFlip: scene symmetry
      - Rotate(±30°): orientation invariance
      - ElasticTransform: deformation robustness

    Pixel transforms (image only):
      - RandomBrightnessContrast: lighting variation
      - HueSaturationValue: color variation
      - GaussNoise: sensor noise robustness

    Normalization: ImageNet mean/std (SegFormer & EfficientNet both use it).
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.5, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=0, p=0.7),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=20, p=0.4
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0
            ),
        ]
    )


def get_val_transforms() -> A.Compose:
    """Validation: no augmentation, only ImageNet normalization."""
    return A.Compose(
        [
            A.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0
            ),
        ]
    )


# ============================================================================
# Dataset
# ============================================================================


class SegmentationDataset(Dataset):
    """
    Loads tile images and masks from preprocessed data/ directory.

    Args:
        img_dir: Directory containing tile images (*.png).
        mask_dir: Directory containing tile masks (*.png, single-channel).
        transforms: Albumentations Compose (None = raw loading).
    """

    def __init__(
        self,
        img_dir: Path,
        mask_dir: Path,
        transforms: Optional[A.Compose] = None,
    ):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms

        self.file_names = sorted(
            [p.name for p in self.img_dir.glob("*.png")]
        )

        if not self.file_names:
            raise RuntimeError(f"No PNG files found in {self.img_dir}")

        # Verify masks exist
        missing_masks = [
            f for f in self.file_names if not (self.mask_dir / f).exists()
        ]
        if missing_masks:
            raise RuntimeError(
                f"Missing masks for {len(missing_masks)} tiles: {missing_masks[:5]}..."
            )

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fname = self.file_names[idx]

        # Load image (H, W, 3) RGB
        image = np.array(Image.open(self.img_dir / fname).convert("RGB"))

        # Load mask (H, W) single-channel, convert to int64 class indices
        mask = np.array(Image.open(self.mask_dir / fname), dtype=np.int64)

        # Apply transforms (Albumentations expects contiguous uint8 arrays)
        # A.Normalize outputs float32 in [0,1] with mean/std applied
        if self.transforms is not None:
            augmented = self.transforms(image=np.ascontiguousarray(image), mask=mask)
            image = augmented["image"]  # float32 (H, W, 3), normalized
            mask = augmented["mask"]     # uint8 (H, W)
            # Channel-first for PyTorch
            image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            # Fallback: no transforms, do manual [0,1] scaling
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mask = torch.from_numpy(mask).long()

        return {"pixel_values": image, "labels": mask}


# ============================================================================
# Factory functions
# ============================================================================


def create_train_dataset(
    img_dir: Path, mask_dir: Path
) -> SegmentationDataset:
    """Create training dataset with full augmentation."""
    return SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        transforms=get_train_transforms(),
    )


def create_val_dataset(
    img_dir: Path, mask_dir: Path
) -> SegmentationDataset:
    """Create validation dataset without augmentation."""
    return SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        transforms=get_val_transforms(),
    )
