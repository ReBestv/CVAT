"""
Preprocessing: RGB mask → class index, tiling, train/val split.

Run once before training:
    python src/preprocess.py

Input:  南原图/*.JPG + 南新/SegmentationClass/*.png
Output: data/{train,val}/{images,masks}/*.png (512×512 tiles)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    COLOR_TO_ID,
    IGNORE_INDEX,
    IMAGE_SIZE,
    IMAGESET_PATH,
    ORIGINAL_IMG_DIR,
    RANDOM_SEED,
    SEGCLASS_DIR,
    TILE_STRIDE,
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    TRAIN_SPLIT,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
)


def parse_imageset(imageset_path: Path) -> list[str]:
    """Parse ImageSets file, return list of base names (without extension)."""
    names = []
    with open(imageset_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


def find_image_path(base_name: str) -> Path | None:
    """Find the actual image file given base name. Try .JPG then .jpg."""
    for ext in (".JPG", ".jpg"):
        p = ORIGINAL_IMG_DIR / f"{base_name}{ext}"
        if p.exists():
            return p
    return None


def find_mask_path(base_name: str) -> Path | None:
    """Find the corresponding SegmentationClass mask file."""
    p = SEGCLASS_DIR / f"{base_name}.png"
    return p if p.exists() else None


def rgb_mask_to_class_ids(
    rgb_mask: np.ndarray, color_to_id: dict[tuple[int, ...], int]
) -> np.ndarray:
    """
    Convert H×W×3 RGB mask to H×W uint8 class index mask.

    Args:
        rgb_mask: (H, W, 3) numpy array, dtype=uint8
        color_to_id: mapping from (r,g,b) tuple to class index

    Returns:
        (H, W) numpy array, dtype=uint8
    """
    class_mask = np.full(rgb_mask.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    for rgb, cls_id in color_to_id.items():
        class_mask[np.all(rgb_mask == np.array(rgb), axis=2)] = cls_id
    return class_mask


def tile_positions(h: int, w: int, size: int, stride: int) -> list[tuple[int, int]]:
    """
    Compute (y, x) top-left positions for overlapping tiles.

    Covers the entire image including right/bottom edges.
    """
    y_positions = list(range(0, h - size + 1, stride))
    x_positions = list(range(0, w - size + 1, stride))

    # Edge fill: add last valid position at bottom/right edge
    if h > size and (h - size) not in y_positions:
        y_positions.append(h - size)
    if w > size and (w - size) not in x_positions:
        x_positions.append(w - size)

    # If image smaller than tile size, just one tile from (0,0)
    if not y_positions:
        y_positions = [0]
    if not x_positions:
        x_positions = [0]

    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    return [(y, x) for y in y_positions for x in x_positions]


def save_tile(img: np.ndarray, out_path: Path) -> None:
    """Save a tile as PNG."""
    Image.fromarray(img).save(out_path)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Discover valid image-mask pairs
    # ------------------------------------------------------------------
    all_names = parse_imageset(IMAGESET_PATH)
    print(f"[1/4] ImageSet entries: {len(all_names)}")

    valid_pairs: list[tuple[str, Path, Path]] = []
    missing: list[str] = []

    for name in all_names:
        img_path = find_image_path(name)
        mask_path = find_mask_path(name)
        if img_path and mask_path:
            valid_pairs.append((name, img_path, mask_path))
        else:
            missing.append(name)

    if missing:
        print(f"  WARNING: {len(missing)} images missing — skipped: {missing}")
    print(f"  Valid image-mask pairs: {len(valid_pairs)}")

    if not valid_pairs:
        print("ERROR: No valid image-mask pairs found. Aborting.")
        return

    # ------------------------------------------------------------------
    # 2. Train/val split at original image level
    # ------------------------------------------------------------------
    random.seed(RANDOM_SEED)
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * TRAIN_SPLIT)

    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    print(f"[2/4] Split: {len(train_pairs)} train / {len(val_pairs)} val")

    # ------------------------------------------------------------------
    # 3. Process: RGB→class index → tile → save
    # ------------------------------------------------------------------
    for d in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    color_to_id = COLOR_TO_ID
    total_train = 0
    total_val = 0

    print("[3/4] Processing tiles...")

    for split_name, pairs, img_dir, mask_dir in [
        ("train", train_pairs, TRAIN_IMG_DIR, TRAIN_MASK_DIR),
        ("val", val_pairs, VAL_IMG_DIR, VAL_MASK_DIR),
    ]:
        for name, img_path, mask_path in tqdm(
            pairs, desc=f"  {split_name}"
        ):
            # Load
            image = np.array(Image.open(img_path).convert("RGB"))
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

            if image.shape[:2] != mask_rgb.shape[:2]:
                print(
                    f"  WARNING: shape mismatch for {name} "
                    f"(image {image.shape[:2]} vs mask {mask_rgb.shape[:2]}), skipping"
                )
                continue

            # Convert RGB mask to class indices
            mask_idx = rgb_mask_to_class_ids(mask_rgb, color_to_id)

            # Tile
            positions = tile_positions(
                image.shape[0], image.shape[1], IMAGE_SIZE, TILE_STRIDE
            )

            for y, x in positions:
                tile_img = image[y : y + IMAGE_SIZE, x : x + IMAGE_SIZE]
                tile_mask = mask_idx[y : y + IMAGE_SIZE, x : x + IMAGE_SIZE]

                tile_name = f"{name}_y{y}_x{x}.png"
                save_tile(tile_img, img_dir / tile_name)
                # mode="L" for single-channel grayscale
                Image.fromarray(tile_mask, mode="L").save(mask_dir / tile_name)

                if split_name == "train":
                    total_train += 1
                else:
                    total_val += 1

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"[4/4] Done!")
    print(f"  Train tiles: {total_train} ({len(train_pairs)} source images)")
    print(f"  Val tiles:   {total_val} ({len(val_pairs)} source images)")
    print(f"  Total:       {total_train + total_val}")
    print(f"\n  Output dirs:")
    print(f"    {TRAIN_IMG_DIR}")
    print(f"    {TRAIN_MASK_DIR}")
    print(f"    {VAL_IMG_DIR}")
    print(f"    {VAL_MASK_DIR}")


if __name__ == "__main__":
    main()
