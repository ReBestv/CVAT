# Spec: CVAT AOI 语义分割微调 Pipeline

**Created:** 2026-05-21  
**Status:** Draft — awaiting review  
**Author:** Sisyphus (brainstorming session)

---

## Overview

基于 CVAT 标注的 AOI（Area of Interest）场景图像，使用预训练语义分割模型进行微调，实现 11 类（10 前景 + background）像素级分类。

### 约束

- 当前数据：39 张（目标 120 张），6000×4000 RGB
- 硬件：本地台式 GPU 8-12GB
- 阶段：实验验证（非生产部署）

---

## 1. Data Summary

### 1.1 Directory Structure (PASCAL VOC Format)

```
E:\CVAT\
├── 南原图/                    # 39 JPG originals (6000×4000 RGB)
├── 南新/                      # Annotations
│   ├── labelmap.txt           # Class definitions: name:r,g,b
│   ├── SegmentationClass/     # 39 RGB PNG masks (color-coded)
│   ├── SegmentationObject/    # 39 RGB PNG masks (instance-level)
│   └── ImageSets/Segmentation/
│       └── default.txt        # 42 entries (3 missing: 5379, 5380, 5386)
```

### 1.2 Classes (11 total)

| ID | Name (CN) | Name (EN) | RGB | Prevalence (sample) |
|----|-----------|-----------|-----|---------------------|
| 0 | 背景 | background | (0,0,0) | 0.8% |
| 1 | 人群 | crowd/people | (12,54,138) | 4.0% |
| 2 | 公共设施 | public facilities | (131,214,41) | 2.0% |
| 3 | 商业 | commercial | (180,40,5) | 23.3% |
| 4 | 地面铺装 | ground paving | (103,22,84) | 10.5% |
| 5 | 天空 | sky | (97,196,248) | 25.3% |
| 6 | 建筑界面 | building interface | (233,161,3) | 20.9% |
| 7 | 文化装饰 | cultural decoration | (60,5,5) | 1.5% |
| 8 | 绿化景观 | green landscape | (12,170,34) | 10.7% |
| 9 | 车 | vehicles | (78,78,78) | 1.1% |

**Key concern**: Classes 0 (0.8%), 7 (1.5%), 9 (1.1%) are severely underrepresented. Combo Loss (Dice + CE) is chosen specifically to mitigate this.

### 1.3 Critical Preprocessing

SegmentationClass masks are **RGB color-coded** (3-channel), NOT single-channel index maps. Must convert via lookup table before training:

```python
color_to_id = {
    (0,0,0): 0, (12,54,138): 1, (131,214,41): 2,
    (180,40,5): 3, (103,22,84): 4, (97,196,248): 5,
    (233,161,3): 6, (60,5,5): 7, (12,170,34): 8, (78,78,78): 9,
}
```

---

## 2. Model Selection: SegFormer-B1

### 2.1 Why SegFormer-B1

| Factor | Decision |
|--------|----------|
| GPU (8-12GB) | B1 uses ~4GB @ 512², leaves room for batch=8–16 |
| Pre-training | ADE20K/Cityscapes — scene parsing domain matches urban AOI |
| Architecture | Hierarchical Transformer (MiT) captures global context critical for sky/building/vegetation classes |
| Overfitting risk | MLP decoder is lightweight (fewer params to overfit on 40 imgs) |
| Ecosystem | HuggingFace `transformers` with official fine-tune cookbook |

### 2.2 Alternatives

| Scenario | Switch to |
|----------|-----------|
| Overfitting observed | SegFormer-B0 (3.8M, even lighter decoder) |
| Need faster iteration | DeepLabV3+ MobileNet-V3 (11M, torchvision native) |
| 120+ images later | SegFormer-B2 (24.7M, higher accuracy ceiling) |

### 2.3 What We're NOT Using

- **SAM**: Prompt-based design, 94M params requiring LoRA adapters. Wrong paradigm for semantic segmentation.
- **Mask2Former**: 47M params (tiny). Mask classification + masked attention too complex for 40-image dataset.
- **U-Net**: Viable, but Transformer global context edges it out for urban scene parsing.

---

## 3. Pipeline Architecture

```
RAW DATA              PREPROCESSING              TRAINING                EVALUATION
─────────             ─────────────              ────────                ──────────
南原图/*.JPG    ──┐                    ┌── Dataset class      ┌── mIoU
                  ├── preprocess.py ──┤  (Albumentations) ──┤── per-class IoU
南新/SegClass    ──┘                    └── DataLoader ───────┘── confusion matrix
  /*.png (RGB)         │                       │                    │
                       ▼                       ▼                    ▼
                  RGB→class idx          SegFormer-B1         TensorBoard
                  80/20 split            HF Trainer           sample pred viz
                  512² tiles             Combo Loss            best checkpoint
                  saved to data/          200 epochs            early stopping
```

---

## 4. Project Structure

```
E:\CVAT\
├── 南原图/                    # Original JPGs (read-only, never modified)
├── 南新/                      # Original annotations (read-only)
├── data/                      # Preprocessed training data (generated)
│   ├── train/
│   │   ├── images/            # 512×512 RGB tiles
│   │   └── masks/             # 512×512 grayscale class-index PNGs
│   └── val/
│       ├── images/
│       └── masks/
├── src/
│   ├── config.py              # All hyperparameters in one place
│   ├── preprocess.py          # RGB mask → class index + tiling + split
│   ├── dataset.py             # PyTorch Dataset with Albumentations
│   ├── train.py               # Training script (HF Trainer or custom loop)
│   └── evaluate.py            # Inference + metrics on test set
├── checkpoints/               # Model .pth / HF model dir
├── logs/                      # TensorBoard logs
└── docs/superpowers/specs/    # This spec + implementation plan
```

### 4.1 Module Boundaries

| Module | Input | Output | Dependencies |
|--------|-------|--------|-------------|
| `preprocess.py` | 南原图/*.JPG, 南新/SegClass/*.png, labelmap.txt | data/{train,val}/{images,masks}/*.png | PIL, numpy, albumentations (resize only) |
| `dataset.py` | data/{train,val}/ dirs | (image_tensor, mask_tensor) batches | albumentations, torch |
| `train.py` | config.py, dataset.py | checkpoints/, logs/ | transformers, torch |
| `evaluate.py` | checkpoint, val split | mIoU, per-class IoU, viz | torch, sklearn |

---

## 5. Training Configuration

### 5.1 Hyperparameters

```python
CONFIG = {
    "model_name": "nvidia/segformer-b1-finetuned-ade-512-512",
    "num_classes": 11,
    "image_size": 512,

    "batch_size": 8,
    "num_workers": 4,
    "train_split": 0.8,

    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "epochs": 200,
    "warmup_steps": 50,
    "lr_scheduler": "cosine",

    "dropout": 0.1,
    "label_smoothing": 0.0,

    "loss_alpha": 0.5,       # Dice weight in combo loss
    "save_top_k": 3,
    "early_stop_patience": 30,
}
```

### 5.2 Two-Phase Training

```
Phase 1 (epochs 1–15):
  - Freeze encoder (MiT-B1 backbone)
  - Train decoder head only
  - LR: 1e-4 (higher since decoder starts random)
  - Purpose: stabilize decoder before joint training

Phase 2 (epochs 16–200):
  - Unfreeze all parameters
  - LR: 3e-5 → cosine anneal to 1e-6
  - Purpose: fine-tune full model
```

### 5.3 Loss Function: Combo Loss

```python
loss = 0.5 * DiceLoss(mode='multiclass') + 0.5 * CrossEntropyLoss()
```

Dice handles class imbalance; CE provides smooth gradients. No class weighting needed with this combination.

### 5.4 Data Augmentation (Albumentations)

Critical for 40-image dataset. Applied on-the-fly per epoch:

| Transform | Target | Probability | Rationale |
|-----------|--------|-------------|-----------|
| RandomResizedCrop(512, scale=0.5-1.0) | Image+Mask | 1.0 | Effective 5-10× dataset expansion |
| HorizontalFlip | Image+Mask | 0.5 | Scene symmetry |
| Rotate(±30°) | Image+Mask | 0.7 | Orientation invariance |
| ElasticTransform | Image+Mask | 0.3 | Deformation robustness |
| BrightnessContrast | Image only | 0.7 | Lighting variation |
| HueSaturationValue | Image only | 0.4 | Color variation |
| GaussNoise | Image only | 0.3 | Sensor noise robustness |

Validation: only `Resize(512, 512)` — no augmentation.

### 5.5 Split Strategy

- 80/20 split at **original image level** (not tile level)
- Ensures tiles from the same image don't leak across train/val
- With 39 images: 31 train, 8 val
- After tiling (512², stride=384, overlap=128): ~2000 train tiles, ~500 val tiles (estimated)

---

## 6. Evaluation & Success Criteria

### 6.1 Metrics

- **Primary**: mIoU (mean Intersection over Union across all 11 classes)
- **Secondary**: Per-class IoU, Pixel Accuracy, Dice Score
- **Diagnostic**: Confusion matrix, class frequency vs prediction frequency

### 6.2 Target Thresholds (Experimental Phase)

| Metric | Minimum Acceptable | Good | Excellent |
|--------|-------------------|------|-----------|
| mIoU | > 40% | > 55% | > 65% |
| Rare class IoU (crowd, decor, vehicles) | > 15% | > 30% | > 45% |
| Dominant class IoU (sky, commercial, building) | > 50% | > 65% | > 75% |

### 6.3 Visual Verification

Every 20 epochs: save 4 random validation samples with side-by-side (original → ground truth → prediction). This catches issues metrics miss (boundary quality, small object handling).

### 6.4 Known Risks

1. **Rare class collapse**: Classes 0, 7, 9 may get 0 IoU. Mitigated by Combo Loss. If still failing, add per-class Dice weighting.
2. **Overfitting**: 39 images is small. Mitigated by aggressive augmentation + early stopping. Monitor train/val gap.
3. **Tile boundary artifacts**: Overlap=128 stride=384 ensures smooth predictions. Post-inference stitching needed for full-image eval.

---

## 7. Implementation Plan (Outline)

| Phase | Steps | Est. Time |
|-------|-------|-----------|
| **0. Environment** | Create conda env, install torch + transformers + albumentations + SMP | 30 min |
| **1. Preprocessing** | Implement `preprocess.py` (RGB→idx, tile, split), run on 39 images | 1 hr |
| **2. Dataset** | Implement `dataset.py` with Albumentations pipeline | 30 min |
| **3. Training** | Implement `train.py`, run 200 epochs | 3-6 hrs (GPU) |
| **4. Evaluation** | Implement `evaluate.py`, compute metrics, generate viz | 30 min |
| **5. Iterate** | Tune hyperparams based on results, re-train if needed | Variable |

---

## 8. Dependencies

```
torch>=2.0
transformers>=4.30
segmentation-models-pytorch
albumentations
numpy
Pillow
tensorboard
scikit-learn
tqdm
```

---

## Appendix A: RGB→Class Conversion Code

```python
def build_color_map(labelmap_path: str) -> dict:
    """Parse labelmap.txt into {(r,g,b): class_id} mapping."""
    color_to_id = {}
    with open(labelmap_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(':')
            name = parts[0]
            rgb = tuple(map(int, parts[1].split(',')))
            cls_id = len(color_to_id)  # 0-indexed, in order
            color_to_id[rgb] = cls_id
    return color_to_id

def rgb_mask_to_class_ids(rgb_mask: np.ndarray, color_map: dict) -> np.ndarray:
    """Convert H×W×3 RGB mask to H×W uint8 class index mask."""
    class_mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
    for rgb, cls_id in color_map.items():
        class_mask[np.all(rgb_mask == np.array(rgb), axis=2)] = cls_id
    return class_mask
```

## Appendix B: Combo Loss Implementation

```python
class ComboLoss(nn.Module):
    """α * DiceLoss + (1-α) * CrossEntropyLoss"""
    def __init__(self, alpha=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        ce = self.ce(pred, target)
        pred_softmax = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()
        intersection = (pred_softmax * target_onehot).sum(dim=(2,3))
        union = pred_softmax.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        return self.alpha * dice_loss + (1 - self.alpha) * ce
```
