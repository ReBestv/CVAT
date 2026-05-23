"""
Loss functions for semantic segmentation.
ComboLoss = α * DiceLoss + (1-α) * CrossEntropyLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComboLoss(nn.Module):
    """
    α * DiceLoss + (1-α) * CrossEntropyLoss

    Dice handles class imbalance naturally (intersection-over-union based).
    CE provides smooth, stable gradients.

    Args:
        alpha: Weight for Dice loss (0-1). Default 0.5 gives equal weight.
        smooth: Smoothing factor for Dice to avoid division by zero.
        num_classes: Number of classes for one-hot encoding.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        alpha: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred: (B, C, H, W) raw logits
            target: (B, H, W) class indices (int64)

        Returns:
            total_loss: scalar loss
            loss_dict: {"ce": float, "dice": float} for logging
        """
        valid = target != self.ignore_index
        if not valid.any():
            zero = pred.sum() * 0.0
            return zero, {"ce": 0.0, "dice": 0.0}

        ce_loss = self.ce(pred, target)

        # Dice loss: compute per class over labeled pixels, then average
        pred_softmax = F.softmax(pred, dim=1)  # (B, C, H, W)

        safe_target = target.clone()
        safe_target[~valid] = 0
        target_onehot = (
            F.one_hot(safe_target, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )  # (B, C, H, W)
        valid_mask = valid.unsqueeze(1)
        pred_softmax = pred_softmax * valid_mask
        target_onehot = target_onehot * valid_mask

        intersection = (pred_softmax * target_onehot).sum(dim=(0, 2, 3))  # (C,)
        cardinality = pred_softmax.sum(dim=(0, 2, 3)) + target_onehot.sum(
            dim=(0, 2, 3)
        )  # (C,)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)  # (C,)
        dice_loss = 1.0 - dice_score.mean()

        total_loss = self.alpha * dice_loss + (1.0 - self.alpha) * ce_loss

        return total_loss, {"ce": ce_loss.item(), "dice": dice_loss.item()}


class DiceLoss(nn.Module):
    """Pure Dice loss for reference / experimentation."""

    def __init__(
        self, num_classes: int, ignore_index: int = 255, smooth: float = 1e-6
    ):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not valid.any():
            return pred.sum() * 0.0

        pred_softmax = F.softmax(pred, dim=1)
        safe_target = target.clone()
        safe_target[~valid] = 0
        target_onehot = (
            F.one_hot(safe_target, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        valid_mask = valid.unsqueeze(1)
        pred_softmax = pred_softmax * valid_mask
        target_onehot = target_onehot * valid_mask
        intersection = (pred_softmax * target_onehot).sum(dim=(0, 2, 3))
        cardinality = pred_softmax.sum(dim=(0, 2, 3)) + target_onehot.sum(
            dim=(0, 2, 3)
        )
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()
