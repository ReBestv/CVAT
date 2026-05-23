import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment
    torch = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from losses import ComboLoss  # noqa: E402


@unittest.skipIf(torch is None, "torch is not installed")
class ComboLossIgnoreIndexTest(unittest.TestCase):
    def test_all_ignored_pixels_have_zero_loss(self):
        criterion = ComboLoss(
            num_classes=config.NUM_CLASSES,
            ignore_index=config.IGNORE_INDEX,
        )
        logits = torch.randn(1, config.NUM_CLASSES, 2, 2, requires_grad=True)
        target = torch.full((1, 2, 2), config.IGNORE_INDEX, dtype=torch.long)

        loss, parts = criterion(logits, target)

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(parts, {"ce": 0.0, "dice": 0.0})

    def test_mixed_targets_ignore_255_without_error(self):
        criterion = ComboLoss(
            num_classes=config.NUM_CLASSES,
            ignore_index=config.IGNORE_INDEX,
        )
        logits = torch.randn(1, config.NUM_CLASSES, 2, 2, requires_grad=True)
        target = torch.tensor([[[config.IGNORE_INDEX, 0], [3, 8]]], dtype=torch.long)

        loss, parts = criterion(logits, target)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(parts["ce"], 0.0)
        self.assertGreaterEqual(parts["dice"], 0.0)


if __name__ == "__main__":
    unittest.main()
