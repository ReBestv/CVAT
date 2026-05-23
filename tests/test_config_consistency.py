import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from preprocess import rgb_mask_to_class_ids  # noqa: E402


class ConfigConsistencyTest(unittest.TestCase):
    def test_class_configuration_ignores_background_label(self):
        label_lines = [
            line.strip()
            for line in config.LABELMAP_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]

        self.assertEqual(config.NUM_CLASSES, len(config.CLASS_NAMES))
        self.assertEqual(config.NUM_CLASSES + 1, len(label_lines))
        self.assertEqual(config.IGNORE_INDEX, 255)
        self.assertNotIn("background", config.CLASS_NAMES)
        self.assertEqual(config.COLOR_TO_ID[(0, 0, 0)], config.IGNORE_INDEX)
        foreground_ids = [
            cls_id
            for cls_id in config.COLOR_TO_ID.values()
            if cls_id != config.IGNORE_INDEX
        ]
        self.assertEqual(sorted(foreground_ids), list(range(config.NUM_CLASSES)))

    def test_unet_local_weights_path_is_under_models(self):
        expected = config.ROOT / "models" / "efficientnet-b0" / "efficientnet-b0-355c32eb.pth"

        self.assertEqual(config.UNET_ENCODER_LOCAL_WEIGHTS, expected)
        self.assertEqual(config.UNET_ENCODER_LOCAL_WEIGHTS.suffix, ".pth")

    def test_deeplab_uses_mobilenetv3_encoder(self):
        self.assertEqual(config.DEEPLAB_ENCODER, "timm-mobilenetv3_large_100")
        self.assertEqual(config.DEEPLAB_ENCODER_WEIGHTS, "imagenet")
        self.assertEqual(config.DEEPLAB_CHECKPOINT_DIR, config.CHECKPOINT_DIR / "deeplab")

    def test_rgb_mask_maps_background_to_ignore_index(self):
        rgb_mask = np.array(
            [
                [(0, 0, 0), (12, 54, 138), (78, 78, 78)],
            ],
            dtype=np.uint8,
        )

        class_mask = rgb_mask_to_class_ids(rgb_mask, config.COLOR_TO_ID)

        np.testing.assert_array_equal(
            class_mask,
            np.array([[config.IGNORE_INDEX, 0, 8]], dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main()
