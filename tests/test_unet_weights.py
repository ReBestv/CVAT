import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment
    torch = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.modules.setdefault("segmentation_models_pytorch", types.SimpleNamespace())
sys.modules.setdefault(
    "torch.utils.tensorboard",
    types.SimpleNamespace(SummaryWriter=object),
)
sys.modules.setdefault(
    "dataset",
    types.SimpleNamespace(create_train_dataset=None, create_val_dataset=None),
)

from train_unet import load_local_encoder_weights  # noqa: E402


class FakeEncoder:
    def __init__(self, load_result):
        self.load_result = load_result

    def load_state_dict(self, state_dict, strict=False):
        self.loaded = state_dict
        self.strict = strict
        return self.load_result

    def state_dict(self):
        return {"weight": object()}


class FakeModel:
    def __init__(self, load_result=None):
        self.encoder = FakeEncoder(load_result)


@unittest.skipIf(torch is None, "torch is not installed")
class LocalEncoderWeightsTest(unittest.TestCase):
    def test_load_local_encoder_weights_accepts_none_load_result(self):
        model = FakeModel(load_result=None)
        weights_path = ROOT / "models" / "efficientnet-b0" / "fake.pth"

        with patch("train_unet.torch.load", return_value={"weight": torch.ones(1)}):
            with patch.object(Path, "exists", return_value=True):
                load_local_encoder_weights(model, weights_path, "cpu")

        self.assertEqual(model.encoder.loaded["weight"].item(), 1.0)
        self.assertFalse(model.encoder.strict)


if __name__ == "__main__":
    unittest.main()
