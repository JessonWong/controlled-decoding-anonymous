import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risk_gate import validate_backbone_compatibility


class RiskGateBackboneCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.backbone_config = SimpleNamespace(
            hidden_size=4096,
            vocab_size=128256,
            num_hidden_layers=32,
        )
        self.risk_config = {
            "hidden_size": 4096,
            "layer_indices": [-1],
        }

    def test_llama31_shape_is_compatible(self):
        validate_backbone_compatibility(
            backbone_config=self.backbone_config,
            risk_config=self.risk_config,
            tokenizer_vocab_size=128256,
            backbone_name="meta-llama/Llama-3.1-8B-Instruct",
        )

    def test_hidden_size_mismatch_fails_before_loading_weights(self):
        incompatible = SimpleNamespace(
            hidden_size=3072,
            vocab_size=128256,
            num_hidden_layers=32,
        )
        with self.assertRaisesRegex(ValueError, "hidden_size=3072"):
            validate_backbone_compatibility(
                backbone_config=incompatible,
                risk_config=self.risk_config,
                tokenizer_vocab_size=128256,
                backbone_name="smaller-model",
            )

    def test_tokenizer_vocab_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "checkpoint tokenizer"):
            validate_backbone_compatibility(
                backbone_config=self.backbone_config,
                risk_config=self.risk_config,
                tokenizer_vocab_size=32000,
                backbone_name="wrong-tokenizer-model",
            )


if __name__ == "__main__":
    unittest.main()
