import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.rescore_risk_gate_cache import (
    build_post_action_prefixes,
    parse_args,
    quantiles,
    resolve_source_manifest,
    resolve_threshold,
    sha256_payload,
    validate_cache_tensors,
)


class IntegerTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return ",".join(str(value) for value in token_ids)


class RiskGateCacheRescoreTest(unittest.TestCase):
    def test_target_tokenizer_compatibility_options_are_opt_in(self):
        required = [
            "--source-dir", "source",
            "--output-dir", "output",
            "--gate-checkpoint", "gate",
            "--gate-model-name", "backbone",
            "--target-tokenizer", "tokenizer",
        ]
        defaults = parse_args(required)
        self.assertFalse(defaults.target_tokenizer_trust_remote_code)
        self.assertFalse(defaults.target_tokenizer_fix_mistral_regex)
        enabled = parse_args(
            required
            + [
                "--target-tokenizer-trust-remote-code",
                "--target-tokenizer-fix-mistral-regex",
            ]
        )
        self.assertTrue(enabled.target_tokenizer_trust_remote_code)
        self.assertTrue(enabled.target_tokenizer_fix_mistral_regex)

    def test_prefixes_match_teacher_forced_target_plus_current_base_action(self):
        prefixes = build_post_action_prefixes(
            IntegerTokenizer(),
            target_ids=[10, 11, 12],
            base_token_ids=[20, 21, 22],
        )
        self.assertEqual(prefixes, ["20", "10,21", "10,11,22"])

    def test_threshold_must_match_checkpoint_operating_point(self):
        config = {"recommended_handoff_threshold": 0.9641336778984433}
        self.assertEqual(resolve_threshold(config, None), config["recommended_handoff_threshold"])
        with self.assertRaisesRegex(ValueError, "differs"):
            resolve_threshold(config, 0.1)

    def test_cache_tensor_validation_fails_on_misalignment(self):
        payload = {
            "labels": torch.tensor([[1, 2]], dtype=torch.long),
            "risk_gate_token_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "risk_gate_scores": torch.tensor([[0.2]], dtype=torch.float32),
            "risk_gate_mask": torch.tensor([[True, False]], dtype=torch.bool),
        }
        with self.assertRaisesRegex(ValueError, "align"):
            validate_cache_tensors(payload, Path("record.pt"))

    def test_quantiles_are_json_finite_scalars(self):
        result = quantiles(torch.tensor([0.0, 0.5, 1.0]))
        self.assertEqual(result["p00"], 0.0)
        self.assertEqual(result["p50"], 0.5)
        self.assertEqual(result["p100"], 1.0)
        json.dumps(result, allow_nan=False)

    def test_resolves_legacy_raw_manifest_without_record_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = {"tokenizer_name": "Qwen/Qwen3-32B"}
            manifest = {
                "configuration": configuration,
                "configuration_fingerprint": sha256_payload(configuration),
            }
            path = root / "cache_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            actual_path, actual_manifest, has_ledger = resolve_source_manifest(root)
            self.assertEqual(actual_path, path)
            self.assertEqual(actual_manifest, manifest)
            self.assertFalse(has_ledger)

    def test_rejects_legacy_raw_manifest_with_bad_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache_manifest.json").write_text(
                json.dumps(
                    {
                        "configuration": {"tokenizer_name": "Qwen/Qwen3-32B"},
                        "configuration_fingerprint": "wrong",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                resolve_source_manifest(root)


if __name__ == "__main__":
    unittest.main()
