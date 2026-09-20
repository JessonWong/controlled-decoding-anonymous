import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from mc_reconstruction import fuse_proxy_logits_with_mc_counts
from rematerialize_proxy_mc_cache import (
    canonical_json_bytes,
    rematerialize_cache,
    sha256_file,
)


class ProxyMCCacheRematerializationTest(unittest.TestCase):
    def _write_input(self, root: Path) -> tuple[Path, Path, torch.Tensor, torch.Tensor]:
        input_dir = root / "input"
        input_dir.mkdir()
        proxy_logits = torch.tensor(
            [[[2.0, 0.0, -1.0], [-1.0, 0.0, 2.0]]], dtype=torch.float16
        )
        mc_counts = torch.tensor([[[3, 1, 0], [0, 1, 3]]], dtype=torch.uint8)
        placeholder = fuse_proxy_logits_with_mc_counts(
            proxy_logits, mc_counts, temperature=1.0, prior_strength=10.0
        ).half()
        configuration = {
            "mc_fusion_mode": "proxy_dirichlet_v1",
            "proxy_temperature": 1.0,
            "proxy_prior_strength": 10.0,
            "shared_vocab_size": 3,
            "proxy_logits_key": "proxy_logits",
            "proxy_logits_semantics": "uncalibrated_raw_logits_shared_vocab",
            "proxy_logits_vocab_size": 3,
            "proxy_logits_dtype": "float16",
            "fused_log_probs_dtype": "float16",
        }
        metadata = dict(configuration)
        payload = {
            "proxy_logits": proxy_logits,
            "mc_counts": mc_counts,
            "log_probs": placeholder,
            "labels": torch.tensor([[0, 2]]),
            "metadata": metadata,
        }
        artifact = input_dir / "record.pt"
        torch.save(payload, artifact)
        source_configuration = {"source": "sampled_openrouter"}
        source_manifest = {
            "configuration": source_configuration,
            "configuration_fingerprint": hashlib.sha256(
                canonical_json_bytes(source_configuration)
            ).hexdigest(),
        }
        manifest = {
            "manifest_schema_version": 1,
            "configuration": configuration,
            "configuration_fingerprint": hashlib.sha256(
                canonical_json_bytes(configuration)
            ).hexdigest(),
            "source_manifest": source_manifest,
            "records": [
                {
                    "file": artifact.name,
                    "dataset_idx": 100,
                    "rows": 2,
                    "output_sha256": sha256_file(artifact),
                }
            ],
            "summary": {"record_count": 1, "row_count": 2},
        }
        (input_dir / "proxy_mc_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        calibration_path = root / "calibration.json"
        calibration = {
            "schema_version": 1,
            "tool": "training/eval_proxy_mc_fusion.py",
            "selection": {
                "objective": "aggregate_held_out_mc_event_mean_nll",
                "uses_answer_labels": False,
                "uses_risk_gate_token_ids": False,
                "split_method": "per_token_binomial",
                "calibration_fraction": 0.5,
                "split_seeds": [0, 1, 2],
                "selected_fusion": {
                    "proxy_temperature": 0.5,
                    "prior_strength": 2.0,
                },
            },
            "data": {"input_dir": str(input_dir.resolve())},
        }
        calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
        return input_dir, calibration_path, proxy_logits, mc_counts

    def test_rewrites_only_fusion_and_builds_verified_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir, calibration_path, proxy_logits, mc_counts = self._write_input(root)
            output_dir = root / "selected"

            result = rematerialize_cache(
                input_dir,
                calibration_path,
                output_dir,
                store_dtype="float32",
                device="cpu",
            )

            self.assertEqual(result["proxy_temperature"], 0.5)
            self.assertEqual(result["proxy_prior_strength"], 2.0)
            output_payload = torch.load(
                output_dir / "record.pt", map_location="cpu", weights_only=False
            )
            torch.testing.assert_close(output_payload["proxy_logits"], proxy_logits)
            torch.testing.assert_close(output_payload["mc_counts"], mc_counts)
            expected = fuse_proxy_logits_with_mc_counts(
                proxy_logits,
                mc_counts,
                temperature=0.5,
                prior_strength=2.0,
            )
            torch.testing.assert_close(output_payload["log_probs"], expected)
            self.assertEqual(output_payload["log_probs"].dtype, torch.float32)
            self.assertEqual(output_payload["metadata"]["proxy_temperature"], 0.5)
            self.assertEqual(output_payload["metadata"]["proxy_prior_strength"], 2.0)

            manifest_path = output_dir / "proxy_mc_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fingerprint = hashlib.sha256(
                canonical_json_bytes(manifest["configuration"])
            ).hexdigest()
            self.assertEqual(manifest["configuration_fingerprint"], fingerprint)
            self.assertEqual(
                output_payload["metadata"]["proxy_mc_configuration_fingerprint"],
                fingerprint,
            )
            self.assertEqual(
                manifest["records"][0]["output_sha256"],
                sha256_file(output_dir / "record.pt"),
            )
            self.assertFalse(manifest["calibration"]["uses_answer_labels"])
            self.assertEqual(manifest["summary"]["rematerialization_api_call_count"], 0)
            self.assertFalse(any(root.glob(".selected.staging-*")))

            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                rematerialize_cache(input_dir, calibration_path, output_dir, device="cpu")

    def test_refuses_calibration_that_used_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir, calibration_path, _, _ = self._write_input(root)
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration["selection"]["uses_answer_labels"] = True
            calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "uses_answer_labels=false"):
                rematerialize_cache(
                    input_dir,
                    calibration_path,
                    root / "invalid-output",
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
