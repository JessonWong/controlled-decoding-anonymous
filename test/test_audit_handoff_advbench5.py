import unittest
from pathlib import Path

from training.audit_handoff_advbench5 import (
    EXPECTED_TOKENIZER_SHA256,
    audit_fusion_metadata,
    audited_proxy_calls,
    parse_args,
    require_checkpoint_fusion_contract,
    require_runtime_fusion_contract,
)


class HandoffFusionAuditContractTest(unittest.TestCase):
    def test_fusion_mode_defaults_to_proxy(self):
        common_args = [
            "--generation",
            "generation.jsonl",
            "--audit",
            "audit.json",
            "--checkpoint",
            "checkpoint",
            "--gate",
            "gate",
            "--benchmark",
            "advbench.csv",
            "--proxy-snapshot",
            "snapshot",
            "--proxy-revision",
            "revision",
        ]
        args = parse_args(common_args)
        self.assertEqual(args.fusion_mode, "proxy")
        raw_args = parse_args([*common_args, "--fusion-mode", "raw"])
        self.assertEqual(raw_args.fusion_mode, "raw")

    def test_raw_checkpoint_and_runtime_accept_no_proxy(self):
        checkpoint_config = {
            "mc_fusion_mode": None,
            "proxy_temperature": None,
            "proxy_prior_strength": None,
            "proxy_model_name_or_path": None,
        }
        require_checkpoint_fusion_contract(checkpoint_config, "raw")
        require_runtime_fusion_contract(
            {}, "raw", Path("snapshot"), "revision", "row 0"
        )
        self.assertEqual(audited_proxy_calls({}, "raw", 7, "row 0"), 0)
        self.assertEqual(
            audited_proxy_calls({"proxy_calls": 0}, "raw", 7, "row 0"), 0
        )
        self.assertEqual(
            audit_fusion_metadata("raw"),
            {"fusion_mode": "raw_mc50", "proxy": None},
        )

    def test_raw_contract_rejects_proxy_evidence(self):
        with self.assertRaisesRegex(ValueError, "non-null proxy fields"):
            require_checkpoint_fusion_contract(
                {"mc_fusion_mode": None, "proxy_temperature": 2.5}, "raw"
            )
        with self.assertRaisesRegex(ValueError, "proxy_fusion"):
            require_runtime_fusion_contract(
                {"proxy_fusion": {}},
                "raw",
                Path("snapshot"),
                "revision",
                "row 0",
            )
        with self.assertRaisesRegex(ValueError, "non-zero proxy calls"):
            audited_proxy_calls({"proxy_calls": 1}, "raw", 7, "row 0")

    def test_proxy_contract_is_unchanged(self):
        snapshot = Path("snapshot")
        require_checkpoint_fusion_contract(
            {
                "mc_fusion_mode": "proxy_dirichlet_v1",
                "proxy_temperature": 2.5,
                "proxy_prior_strength": 8.0,
                "proxy_tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            },
            "proxy",
        )
        require_runtime_fusion_contract(
            {
                "proxy_fusion": {
                    "mc_fusion_mode": "proxy_dirichlet_v1",
                    "proxy_model_name_or_path": str(snapshot),
                    "proxy_model_revision": "revision",
                    "proxy_tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
                    "proxy_temperature": 2.5,
                    "proxy_prior_strength": 8.0,
                    "proxy_dtype": "float16",
                    "proxy_quantization": "none",
                }
            },
            "proxy",
            snapshot,
            "revision",
            "row 0",
        )
        self.assertEqual(
            audited_proxy_calls({"proxy_calls": 7}, "proxy", 7, "row 0"), 7
        )
        self.assertEqual(
            audit_fusion_metadata("proxy"),
            {"proxy": {"temperature": 2.5, "prior_strength": 8.0}},
        )


if __name__ == "__main__":
    unittest.main()
