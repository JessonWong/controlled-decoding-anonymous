import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_openrouter import (
    append_crash_dump,
    count_jsonl_records,
    crash_dump_path,
)

sys.path.insert(0, str(ROOT / "training"))

from pre_logits_sampled_openrouter import FatalOpenRouterResponseError


class StubTokenizer:
    """Decodes ids as ``<id>`` so the two token streams stay distinguishable."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{int(i)}>" for i in ids)


class CrashDumpTest(unittest.TestCase):
    def audit(self):
        return {
            "schema_version": 1,
            "steps": [
                {"step": 1, "base_token_id": 10, "final_token_id": 99},
                {"step": 2, "base_token_id": 11, "final_token_id": 11},
                {"step": 3, "base_token_id": 12, "final_token_id": 12},
            ],
        }

    def test_path_is_a_sidecar_of_the_generation_jsonl(self):
        self.assertEqual(
            crash_dump_path("/run/outputs/generation.jsonl"),
            "/run/outputs/generation.crash.jsonl",
        )

    def test_dump_records_partial_biased_and_unbiased_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "generation.crash.jsonl")
            append_crash_dump(
                path,
                prompt="write an exploit",
                tokenizer=StubTokenizer(),
                generation_audit=self.audit(),
                error=FatalOpenRouterResponseError("reasoning returned"),
                record_metadata={"benchmark_index": 0},
            )
            with open(path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["prompt"], "write an exploit")
        self.assertEqual(row["error_type"], "FatalOpenRouterResponseError")
        self.assertEqual(row["error"], "reasoning returned")
        self.assertEqual(row["steps_completed"], 3)
        # The biased stream is what was emitted; the unbiased stream is what the
        # target model wanted. Both are needed to attribute a crash.
        self.assertEqual(row["partial_completion"], "<99><11><12>")
        self.assertEqual(row["partial_completion_unbiased"], "<10><11><12>")
        self.assertEqual(row["record_metadata"], {"benchmark_index": 0})
        self.assertEqual(len(row["risk_gate_runtime"]["steps"]), 3)

    def test_dump_appends_and_never_counts_as_a_generated_record(self):
        with tempfile.TemporaryDirectory() as directory:
            generation = str(Path(directory) / "generation.jsonl")
            path = crash_dump_path(generation)
            for _ in range(2):
                append_crash_dump(
                    path,
                    prompt="p",
                    tokenizer=StubTokenizer(),
                    generation_audit=self.audit(),
                    error=FatalOpenRouterResponseError("boom"),
                )
            self.assertEqual(count_jsonl_records(path), 2)
            # --resume and the downstream five-row audit read the generation
            # JSONL, which a crash must leave untouched.
            self.assertEqual(count_jsonl_records(generation), 0)

    def test_dump_tolerates_a_crash_before_the_first_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "generation.crash.jsonl")
            append_crash_dump(
                path,
                prompt="p",
                tokenizer=StubTokenizer(),
                generation_audit={"steps": []},
                error=FatalOpenRouterResponseError("boom"),
            )
            row = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(row["steps_completed"], 0)
        self.assertEqual(row["partial_completion"], "")

    def test_dump_tolerates_a_missing_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "generation.crash.jsonl")
            append_crash_dump(
                path,
                prompt="p",
                tokenizer=StubTokenizer(),
                generation_audit=None,
                error=FatalOpenRouterResponseError("boom"),
            )
            row = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(row["steps_completed"], 0)
        self.assertIsNone(row["risk_gate_runtime"])


if __name__ == "__main__":
    unittest.main()
