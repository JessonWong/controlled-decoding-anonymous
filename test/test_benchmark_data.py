import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_data import collect_prompt_records, load_benchmark_records


class BenchmarkDataTest(unittest.TestCase):
    def test_harmbench_csv_preserves_ids_and_contextual_prompt_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "behaviors.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Behavior",
                        "FunctionalCategory",
                        "SemanticCategory",
                        "Tags",
                        "ContextString",
                        "BehaviorID",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Behavior": "behavior",
                        "FunctionalCategory": "standard",
                        "SemanticCategory": "category",
                        "Tags": "context",
                        "ContextString": "context",
                        "BehaviorID": "behavior-1",
                    }
                )
            records = load_benchmark_records("harmbench", path)

        self.assertEqual(records[0].prompt, "context\n\n---\n\nbehavior")
        self.assertEqual(records[0].metadata["benchmark_id"], "behavior-1")
        self.assertEqual(records[0].metadata["benchmark"], "harmbench")

    def test_sorrybench_accepts_turns_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "question.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question_id": 7,
                        "category": "crime",
                        "turns": ["unsafe instruction"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_benchmark_records("sorrybench", path)

        self.assertEqual(records[0].prompt, "unsafe instruction")
        self.assertEqual(records[0].metadata["benchmark_id"], 7)
        self.assertEqual(records[0].metadata["benchmark_category"], "crime")

    def test_benchmark_cannot_be_combined_with_prompt_file(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            collect_prompt_records(
                benchmark="advbench",
                prompt="prompt",
            )

    def test_plain_prompt_file_remains_metadata_free(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.txt"
            path.write_text("first\n\nsecond\n", encoding="utf-8")
            records = collect_prompt_records(prompt_file=path)

        self.assertEqual([record.prompt for record in records], ["first", "second"])
        self.assertEqual(records[0].metadata, {})


if __name__ == "__main__":
    unittest.main()
