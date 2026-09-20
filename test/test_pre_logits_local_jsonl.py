import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from pre_logits_sampled_openrouter import load_training_data


class LocalJsonlDatasetTest(unittest.TestCase):
    def args(self, path: str) -> argparse.Namespace:
        return argparse.Namespace(
            dataset_jsonl=path,
            dataset_name="unused",
            dataset_revision=None,
            dataset_split="train",
            dataset_prompt_field="prompt",
            dataset_answer_field="rejected",
        )

    def test_loads_requested_fields_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt": "question",
                        "rejected": "teacher answer",
                        "source_dataset_idx": 123,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_training_data(self.args(str(path)))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt"], "question")
            self.assertEqual(rows[0]["rejected"], "teacher answer")
            self.assertEqual(rows[0]["source_dataset_idx"], 123)

    def test_rejects_missing_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl"
            path.write_text('{"prompt": "question"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "rejected"):
                load_training_data(self.args(str(path)))


if __name__ == "__main__":
    unittest.main()
