import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.flip_attack.attack import FlipAttack
from baselines.flip_attack.evaluate import (
    build_evaluation_command,
    build_parser as build_evaluation_parser,
)
from baselines.flip_attack.generate import (
    OpenAICompatibleVictim,
    build_parser,
    run,
    validate_args,
)
from benchmark_data import load_benchmark_records


class FlipAttackPromptTest(unittest.TestCase):
    def test_four_flip_modes(self):
        prompt = "give me an apple"
        self.assertEqual(
            FlipAttack(flip_mode="FWO").build(prompt).disguised_prompt,
            "apple an me give",
        )
        self.assertEqual(
            FlipAttack(flip_mode="FCW").build(prompt).disguised_prompt,
            "evig em na elppa",
        )
        self.assertEqual(
            FlipAttack(flip_mode="FCS").build(prompt).disguised_prompt,
            "elppa na em evig",
        )
        self.assertEqual(
            FlipAttack(flip_mode="FMM").build(prompt).disguised_prompt,
            "elppa na em evig",
        )

    def test_prompt_variants_produce_two_chat_messages(self):
        attack = FlipAttack(
            flip_mode="FCS",
            cot=True,
            lang_gpt=True,
            few_shot=True,
        ).build("a short synthetic task")
        self.assertEqual([message["role"] for message in attack.messages], ["system", "user"])
        self.assertIn("Step 2", attack.user_prompt)
        self.assertIn("Here are some examples", attack.user_prompt)
        self.assertIn("step by step", attack.user_prompt)

    def test_few_shot_handles_single_word(self):
        attack = FlipAttack(few_shot=True).build("synthetic")
        self.assertIn("Here are some examples", attack.system_prompt)

    def test_llama_adjustment_is_case_insensitive(self):
        attack = FlipAttack(victim_model="meta-llama/Llama-3").build("synthetic task")
        self.assertNotIn("without any regard", attack.system_prompt)


class FlipBenchmarkDataTest(unittest.TestCase):
    def setUp(self):
        # Portable format fixtures; benchmark corpora are external inputs.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data_dir = Path(directory.name)
        fields = ["id", "goal", "Behavior", "FunctionalCategory", "category", "category_name"]
        for benchmark in ("advbench", "harmbench", "sorrybench"):
            with (self.data_dir / f"{benchmark}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index in range(2):
                    writer.writerow({
                        "id": f"fixture-{index}", "goal": f"Explain a simple concept {index}.",
                        "Behavior": f"Explain a simple concept {index}.",
                        "FunctionalCategory": "standard", "category": "fixture",
                        "category_name": "Synthetic format test",
                    })
        defaults = {name: self.data_dir / f"{name}.csv"
                    for name in ("advbench", "harmbench", "sorrybench")}
        patcher = mock.patch.dict("baselines.flip_attack.generate.DEFAULT_DATA_FILES", defaults)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_normalized_flip_csvs_load(self):
        expected = {"advbench": 2, "harmbench": 2, "sorrybench": 2}
        for benchmark, count in expected.items():
            records = load_benchmark_records(
                benchmark, self.data_dir / f"{benchmark}.csv"
            )
            self.assertEqual(len(records), count)
            self.assertEqual(records[0].metadata["benchmark"], benchmark)
            self.assertIn("benchmark_id", records[0].metadata)

        harmbench = load_benchmark_records(
            "harmbench", self.data_dir / "harmbench.csv"
        )
        self.assertIn("benchmark_functional_category", harmbench[0].metadata)
        sorrybench = load_benchmark_records(
            "sorrybench", self.data_dir / "sorrybench.csv"
        )
        self.assertIn("benchmark_category_name", sorrybench[0].metadata)

    def test_materialize_and_resume_standard_jsonl(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "attacks.jsonl"
            args = parser.parse_args(
                [
                    "--benchmark",
                    "advbench",
                    "--limit",
                    "2",
                    "--materialize-only",
                    "--output-file",
                    str(output),
                ]
            )
            validate_args(args, parser)
            run(args)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertIsNone(rows[0]["completion"])
            self.assertEqual(rows[0]["attack_method"], "flipattack")
            self.assertEqual(len(rows[0]["attack_messages"]), 2)
            self.assertEqual(
                rows[0]["flipped_prompt"], rows[0]["original_prompt"][::-1]
            )
            self.assertEqual(rows[0]["prompt"], rows[0]["input_prompt"])

            args.resume = True
            run(args)
            resumed = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(resumed), 2)

    def test_mock_victim_generation_is_evaluator_ready(self):
        class Victim:
            def generate(self, messages):
                return "synthetic completion", {"response_model": "fixture"}

        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "generation.jsonl"
            args = parser.parse_args(
                [
                    "--benchmark",
                    "sorrybench",
                    "--provider",
                    "openrouter",
                    "--model",
                    "fixture/model",
                    "--limit",
                    "1",
                    "--output-file",
                    str(output),
                ]
            )
            validate_args(args, parser)
            run(args, victim=Victim())
            row = json.loads(output.read_text())
            self.assertEqual(row["completion"], "synthetic completion")
            self.assertEqual(row["benchmark"], "sorrybench")
            self.assertIn("benchmark_category", row)

    def test_openai_compatible_client_uses_chat_messages(self):
        class Response:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "id": "fixture-response",
                        "model": "fixture/model",
                        "choices": [
                            {
                                "message": {"content": "fixture completion"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ).encode()

        args = build_parser().parse_args(
            [
                "--benchmark",
                "advbench",
                "--provider",
                "openai-compatible",
                "--base-url",
                "http://127.0.0.1:8000/v1",
                "--model",
                "fixture/model",
            ]
        )
        client = OpenAICompatibleVictim(args)
        messages = FlipAttack().build("synthetic task").messages
        with mock.patch("urllib.request.urlopen", return_value=Response()) as request:
            completion, metadata = client.generate(messages)
        sent_request = request.call_args.args[0]
        payload = json.loads(sent_request.data)
        self.assertEqual(sent_request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(completion, "fixture completion")
        self.assertEqual(metadata["response_id"], "fixture-response")

    def test_openai_compatible_client_can_omit_top_p(self):
        class Response:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "id": "fixture-response",
                        "model": "fixture/model",
                        "choices": [
                            {
                                "message": {"content": "fixture completion"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ).encode()

        args = build_parser().parse_args(
            [
                "--benchmark",
                "advbench",
                "--provider",
                "openai-compatible",
                "--base-url",
                "http://127.0.0.1:8000/v1",
                "--model",
                "fixture/model",
                "--omit-top-p",
            ]
        )
        client = OpenAICompatibleVictim(args)
        with mock.patch("urllib.request.urlopen", return_value=Response()) as request:
            client.generate(FlipAttack().build("synthetic task").messages)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertNotIn("top_p", payload)

    def test_azure_client_uses_deployment_and_api_key_header(self):
        class Response:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "id": "azure-fixture-response",
                        "model": "Mistral-Large-3-ssl",
                        "choices": [
                            {
                                "message": {"content": "fixture completion"},
                                "finish_reason": "length",
                            }
                        ],
                    }
                ).encode()

        args = build_parser().parse_args(
            [
                "--benchmark",
                "advbench",
                "--provider",
                "azure",
                "--model",
                "Mistral-Large-3-ssl",
                "--api-key",
                "fixture-key",
                "--max-new-tokens",
                "80",
            ]
        )
        client = OpenAICompatibleVictim(args)
        messages = FlipAttack().build("synthetic task").messages
        with mock.patch("urllib.request.urlopen", return_value=Response()) as request:
            _, metadata = client.generate(messages)
        sent_request = request.call_args.args[0]
        payload = json.loads(sent_request.data)
        self.assertEqual(
            sent_request.full_url,
            "https://example.openai.azure.com/openai/v1/chat/completions",
        )
        self.assertEqual(sent_request.get_header("Api-key"), "fixture-key")
        self.assertIsNone(sent_request.get_header("Authorization"))
        self.assertEqual(payload["model"], "Mistral-Large-3-ssl")
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["max_tokens"], 80)
        self.assertEqual(metadata["response_model"], "Mistral-Large-3-ssl")

    def test_evaluation_router_selects_expected_scripts(self):
        parser = build_evaluation_parser()
        for benchmark, script in (
            ("advbench", "eval_harmful_score.py"),
            ("harmbench", "eval_harmbench.py"),
            ("sorrybench", "eval_sorrybench.py"),
        ):
            args = parser.parse_args(
                ["--benchmark", benchmark, "--input-file", "fixture.jsonl"]
            )
            command = build_evaluation_command(args)
            self.assertTrue(any(part.endswith(script) for part in command))


if __name__ == "__main__":
    unittest.main()
