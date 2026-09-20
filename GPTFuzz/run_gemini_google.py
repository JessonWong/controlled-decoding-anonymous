"""Run the upstream GPTFuzz loop directly against Google's Gemini API."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path

from google import genai
from google.genai import types


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from gptfuzzer.fuzzer import GPTFuzzer  # noqa: E402
from gptfuzzer.fuzzer.mutator import (  # noqa: E402
    MutateRandomSinglePolicy,
    OpenAIMutatorCrossOver,
    OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar,
    OpenAIMutatorRephrase,
    OpenAIMutatorShorten,
)
from gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy  # noqa: E402
from gptfuzzer.llm import LLM  # noqa: E402
from gptfuzzer.utils.predict import RoBERTaPredictor  # noqa: E402


def read_api_key(path: Path) -> str:
    value = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if value:
        return value
    spec = importlib.util.spec_from_file_location("gptfuzz_google_api_key", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load API key file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY_ALTERNATE",
        "GEMINI_API_KEY_SECONDARY",
    ):
        value = getattr(module, name, None)
        if value:
            return str(value)
    raise RuntimeError(f"No Google Gemini API key found in {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_seeds(path: Path, limit: int | None) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seeds = [row["text"] for row in rows if row.get("text")]
    if limit is not None:
        seeds = seeds[:limit]
    if not seeds:
        raise ValueError(f"No seed prompts found in {path}")
    return seeds


def load_questions(path: Path, indices: list[int]) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and {"index", "text"}.issubset(rows[0]):
        by_index = {int(row["index"]): row["text"] for row in rows}
    elif rows and "goal" in rows[0]:
        by_index = {index: row["goal"] for index, row in enumerate(rows)}
    else:
        raise ValueError(f"Unsupported question file format: {path}")
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"Question indices not found in {path}: {missing}")
    return [by_index[index] for index in indices]


def parse_indices(value: str) -> list[int]:
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("At least one question index is required")
    return indices


class GeminiGoogleLLM(LLM):
    """Threaded, order-preserving Gemini API adapter for GPTFuzz."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        system_message: str | None,
        temperature: float,
        max_tokens: int,
        max_trials: int,
        retry_sleep: float,
        max_workers: int,
        thinking_budget: int,
        safety_threshold: str,
    ) -> None:
        super().__init__()
        self.model_path = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_trials = max_trials
        self.retry_sleep = retry_sleep
        self.max_workers = max(1, int(max_workers))
        self.thinking_budget = thinking_budget
        self.safety_threshold = safety_threshold
        self.client = genai.Client(api_key=api_key)
        self.calls = 0
        self.failures = 0
        self._counter_lock = threading.Lock()

    def _config(self, temperature: float, max_tokens: int):
        categories = (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        )
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.95,
            top_k=40,
            system_instruction=self.system_message or None,
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
            safety_settings=[
                types.SafetySetting(category=category, threshold=self.safety_threshold)
                for category in categories
            ],
        )

    @staticmethod
    def _response_text(response) -> str:
        try:
            text = response.text
        except Exception:  # blocked/empty SDK responses may raise on .text
            text = None
        if text:
            return str(text)
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", None) or []
            text_parts = [getattr(part, "text", "") for part in parts]
            return "".join(part for part in text_parts if part)
        return " "

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
        max_trials: int | None = None,
        failure_sleep_time: float | None = None,
    ) -> list[str]:
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        max_trials = self.max_trials if max_trials is None else max_trials
        failure_sleep_time = (
            self.retry_sleep if failure_sleep_time is None else failure_sleep_time
        )
        outputs = []
        for _ in range(n):
            result = " "
            for attempt in range(max_trials + 1):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_path,
                        contents=prompt,
                        config=self._config(temperature, max_tokens),
                    )
                    result = self._response_text(response)
                    with self._counter_lock:
                        self.calls += 1
                    break
                except Exception as exc:  # noqa: BLE001 - retry API failures
                    with self._counter_lock:
                        self.failures += 1
                    if attempt >= max_trials:
                        logging.error(
                            "Gemini request failed after %d retries for model %s: %s",
                            max_trials,
                            self.model_path,
                            exc,
                        )
                        break
                    logging.warning(
                        "Gemini request failed (%d/%d) for model %s: %s",
                        attempt + 1,
                        max_trials,
                        self.model_path,
                        exc,
                    )
                    time.sleep(failure_sleep_time)
            outputs.append(result)
        return outputs

    def generate_batch(self, prompts: list[str]) -> list[str]:
        if self.max_workers == 1 or len(prompts) <= 1:
            return [self.generate(prompt)[0] for prompt in prompts]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # executor.map preserves the input question order required by GPTFuzz.
            return list(executor.map(lambda prompt: self.generate(prompt)[0], prompts))


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="[%H:%M:%S]",
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    api_key = read_api_key(Path(args.api_key_file).expanduser().resolve())
    seeds = load_seeds(Path(args.seed_path).expanduser().resolve(), args.seed_limit)
    questions = load_questions(
        Path(args.questions_path).expanduser().resolve(), args.question_indices
    )

    mutator = GeminiGoogleLLM(
        args.model,
        api_key,
        system_message="You are a helpful assistant.",
        temperature=args.temperature,
        max_tokens=args.mutator_max_tokens,
        max_trials=args.retry_count,
        retry_sleep=args.retry_sleep,
        max_workers=1,
        thinking_budget=args.thinking_budget,
        safety_threshold=args.safety_threshold,
    )
    target = GeminiGoogleLLM(
        args.model,
        api_key,
        system_message=None,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        max_trials=args.retry_count,
        retry_sleep=args.retry_sleep,
        max_workers=args.target_workers,
        thinking_budget=args.thinking_budget,
        safety_threshold=args.safety_threshold,
    )
    predictor = RoBERTaPredictor(args.predictor_model, device=args.predictor_device)
    fuzzer = GPTFuzzer(
        questions=questions,
        target=target,
        predictor=predictor,
        initial_seed=seeds,
        mutate_policy=MutateRandomSinglePolicy(
            [
                OpenAIMutatorCrossOver(mutator, temperature=args.temperature, max_tokens=args.mutator_max_tokens, max_trials=args.retry_count, failure_sleep_time=args.retry_sleep),
                OpenAIMutatorExpand(mutator, temperature=args.temperature, max_tokens=args.mutator_max_tokens, max_trials=args.retry_count, failure_sleep_time=args.retry_sleep),
                OpenAIMutatorGenerateSimilar(mutator, temperature=args.temperature, max_tokens=args.mutator_max_tokens, max_trials=args.retry_count, failure_sleep_time=args.retry_sleep),
                OpenAIMutatorRephrase(mutator, temperature=args.temperature, max_tokens=args.mutator_max_tokens, max_trials=args.retry_count, failure_sleep_time=args.retry_sleep),
                OpenAIMutatorShorten(mutator, temperature=args.temperature, max_tokens=args.mutator_max_tokens, max_trials=args.retry_count, failure_sleep_time=args.retry_sleep),
            ],
            concatentate=True,
        ),
        select_policy=MCTSExploreSelectPolicy(),
        energy=args.energy,
        max_jailbreak=args.max_jailbreak,
        max_query=args.max_query,
        generate_in_batch=args.target_workers > 1,
        result_file=str(output),
    )
    manifest = {
        "source_repo": "https://github.com/sherdencooper/GPTFuzz.git",
        "source_commit": "0c26ccc",
        "backend": "google_gemini_api",
        "target_model": args.model,
        "mutator_model": args.model,
        "questions": questions,
        "question_indices": args.question_indices,
        "questions_path": str(Path(args.questions_path).expanduser().resolve()),
        "questions_sha256": sha256_file(Path(args.questions_path).expanduser().resolve()),
        "seed_path": str(Path(args.seed_path).expanduser().resolve()),
        "seed_sha256": sha256_file(Path(args.seed_path).expanduser().resolve()),
        "seed_count": len(seeds),
        "predictor_model": args.predictor_model,
        "max_query": args.max_query,
        "max_jailbreak": args.max_jailbreak,
        "energy": args.energy,
        "max_new_tokens": args.max_new_tokens,
        "mutator_max_tokens": args.mutator_max_tokens,
        "temperature": args.temperature,
        "thinking_budget": args.thinking_budget,
        "safety_threshold": args.safety_threshold,
        "target_workers": args.target_workers,
        "generate_in_batch": args.target_workers > 1,
        "random_seed": args.seed,
        "output": str(output),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(__import__("json").dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logging.info(
        "Starting official GPTFuzz: backend=google_gemini_api model=%s questions=%d seeds=%d max_query=%d",
        args.model,
        len(questions),
        len(seeds),
        args.max_query,
    )
    try:
        fuzzer.run()
    finally:
        manifest.update(
            {
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "mutator_api_calls": mutator.calls,
                "mutator_failures": mutator.failures,
                "target_api_calls": target.calls,
                "target_failures": target.failures,
            }
        )
        manifest_path.write_text(__import__("json").dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official GPTFuzz against Gemini via Google API")
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--api-key-file", default=str(PROJECT_ROOT / "api_key.py"))
    parser.add_argument("--seed-path", default=str(REPO_ROOT / "datasets/prompts/GPTFuzzer.csv"))
    parser.add_argument("--seed-limit", type=int, default=None)
    parser.add_argument("--questions-path", required=True)
    parser.add_argument("--question-indices", type=parse_indices, required=True)
    parser.add_argument("--predictor-model", default="hubert233/GPTFuzz")
    parser.add_argument("--predictor-device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-query", type=int, default=1000)
    parser.add_argument("--max-jailbreak", type=int, default=-1)
    parser.add_argument("--energy", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--mutator-max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retry-count", type=int, default=8)
    parser.add_argument("--retry-sleep", type=float, default=15.0)
    parser.add_argument("--target-workers", type=int, default=8)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--safety-threshold", default="BLOCK_NONE")
    parser.add_argument("--seed", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
