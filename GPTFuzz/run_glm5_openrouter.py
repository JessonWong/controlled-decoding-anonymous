"""Run the upstream GPTFuzzer loop against GLM-5 through OpenRouter.

The fuzzer implementation, seed data, selector, mutators, and RoBERTa
predictor are imported from the cloned upstream GPTFuzz repository.  This
entrypoint only supplies an OpenAI-compatible target/mutator backend and a
small amount of experiment bookkeeping for the remote GLM-5 endpoint.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import logging
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


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


def read_api_key_from_file(path: Path) -> str | None:
    spec = importlib.util.spec_from_file_location("gptfuzz_api_key", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "OPENROUTER_API_KEY",
        "openrouter_api_key",
        "OPEN_ROUTER_API_KEY",
        "api_key",
    ):
        value = getattr(module, name, None)
        if value:
            return str(value)
    return None


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    if args.api_key_env:
        import os

        value = os.environ.get(args.api_key_env)
        if value:
            return value
    if args.api_key_file:
        value = read_api_key_from_file(Path(args.api_key_file).expanduser())
        if value:
            return value
    raise RuntimeError(
        "OpenRouter API key not found; set OPENROUTER_API_KEY or pass "
        "--api-key-file."
    )


class OpenRouterLLM(LLM):
    """Minimal GPTFuzzer LLM adapter for OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        *,
        system_message: str | None,
        temperature: float,
        max_tokens: int,
        max_trials: int,
        retry_sleep: float,
        provider_order: list[str] | None,
        allow_fallbacks: bool,
        reasoning_enabled: bool,
        title: str,
        max_workers: int = 1,
        request_timeout: float = 120.0,
    ) -> None:
        super().__init__()
        self.model_path = model
        self.system_message = system_message
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.max_trials = max_trials
        self.retry_sleep = retry_sleep
        self.provider_order = provider_order
        self.allow_fallbacks = allow_fallbacks
        self.reasoning_enabled = reasoning_enabled
        self.max_workers = max(1, int(max_workers))
        self.request_timeout = float(request_timeout)
        self.calls = 0
        self.failures = 0
        self._counter_lock = threading.Lock()
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            max_retries=0,
            timeout=self.request_timeout,
            default_headers={
                "HTTP-Referer": "https://github.com/sherdencooper/GPTFuzz",
                "X-Title": title,
            },
        )

    def _extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "reasoning": {"enabled": self.reasoning_enabled},
        }
        if self.provider_order:
            body["provider"] = {
                "order": self.provider_order,
                "allow_fallbacks": self.allow_fallbacks,
            }
        return body

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
        max_trials: int | None = None,
        failure_sleep_time: float | None = None,
    ) -> list[str]:
        temperature = self.default_temperature if temperature is None else temperature
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        max_trials = self.max_trials if max_trials is None else max_trials
        failure_sleep_time = self.retry_sleep if failure_sleep_time is None else failure_sleep_time
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_trials + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    extra_body=self._extra_body(),
                )
                with self._counter_lock:
                    self.calls += 1
                outputs = []
                for choice in response.choices:
                    content = choice.message.content or " "
                    outputs.append(str(content))
                if len(outputs) != n:
                    raise RuntimeError(
                        f"OpenRouter returned {len(outputs)} choices, expected {n}."
                    )
                return outputs
            except Exception as exc:  # noqa: BLE001 - upstream also retries broad API errors
                with self._counter_lock:
                    self.failures += 1
                if attempt >= max_trials:
                    logging.error(
                        "OpenRouter request failed after %d retries for model %s: %s",
                        max_trials,
                        self.model_path,
                        exc,
                    )
                    break
                logging.warning(
                    "OpenRouter request failed (%d/%d) for model %s: %s",
                    attempt + 1,
                    max_trials,
                    self.model_path,
                    exc,
                )
                time.sleep(failure_sleep_time)
        return [" "] * n

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate one response per prompt, preserving prompt order.

        GPTFuzzer's multi-question mode calls this once per mutated template.
        The upstream implementation can use a local batch backend; for the
        remote OpenRouter backend we use bounded I/O concurrency instead.
        """
        if self.max_workers == 1 or len(prompts) <= 1:
            return [self.generate(prompt)[0] for prompt in prompts]

        def generate_one(prompt: str) -> str:
            return self.generate(prompt)[0]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(generate_one, prompts))


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
        raise ValueError(f"No seed prompts found in {path}.")
    return seeds


def load_questions(path: Path, indices: list[int]) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and {"index", "text"}.issubset(rows[0]):
        by_index = {int(row["index"]): row["text"] for row in rows}
    elif rows and "goal" in rows[0]:
        # AdvBench's harmful_behaviors.csv uses goal,target and has no index.
        # Preserve the source-file order so indices 0..99 mean the first 100
        # AdvBench behaviors.
        by_index = {index: row["goal"] for index, row in enumerate(rows)}
    else:
        raise ValueError(
            f"Unsupported question file format in {path}; expected index/text or goal/target."
        )
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"Question indices not found in {path}: {missing}")
    return [by_index[index] for index in indices]


def parse_indices(value: str) -> list[int]:
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("At least one question index is required.")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official GPTFuzz against GLM-5 via OpenRouter")
    parser.add_argument("--target-model", default="z-ai/glm-5")
    parser.add_argument("--mutator-model", default="z-ai/glm-5")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--provider-order", nargs="+", default=["DeepInfra"])
    parser.add_argument("--allow-fallbacks", action="store_true")
    parser.add_argument("--reasoning", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--api-key-file", default=str(PROJECT_ROOT / "api_key.py"))
    parser.add_argument("--seed-path", default=str(REPO_ROOT / "datasets/prompts/GPTFuzzer.csv"))
    parser.add_argument("--seed-limit", type=int, default=None)
    parser.add_argument("--questions-path", default=str(REPO_ROOT / "datasets/questions/question_list.csv"))
    parser.add_argument("--question-indices", type=parse_indices, default=[5, 2])
    parser.add_argument("--predictor-model", default="hubert233/GPTFuzz")
    parser.add_argument("--predictor-device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-query", type=int, default=1000)
    parser.add_argument("--max-jailbreak", type=int, default=1)
    parser.add_argument("--energy", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--mutator-max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retry-count", type=int, default=8)
    parser.add_argument("--retry-sleep", type=float, default=15.0)
    parser.add_argument(
        "--target-workers",
        type=int,
        default=1,
        help="Bounded concurrent target requests for multi-question mode.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request OpenRouter timeout in seconds.",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--title", default="GPTFuzz GLM-5 reproduction")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="[%H:%M:%S]",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    api_key = resolve_api_key(args)
    seeds = load_seeds(Path(args.seed_path).expanduser().resolve(), args.seed_limit)
    questions = load_questions(Path(args.questions_path).expanduser().resolve(), args.question_indices)

    provider_order = list(args.provider_order) if args.provider_order else None
    reasoning_enabled = args.reasoning == "enabled"
    mutator = OpenRouterLLM(
        args.mutator_model,
        api_key,
        args.base_url,
        system_message="You are a helpful assistant.",
        temperature=args.temperature,
        max_tokens=args.mutator_max_tokens,
        max_trials=args.retry_count,
        retry_sleep=args.retry_sleep,
        provider_order=provider_order,
        allow_fallbacks=args.allow_fallbacks,
        reasoning_enabled=reasoning_enabled,
        title=args.title,
        max_workers=1,
        request_timeout=args.request_timeout,
    )
    target = OpenRouterLLM(
        args.target_model,
        api_key,
        args.base_url,
        system_message=None,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        max_trials=args.retry_count,
        retry_sleep=args.retry_sleep,
        provider_order=provider_order,
        allow_fallbacks=args.allow_fallbacks,
        reasoning_enabled=reasoning_enabled,
        title=args.title,
        max_workers=args.target_workers,
        request_timeout=args.request_timeout,
    )
    predictor = RoBERTaPredictor(args.predictor_model, device=args.predictor_device)

    fuzzer = GPTFuzzer(
        questions=questions,
        target=target,
        predictor=predictor,
        initial_seed=seeds,
        mutate_policy=MutateRandomSinglePolicy(
            [
                OpenAIMutatorCrossOver(
                    mutator,
                    temperature=args.temperature,
                    max_tokens=args.mutator_max_tokens,
                    max_trials=args.retry_count,
                    failure_sleep_time=args.retry_sleep,
                ),
                OpenAIMutatorExpand(
                    mutator,
                    temperature=args.temperature,
                    max_tokens=args.mutator_max_tokens,
                    max_trials=args.retry_count,
                    failure_sleep_time=args.retry_sleep,
                ),
                OpenAIMutatorGenerateSimilar(
                    mutator,
                    temperature=args.temperature,
                    max_tokens=args.mutator_max_tokens,
                    max_trials=args.retry_count,
                    failure_sleep_time=args.retry_sleep,
                ),
                OpenAIMutatorRephrase(
                    mutator,
                    temperature=args.temperature,
                    max_tokens=args.mutator_max_tokens,
                    max_trials=args.retry_count,
                    failure_sleep_time=args.retry_sleep,
                ),
                OpenAIMutatorShorten(
                    mutator,
                    temperature=args.temperature,
                    max_tokens=args.mutator_max_tokens,
                    max_trials=args.retry_count,
                    failure_sleep_time=args.retry_sleep,
                ),
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
        "target_model": args.target_model,
        "mutator_model": args.mutator_model,
        "base_url": args.base_url,
        "provider_order": provider_order,
        "allow_fallbacks": args.allow_fallbacks,
        "reasoning": args.reasoning,
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
        "target_workers": args.target_workers,
        "request_timeout": args.request_timeout,
        "generate_in_batch": args.target_workers > 1,
        "random_seed": args.seed,
        "output": str(output),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    logging.info(
        "Starting official GPTFuzz: target=%s mutator=%s questions=%d seeds=%d max_query=%d",
        args.target_model,
        args.mutator_model,
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
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main(parse_args())
