"""Small OpenRouter chat-completions client used by adaptive baselines.

The project already has several endpoint-specific clients.  This module keeps
the PAIR and LogiBreak reference-style runners independent of optional SDKs,
while preserving provider routing and usage metadata in every response.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


class OpenRouterClient:
    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        api_key_file: str | None = None,
        provider_order: Iterable[str] = ("DeepInfra",),
        allow_fallbacks: bool = False,
        request_timeout: float = 120.0,
        max_retries: int = 8,
        retry_sleep: float = 5.0,
        reasoning_enabled: bool = False,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = self._resolve_key(api_key_env, api_key_file)
        self.provider_order = list(provider_order)
        self.allow_fallbacks = bool(allow_fallbacks)
        self.request_timeout = float(request_timeout)
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep)
        self.reasoning_enabled = bool(reasoning_enabled)
        self.calls = 0
        self.failures = 0
        self.total_cost = 0.0

    @staticmethod
    def _resolve_key(env_name: str, api_key_file: str | None) -> str:
        value = os.environ.get(env_name)
        if value:
            return value
        if not api_key_file:
            raise RuntimeError(f"Missing API key environment variable: {env_name}")
        path = Path(api_key_file).expanduser().resolve()
        spec = importlib.util.spec_from_file_location("_baseline_api_keys", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load API key file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidates = [
            env_name,
            "OPENROUTER_API_KEY",
        ]
        for name in candidates:
            candidate = getattr(module, name, None)
            if candidate:
                return str(candidate)
        raise RuntimeError(f"No OpenRouter key found in {path}")

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(
                item.get("text", "")
                for item in value
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
        return ""

    def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "provider": {
                "order": self.provider_order,
                "allow_fallbacks": self.allow_fallbacks,
            },
            "reasoning": {"enabled": self.reasoning_enabled},
        }
        if seed is not None:
            payload["seed"] = seed
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://anonymous.invalid",
                "X-Title": "Anonymous baseline evaluation",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                choices = parsed.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    raise RuntimeError(f"OpenRouter response has no choices: {parsed!r}")
                message = choices[0].get("message") or {}
                text = self._text(message.get("content"))
                usage = parsed.get("usage") or {}
                with_cost = dict(usage)
                if parsed.get("model") is not None:
                    with_cost["response_model"] = parsed["model"]
                if parsed.get("id") is not None:
                    with_cost["response_id"] = parsed["id"]
                self.calls += 1
                self.total_cost += float(usage.get("cost") or 0.0)
                return text, with_cost
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:1000]
                except Exception:
                    detail = "<unavailable>"
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                self.failures += 1
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep)
            except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as exc:
                last_error = exc
                self.failures += 1
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep)
        raise RuntimeError(
            f"OpenRouter request failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error
