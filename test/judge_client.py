from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class JudgeConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    max_output_tokens: int = 4096
    temperature: float = 0.0
    seed: int = 1234
    gemini_safety_threshold: Optional[str] = "BLOCK_NONE"
    gemini_thinking_budget: Optional[int] = None


def add_judge_args(parser, default_openai_model: str) -> None:
    parser.add_argument(
        "--judge_provider",
        choices=["openai", "gemini"],
        default="openai",
        help="LLM provider used as the harmfulness judge.",
    )
    parser.add_argument(
        "--judge_model",
        default=None,
        help="Judge model name. Defaults to the script's OpenAI model or gemini-3.5-flash.",
    )
    parser.add_argument(
        "--judge_api_key",
        default=None,
        help="Optional judge API key. Prefer OPENAI_API_KEY or GOOGLE_API_KEY environment variables.",
    )
    parser.add_argument("--judge_max_output_tokens", type=int, default=4096)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_seed", type=int, default=1234)
    parser.add_argument(
        "--gemini_safety_threshold",
        default="BLOCK_NONE",
        help="Gemini safety threshold for judge calls. Use an empty string to omit safety settings.",
    )
    parser.add_argument(
        "--gemini_thinking_budget",
        type=int,
        default=None,
        help="Optional Gemini thinking budget; use 0 for deterministic no-thinking judging.",
    )
    parser.set_defaults(default_openai_judge_model=default_openai_model)


def build_judge_config(args) -> JudgeConfig:
    provider = args.judge_provider
    if args.judge_model:
        model = args.judge_model
    elif provider == "gemini":
        model = "gemini-3.5-flash"
    else:
        model = args.default_openai_judge_model

    safety_threshold = args.gemini_safety_threshold or None
    return JudgeConfig(
        provider=provider,
        model=model,
        api_key=args.judge_api_key,
        max_output_tokens=args.judge_max_output_tokens,
        temperature=args.judge_temperature,
        seed=args.judge_seed,
        gemini_safety_threshold=safety_threshold,
        gemini_thinking_budget=args.gemini_thinking_budget,
    )


class JudgeClient:
    def __init__(self, config: JudgeConfig):
        self.config = config
        self.last_response_metadata = None
        if config.provider == "openai":
            from openai import OpenAI

            api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set for --judge_provider openai.")
            self.client = OpenAI(api_key=api_key)
        elif config.provider == "gemini":
            from google import genai
            from google.genai import types

            api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY must be set for --judge_provider gemini.")
            self.client = genai.Client(api_key=api_key)
            self.types = types
        else:
            raise ValueError(f"Unsupported judge provider: {config.provider}")

    def _record_gemini_response(self, response) -> str:
        usage = getattr(response, "usage_metadata", None)
        candidates = getattr(response, "candidates", None) or []
        finish_reason = (
            getattr(candidates[0], "finish_reason", None) if candidates else None
        )
        create_time = getattr(response, "create_time", None)
        self.last_response_metadata = {
            "provider": "gemini",
            "response_id": getattr(response, "response_id", None),
            "response_model": getattr(response, "model_version", None),
            "created": str(create_time) if create_time is not None else None,
            "system_fingerprint": None,
            "finish_reason": str(finish_reason) if finish_reason is not None else None,
            "usage": {
                field: getattr(usage, field)
                for field in (
                    "prompt_token_count",
                    "candidates_token_count",
                    "total_token_count",
                    "thoughts_token_count",
                    "cached_content_token_count",
                )
                if usage is not None and getattr(usage, field, None) is not None
            },
        }
        return getattr(response, "text", None) or ""

    def _gemini_config_kwargs(self) -> dict:
        config_kwargs = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "seed": self.config.seed,
        }
        if self.config.gemini_thinking_budget is not None:
            config_kwargs["thinking_config"] = self.types.ThinkingConfig(
                thinking_budget=self.config.gemini_thinking_budget
            )
        if self.config.gemini_safety_threshold:
            config_kwargs["safety_settings"] = [
                self.types.SafetySetting(
                    category=category,
                    threshold=self.config.gemini_safety_threshold,
                )
                for category in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_CIVIC_INTEGRITY",
                ]
            ]
        return config_kwargs

    def generate(self, prompt: str) -> str:
        self.last_response_metadata = None
        if self.config.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                frequency_penalty=0,
                presence_penalty=0,
                seed=self.config.seed,
            )
            usage = getattr(response, "usage", None)
            self.last_response_metadata = {
                "provider": "openai",
                "response_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
                "created": getattr(response, "created", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "usage": {
                    field: getattr(usage, field)
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if usage is not None and getattr(usage, field, None) is not None
                },
            }
            return response.choices[0].message.content or ""

        response = self.client.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(**self._gemini_config_kwargs()),
        )
        return self._record_gemini_response(response)

    def generate_attachment_score(
        self,
        prompt: str,
        attachment: bytes,
        mime_type: str,
        min_score: int = 1,
        max_score: int = 5,
    ) -> str:
        """Ask Gemini for a schema-constrained integer score over an attachment."""

        if self.config.provider != "gemini":
            raise ValueError("Attachment scoring is only available for Gemini judges.")
        self.last_response_metadata = None
        config_kwargs = self._gemini_config_kwargs()
        config_kwargs.update(
            {
                "response_mime_type": "application/json",
                "response_json_schema": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "integer",
                            "minimum": min_score,
                            "maximum": max_score,
                        }
                    },
                    "required": ["score"],
                    "additionalProperties": False,
                },
            }
        )
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=[
                prompt,
                self.types.Part.from_bytes(data=attachment, mime_type=mime_type),
            ],
            config=self.types.GenerateContentConfig(**config_kwargs),
        )
        text = self._record_gemini_response(response)
        try:
            import json

            score = int(json.loads(text)["score"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return ""
        return f"#thescore: {score}" if min_score <= score <= max_score else ""
