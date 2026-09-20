import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "pre_logits_sampled_bedrock.py"
)
SPEC = importlib.util.spec_from_file_location("pre_logits_sampled_bedrock", MODULE_PATH)
bedrock = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bedrock)


def test_normalize_visible_bedrock_response():
    response = bedrock.normalize_bedrock_response(
        {
            "output": {
                "message": {
                    "content": [{"text": " continuation"}],
                }
            },
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 10, "outputTokens": 1, "totalTokens": 11},
            "ResponseMetadata": {"RequestId": "request-1"},
        },
        model_id="deepseek.v3.2",
        max_tokens=1,
        temperature=1.0,
    )

    assert response.content == " continuation"
    assert response.reasoning is None
    assert response.usage["completion_tokens"] == 1
    assert response.generation_id == "request-1"
    assert response.finish_reason == "length"
    assert response.native_finish_reason == "max_tokens"
    assert response.request_max_tokens == 1


def test_normalize_reasoning_uses_sentinel_not_chain_of_thought():
    response = bedrock.normalize_bedrock_response(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "private reasoning"}
                            }
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 8, "totalTokens": 18},
        },
        model_id="us.deepseek.r1-v1:0",
        max_tokens=8,
        temperature=0.0,
    )

    assert response.content == ""
    assert response.reasoning == "<bedrock_reasoning_observed>"
    assert "private reasoning" not in response.reasoning


def test_nova_2_disables_reasoning_explicitly():
    assert bedrock.bedrock_additional_fields("us.amazon.nova-2-lite-v1:0") == {
        "reasoningConfig": {"type": "disabled"}
    }
    assert bedrock.bedrock_additional_fields("amazon.nova-lite-v1:0") is None


def test_native_deepseek_tokenizer_bypasses_model_config(tmp_path):
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text("{}", encoding="utf-8")
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps(
            {
                "bos_token": {
                    "__type": "AddedToken",
                    "content": "<bos>",
                    "normalized": True,
                },
                "eos_token": "<eos>",
                "pad_token": "<eos>",
                "model_max_length": 131072,
            }
        ),
        encoding="utf-8",
    )
    tokenizer = MagicMock()
    tokenizer.init_kwargs = {}

    def download(_repo, filename, revision=None):
        assert revision == "pinned-commit"
        return str(
            tokenizer_json if filename == "tokenizer.json" else tokenizer_config
        )

    with (
        patch("huggingface_hub.hf_hub_download", side_effect=download),
        patch("transformers.PreTrainedTokenizerFast", return_value=tokenizer),
    ):
        loaded = bedrock.load_deepseek_v32_native_tokenizer(
            "deepseek-ai/DeepSeek-V3.2",
            revision="pinned-commit",
        )

    assert loaded is tokenizer
    assert tokenizer.model_max_length == 131072
    assert tokenizer.init_kwargs["_commit_hash"] == "pinned-commit"


def test_native_deepseek_tokenizer_loader_ignores_other_repositories():
    assert bedrock.load_deepseek_v32_native_tokenizer("Qwen/Qwen3-1.7B") is None
