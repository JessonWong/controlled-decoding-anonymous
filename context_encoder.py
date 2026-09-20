"""Frozen, pre-action text features shared by BiasNet cache and inference paths."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import torch


CONTEXT_PROTOCOL = "chat_header_plus_pre_action_prefix_v1"
CONTEXT_MANIFEST = "context_manifest.json"
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_identity(tokenizer) -> dict:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_spec = json.loads(backend.to_str()) if backend is not None else None
    if backend_spec is not None:
        # These are per-call batching settings, not the tokenizer mapping.
        backend_spec.pop("padding", None)
        backend_spec.pop("truncation", None)
    return {
        "vocab_sha256": json_sha256(tokenizer.get_vocab()),
        "backend_sha256": json_sha256(backend_spec),
        "special_tokens_sha256": json_sha256(tokenizer.special_tokens_map),
        "chat_template_sha256": json_sha256(tokenizer.chat_template),
    }


def validate_contract(contract: dict) -> None:
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ValueError("Unsupported context encoder contract.")
    if contract.get("protocol") != CONTEXT_PROTOCOL:
        raise ValueError("Context features must use pre-action prefixes.")
    if contract.get("dtype") not in DTYPES or contract.get("quantization") != "none":
        raise ValueError("Unsupported context encoder precision/quantization.")
    if (contract.get("pooling") != "last_non_padding_final_layer" or contract.get("overflow") != "error"
            or contract.get("position_ids") != "non_padding_cumsum_v1"):
        raise ValueError("Unsupported context pooling or truncation contract.")
    for name in ("hidden_size", "max_length"):
        if not isinstance(contract.get(name), int) or contract[name] <= 0:
            raise ValueError(f"Context contract {name} must be positive.")
    for name in ("model_identity", "tokenizer_identity"):
        if not isinstance(contract.get(name), dict) or not contract[name]:
            raise ValueError(f"Context contract is missing {name}.")
    for name in ("model_name_or_path", "tokenizer_name_or_path"):
        if not isinstance(contract.get(name), str) or not contract[name]:
            raise ValueError(f"Context contract is missing {name}.")


def render_context(tokenizer, prompt: str, prefix: str) -> str:
    """No assistant end marker, candidate token, gold token, or future answer."""
    if not isinstance(prompt, str) or not prompt or not isinstance(prefix, str):
        raise ValueError("Context needs a nonempty prompt and a string pre-action prefix.")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Context encoder requires an explicit chat template.")
    header = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )
    return header + prefix


def cache_states(payload: dict) -> tuple[str, list[str]]:
    """Use exact materialized row prefixes; never reconstruct them from labels."""
    scores, labels = payload.get("log_probs"), payload.get("labels")
    if (not isinstance(scores, torch.Tensor) or scores.ndim != 3 or scores.shape[0] != 1
            or not isinstance(labels, torch.Tensor) or tuple(labels.shape) != tuple(scores.shape[:2])):
        raise ValueError("Context source needs log_probs [1,L,V] and labels [1,L].")
    prompt, prefixes = payload.get("prompt_text"), payload.get("sampled_prefix_texts")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("MC cache lacks prompt_text; recover the original sampler text first.")
    if (not isinstance(prefixes, list) or len(prefixes) != labels.shape[1]
            or not prefixes or not all(isinstance(p, str) for p in prefixes)):
        raise ValueError("MC cache needs one exact sampled_prefix_texts entry per materialized row.")
    return prompt, prefixes


def state_identity(prompt: str, prefixes: list[str]) -> dict:
    return {"prompt_sha256": text_sha256(prompt),
            "prefix_text_sha256": [text_sha256(p) for p in prefixes]}


def states_from_manifest(source: Path, payload: dict, manifest: dict) -> tuple[str, list[str]]:
    """Read explicitly recovered legacy text, bound to the exact source bytes."""
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != CONTEXT_PROTOCOL:
        raise ValueError("Unsupported recovered context state manifest.")
    entry = manifest.get("files", {}).get(source.name, {})
    if entry.get("source_cache_sha256") != file_sha256(source):
        raise ValueError(f"Recovered text source checksum mismatch for {source.name}.")
    return cache_states({**payload, "prompt_text": entry.get("prompt_text"),
                         "sampled_prefix_texts": entry.get("sampled_prefix_texts")})


def load_context_manifest(directory: str | Path) -> dict:
    manifest = json.loads((Path(directory) / CONTEXT_MANIFEST).read_text())
    contract = manifest.get("contract")
    validate_contract(contract)
    if manifest.get("schema_version") != 1 or manifest.get("contract_sha256") != json_sha256(contract):
        raise ValueError("Context manifest contract checksum mismatch.")
    return manifest


def load_context_sidecar(source: str | Path, payload: dict, directory: str | Path,
                         manifest: dict) -> torch.Tensor:
    source = Path(source)
    sidecar_path = Path(directory) / source.name
    entry = manifest.get("files", {}).get(source.name)
    if not isinstance(entry, dict) or not sidecar_path.is_file():
        raise ValueError(f"Context manifest/sidecar missing for {source.name}.")
    if entry.get("sidecar_sha256") != file_sha256(sidecar_path):
        raise ValueError(f"Context sidecar checksum mismatch for {source.name}.")
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    if "prompt_text" in payload or "sampled_prefix_texts" in payload:
        prompt, prefixes = cache_states(payload)
    else:
        recovery = sidecar.get("recovered_state")
        if (not isinstance(recovery, dict) or not manifest.get("state_manifest_sha256")
                or sidecar.get("state_manifest_sha256") != manifest["state_manifest_sha256"]):
            raise ValueError("Legacy cache requires explicitly recovered context states.")
        prompt, prefixes = cache_states({**payload, "prompt_text": recovery.get("prompt_text"),
                                         "sampled_prefix_texts": recovery.get("sampled_prefix_texts")})
    if sidecar.get("schema_version") != 1:
        raise ValueError("Unsupported context sidecar schema.")
    for key, value in state_identity(prompt, prefixes).items():
        if sidecar.get(key) != value:
            raise ValueError(f"Context row alignment mismatch: {key} in {source.name}.")
    source_hash = file_sha256(source)
    if sidecar.get("source_cache_sha256") != source_hash or entry.get("source_cache_sha256") != source_hash:
        raise ValueError(f"MC source cache changed for {source.name}.")
    if sidecar.get("context_contract_sha256") != manifest["contract_sha256"]:
        raise ValueError("Sidecar context encoder contract mismatch.")
    if sidecar.get("source_row_indices") != list(range(len(prefixes))):
        raise ValueError("Context row indices differ from materialized MC row order.")
    features = sidecar.get("context_features")
    if (not isinstance(features, torch.Tensor)
            or tuple(features.shape) != (len(prefixes), manifest["contract"]["hidden_size"])
            or features.dtype != torch.float16 or not torch.isfinite(features).all()):
        raise ValueError("Invalid context feature shape, storage dtype, or values.")
    return features


def model_identity(config, source: str) -> dict:
    config_dict = config.to_dict()
    for key in ("_name_or_path", "_commit_hash", "transformers_version", "torch_dtype", "dtype"):
        config_dict.pop(key, None)
    path = Path(source).expanduser().resolve()
    revision = getattr(config, "_commit_hash", None)
    if path.is_dir():
        if path.parent.name == "snapshots" and re.fullmatch(r"[a-f0-9]{40}", path.name):
            revision = path.name
        else:
            weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
            if not weights:
                raise ValueError("Local context model needs safetensors or pytorch_model weights.")
            return {"config_sha256": json_sha256(config_dict),
                    "weights_sha256": json_sha256({p.name: file_sha256(p) for p in weights})}
    if not revision:
        raise ValueError("Cannot resolve an immutable context model revision.")
    return {"config_sha256": json_sha256(config_dict), "revision": revision}


class FrozenContextEncoder:
    def __init__(self, backbone, tokenizer, contract: dict, device):
        validate_contract(contract)
        self.backbone, self.tokenizer = backbone, tokenizer
        self.contract, self.device = contract, torch.device(device)
        self.calls = 0
        self.input_tokens = 0
        self.latency_seconds = 0.0
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Context tokenizer needs a padding or EOS token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def load(cls, model_name_or_path: str, *, tokenizer_name_or_path=None, revision=None,
             tokenizer_revision=None, dtype="float16", max_length=1024, device="cuda:0",
             local_files_only=False, expected_contract=None):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if dtype not in DTYPES or max_length <= 0:
            raise ValueError("Invalid context dtype or max_length.")
        tokenizer_source = tokenizer_name_or_path or model_name_or_path
        config = AutoConfig.from_pretrained(model_name_or_path, revision=revision,
                                            local_files_only=local_files_only)
        if getattr(config, "quantization_config", None) is not None:
            raise ValueError("Context encoder contract requires unquantized model weights.")
        model_limit = getattr(config, "max_position_embeddings", None)
        if model_limit is not None and max_length > model_limit:
            raise ValueError("Context max_length exceeds encoder max_position_embeddings.")
        identity = model_identity(config, model_name_or_path)
        resolved_revision = identity.get("revision", revision)
        resolved_tokenizer_revision = tokenizer_revision or (
            resolved_revision if tokenizer_source == model_name_or_path else None
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, revision=resolved_tokenizer_revision,
            use_fast=True, local_files_only=local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        contract = {
            "schema_version": 1, "protocol": CONTEXT_PROTOCOL,
            "model_name_or_path": str(model_name_or_path), "model_revision": resolved_revision,
            "tokenizer_name_or_path": str(tokenizer_source),
            "tokenizer_revision": resolved_tokenizer_revision,
            "model_identity": identity, "tokenizer_identity": tokenizer_identity(tokenizer),
            "hidden_size": int(config.hidden_size), "dtype": dtype, "quantization": "none",
            "max_length": int(max_length), "overflow": "error",
            "pooling": "last_non_padding_final_layer",
            "position_ids": "non_padding_cumsum_v1",
        }
        if expected_contract is not None:
            validate_contract(expected_contract)
            portable_fields = set(contract) - {"model_name_or_path", "model_revision",
                                                "tokenizer_name_or_path", "tokenizer_revision"}
            differences = [k for k in portable_fields if contract[k] != expected_contract.get(k)]
            if differences:
                raise ValueError(f"Context encoder differs from training contract: {sorted(differences)}.")
            contract = expected_contract
        backbone = AutoModel.from_pretrained(
            model_name_or_path, revision=resolved_revision, config=config,
            torch_dtype=DTYPES[dtype], local_files_only=local_files_only,
        ).to(device)
        return cls(backbone, tokenizer, contract, device)

    @classmethod
    def from_contract(cls, contract: dict, *, device="cuda:0", model_override=None,
                      tokenizer_override=None, local_files_only=False):
        validate_contract(contract)
        return cls.load(
            model_override or contract["model_name_or_path"],
            tokenizer_name_or_path=tokenizer_override or contract["tokenizer_name_or_path"],
            revision=contract.get("model_revision"), tokenizer_revision=contract.get("tokenizer_revision"),
            dtype=contract["dtype"], max_length=contract["max_length"], device=device,
            local_files_only=local_files_only, expected_contract=contract,
        )

    @torch.no_grad()
    def encode(self, prompts: list[str], prefixes: list[str]) -> torch.Tensor:
        if len(prompts) != len(prefixes) or not prompts:
            raise ValueError("Context encoder needs equal nonempty prompt/prefix batches.")
        texts = [render_context(self.tokenizer, p, y) for p, y in zip(prompts, prefixes)]
        encoded = self.tokenizer(texts, padding=True, truncation=False,
                                 add_special_tokens=False, return_tensors="pt")
        mask = encoded["attention_mask"]
        if (mask.sum(-1) == 0).any() or mask.shape[1] > self.contract["max_length"]:
            raise ValueError("Context is empty or exceeds max_length; silent truncation is disabled.")
        encoded = {k: v.to(self.device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
        # Keep absolute/rotary positions independent of other rows' padding lengths.
        encoded["position_ids"] = (encoded["attention_mask"].long().cumsum(-1) - 1).clamp_min(0)
        self.backbone.eval()
        started = time.perf_counter()
        outputs = self.backbone(**encoded, use_cache=False)
        positions = torch.arange(mask.shape[1], device=self.device)[None, :]
        last = (encoded["attention_mask"].long() * positions).max(-1).values
        hidden = outputs.last_hidden_state[torch.arange(len(prompts), device=self.device), last]
        # Apply the same FP16 storage rounding online as in the offline sidecar.
        result = hidden.detach().to(dtype=torch.float16).contiguous()
        if result.shape != (len(prompts), self.contract["hidden_size"]) or not torch.isfinite(result).all():
            raise ValueError("Context encoder returned invalid hidden features.")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.latency_seconds += time.perf_counter() - started
        self.calls += 1
        self.input_tokens += int(mask.sum())
        return result
