"""Fuse a CROSS-FAMILY proxy into a local MC cache, in the target's own vocabulary.

``training/fuse_proxy_into_local_cache.py`` requires the proxy to share the
target's tokenizer, so the Qwen line has only ever used a Qwen proxy.  That makes
"the proxy is a generic density smoother, not a target prior" untestable: family
and vocabulary are confounded.

This script removes the confound while changing nothing else.  The cache, the
coordinate system (151669 Qwen ids), the BiasNet ``lm_head`` initialisation, the
training recipe and the decoder all stay byte-identical to the same-family arm.
The ONLY variable is which model supplies the dense prior ``q``, obtained via the
first-token projection in :mod:`foreign_proxy`.

Set ``--proxy_model_name_or_path`` to a same-family model to get the projection
CONTROL arm, which isolates any cost of the projection itself from the effect of
changing families.
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from foreign_proxy import (  # noqa: E402
    ForeignProxyRuntime,
    load_or_build_first_token_map,
    tokenizer_fingerprint,
)
from mc_reconstruction import fuse_proxy_logits_with_mc_counts  # noqa: E402

# Must match train_biasnet.PROXY_MC_FUSION_MODE.
FUSION_MODE = "proxy_dirichlet_v1"
# Deliberately distinct from messages_for_prefix_qwen_v1 so a foreign-proxy cache
# can never be mistaken for a same-family one in a checkpoint audit.
CHAT_TEMPLATE_PROTOCOL = "messages_for_prefix_foreign_proxy_v1"
PROXY_VOCAB_TAIL_POLICY = "first_token_projection_to_target_vocab"
QWEN_HARD_NO_THINK_PREFILL = "<think>\n\n</think>\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc_dir", required=True, help="MC cache providing mc_counts.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--proxy_model_name_or_path", required=True)
    parser.add_argument("--proxy_tokenizer_name_or_path", default=None)
    parser.add_argument(
        "--target_tokenizer_name_or_path",
        required=True,
        help="Tokenizer defining the cache's coordinate system (decodes labels).",
    )
    parser.add_argument("--first_token_map", default=None, help="Cache path for the projection map.")
    parser.add_argument("--dataset_name", default="LLM-LAT/harmful-dataset")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--proxy_temperature", type=float, default=2.0)
    parser.add_argument("--prior_strength", type=float, default=8.0)
    parser.add_argument("--shared_vocab_size", type=int, default=151669)
    parser.add_argument("--floor_logit", type=float, default=-30.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--store_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--torch_dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--append_no_think", action="store_true")
    parser.add_argument("--qwen_hard_no_think_prefill", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--proxy_model_revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.proxy_temperature <= 0 or args.prior_strength <= 0:
        raise ValueError("--proxy_temperature and --prior_strength must be positive.")
    os.makedirs(args.output_dir, exist_ok=True)
    store_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]

    target_tokenizer = AutoTokenizer.from_pretrained(
        args.target_tokenizer_name_or_path, local_files_only=args.local_files_only
    )
    proxy_tokenizer = AutoTokenizer.from_pretrained(
        args.proxy_tokenizer_name_or_path or args.proxy_model_name_or_path,
        local_files_only=args.local_files_only,
    )
    proxy = AutoModelForCausalLM.from_pretrained(
        args.proxy_model_name_or_path,
        dtype=torch_dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    proxy.eval()
    device = next(proxy.parameters()).device

    first_token_ids = load_or_build_first_token_map(
        args.first_token_map,
        target_tokenizer,
        proxy_tokenizer,
        args.shared_vocab_size,
    )
    mapped = int((first_token_ids >= 0).sum().item())
    distinct = int(torch.unique(first_token_ids[first_token_ids >= 0]).numel())

    runtime = ForeignProxyRuntime(
        proxy, proxy_tokenizer, first_token_ids, device, floor_logit=args.floor_logit
    )

    dataset = load_dataset(args.dataset_name)[args.dataset_split]
    names = sorted(name for name in os.listdir(args.mc_dir) if name.endswith(".pt"))
    if not names:
        raise ValueError(f"No .pt caches found in {args.mc_dir}")

    proxy_config_fields = {
        "mc_fusion_mode": FUSION_MODE,
        "proxy_model_name_or_path": args.proxy_model_name_or_path,
        "proxy_model_revision": args.proxy_model_revision or "local_snapshot",
        "proxy_tokenizer_sha256": tokenizer_fingerprint(proxy_tokenizer),
        "proxy_temperature": float(args.proxy_temperature),
        "proxy_prior_strength": float(args.prior_strength),
        "proxy_chat_template_protocol": CHAT_TEMPLATE_PROTOCOL,
        "shared_vocab_size": int(args.shared_vocab_size),
        "proxy_vocab_size": int(proxy.get_output_embeddings().weight.shape[0]),
        "proxy_vocab_tail_policy": PROXY_VOCAB_TAIL_POLICY,
        "proxy_dtype": args.torch_dtype,
        "proxy_quantization": "none",
        # Foreign-projection provenance, so the arm is auditable from the cache.
        "foreign_proxy_projection": "first_token_v1",
        "target_tokenizer_name_or_path": args.target_tokenizer_name_or_path,
        "target_tokenizer_sha256": tokenizer_fingerprint(target_tokenizer),
        "projection_mapped_coordinates": mapped,
        "projection_distinct_proxy_ids": distinct,
        "projection_floor_logit": float(args.floor_logit),
    }

    converted = 0
    rows = 0
    for name in tqdm(names, desc="Fusing foreign proxy + MC"):
        out_path = os.path.join(args.output_dir, name)
        if os.path.exists(out_path):
            continue
        payload = torch.load(os.path.join(args.mc_dir, name), map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata") or {})
        index = metadata.get("dataset_index")
        if index is None:
            raise ValueError(f"{name} has no dataset_index; cannot rebuild the proxy prefix.")
        question = dataset[int(index)]["prompt"]
        if args.append_no_think:
            question = question + "\n/no_think"

        labels = payload["labels"]
        mc_counts = payload["mc_counts"][0].to(torch.int64)
        answer_ids = labels[0].tolist()

        # Position i is conditioned on the answer text produced BEFORE it, decoded
        # from the target's ids and re-encoded by the proxy.  One independent
        # forward per position, so no cross-tokenizer prefix alignment is assumed.
        lead = QWEN_HARD_NO_THINK_PREFILL if args.qwen_hard_no_think_prefill else ""
        answer_prefixes = [
            lead + target_tokenizer.decode(answer_ids[:i], skip_special_tokens=False)
            for i in range(len(answer_ids))
        ]
        proxy_log_probs = runtime.teacher_forced_log_probs(
            question, answer_prefixes, batch_size=args.batch_size
        )
        if proxy_log_probs.shape[0] != len(answer_ids):
            raise ValueError(f"{name}: projected proxy rows misaligned with labels.")

        fused = fuse_proxy_logits_with_mc_counts(
            proxy_log_probs.to(device).float(),
            mc_counts.to(device),
            temperature=args.proxy_temperature,
            prior_strength=args.prior_strength,
        )
        if fused.shape[1] != args.shared_vocab_size:
            raise ValueError(
                f"{name}: fused vocabulary {fused.shape[1]} != {args.shared_vocab_size}"
            )

        metadata.update(
            {
                "source": "foreign_proxy_fused_local",
                "log_probs_semantics": "proxy_dirichlet_posterior_predictive_log_probs",
                "mc_counts_key": "mc_counts",
                **proxy_config_fields,
            }
        )
        result = {
            "log_probs": fused.to(store_dtype).unsqueeze(0).cpu(),
            "labels": labels,
            "mc_counts": payload["mc_counts"],
            "valid_sample_counts": payload["valid_sample_counts"],
            "metadata": metadata,
        }
        for key in ("risk_gate_mask", "risk_gate_scores", "risk_gate_token_ids"):
            if key in payload:
                result[key] = payload[key]
        torch.save(result, out_path)
        converted += 1
        rows += labels.numel()

    manifest = {
        "configuration": {
            "source": "foreign_proxy_fused_local",
            "mc_dir": args.mc_dir,
            "append_no_think": bool(args.append_no_think),
            "qwen_hard_no_think_prefill": bool(args.qwen_hard_no_think_prefill),
            "batch_size": int(args.batch_size),
            **proxy_config_fields,
        },
        "records_written": converted,
        "records_total": len(names),
        "rows_written": rows,
    }
    with open(os.path.join(args.output_dir, "foreign_proxy_mc_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
