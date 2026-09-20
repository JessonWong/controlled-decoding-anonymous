"""Build a proxy+MC fused cache from a locally produced exact/MC cache pair.

``training/materialize_proxy_mc_cache.py`` targets the OpenRouter cache schema
and validates provider/model provenance that a locally produced cache does not
carry. This script performs the same Dirichlet fusion -- it calls the very same
``fuse_proxy_logits_with_mc_counts`` -- against the local cache layout written by
``training/pre_logits_exact_openweight.py`` / ``training/mc50_cache_from_exact.py``.

Dense proxy logits restore the coordinates that a 50-sample Monte Carlo estimate
collapses onto a uniform floor, which is the information BiasNet's pseudo-inverse
input mapping needs.
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

from mc_reconstruction import fuse_proxy_logits_with_mc_counts  # noqa: E402
from training.materialize_proxy_mc_cache import tokenizer_mapping_sha256  # noqa: E402
from training.pre_logits_exact_openweight import build_assistant_prefix  # noqa: E402

# Must match train_biasnet.PROXY_MC_FUSION_MODE / materialize_proxy_mc_cache.FUSION_MODE.
FUSION_MODE = "proxy_dirichlet_v1"
CHAT_TEMPLATE_PROTOCOL = "messages_for_prefix_qwen_v1"
PROXY_VOCAB_TAIL_POLICY = "crop_to_shared_vocab_before_softmax"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc_dir", required=True, help="MC cache providing mc_counts.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--proxy_model_name_or_path", required=True)
    parser.add_argument("--proxy_tokenizer_name_or_path", default=None)
    parser.add_argument("--dataset_name", default="LLM-LAT/harmful-dataset")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--proxy_temperature", type=float, default=2.0)
    parser.add_argument("--prior_strength", type=float, default=8.0)
    parser.add_argument("--shared_vocab_size", type=int, default=151669)
    parser.add_argument("--store_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--torch_dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
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

    tokenizer = AutoTokenizer.from_pretrained(
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

    dataset = load_dataset(args.dataset_name)[args.dataset_split]
    names = sorted(name for name in os.listdir(args.mc_dir) if name.endswith(".pt"))
    if not names:
        raise ValueError(f"No .pt caches found in {args.mc_dir}")

    proxy_config_fields = {
        "mc_fusion_mode": FUSION_MODE,
        "proxy_model_name_or_path": args.proxy_model_name_or_path,
        "proxy_model_revision": args.proxy_model_revision or "local_snapshot",
        "proxy_tokenizer_sha256": tokenizer_mapping_sha256(tokenizer),
        "proxy_temperature": float(args.proxy_temperature),
        "proxy_prior_strength": float(args.prior_strength),
        "proxy_chat_template_protocol": CHAT_TEMPLATE_PROTOCOL,
        "shared_vocab_size": int(args.shared_vocab_size),
        "proxy_vocab_size": int(proxy.get_output_embeddings().weight.shape[0]),
        "proxy_vocab_tail_policy": PROXY_VOCAB_TAIL_POLICY,
        "proxy_dtype": args.torch_dtype,
        "proxy_quantization": "none",
    }

    converted = 0
    rows = 0
    for name in tqdm(names, desc="Fusing proxy + MC"):
        out_path = os.path.join(args.output_dir, name)
        if os.path.exists(out_path):
            continue
        payload = torch.load(os.path.join(args.mc_dir, name), map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata") or {})
        index = metadata.get("dataset_index")
        if index is None:
            raise ValueError(f"{name} has no dataset_index; cannot rebuild the proxy prefix.")
        item = dataset[int(index)]
        question = item["prompt"]
        if args.append_no_think:
            question = question + "\n/no_think"

        labels = payload["labels"]
        mc_counts = payload["mc_counts"][0].to(torch.int64)
        answer_ids = labels[0].tolist()

        prefix_text = build_assistant_prefix(tokenizer, question, args.qwen_hard_no_think_prefill)
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        input_ids = torch.tensor([list(prefix_ids) + answer_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = proxy(input_ids=input_ids)
        proxy_logits = out.logits[0, len(prefix_ids) - 1 : input_ids.shape[1] - 1, :].float()
        if proxy_logits.shape[0] != len(answer_ids):
            raise ValueError(f"{name}: proxy logits misaligned with labels.")

        # The helper crops any padded proxy tail down to the shared vocabulary.
        fused = fuse_proxy_logits_with_mc_counts(
            proxy_logits,
            mc_counts.to(proxy_logits.device),
            temperature=args.proxy_temperature,
            prior_strength=args.prior_strength,
        )
        if fused.shape[1] != args.shared_vocab_size:
            raise ValueError(
                f"{name}: fused vocabulary {fused.shape[1]} != {args.shared_vocab_size}"
            )

        metadata.update(
            {
                "source": "proxy_fused_local",
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
            "source": "proxy_fused_local",
            "mc_dir": args.mc_dir,
            "proxy_model_name_or_path": args.proxy_model_name_or_path,
            "proxy_temperature": args.proxy_temperature,
            "prior_strength": args.prior_strength,
            "shared_vocab_size": args.shared_vocab_size,
            "mc_fusion_mode": FUSION_MODE,
        },
        "summary": {"record_count": converted, "row_count": rows},
        "manifest_schema_version": 1,
    }
    with open(os.path.join(args.output_dir, "proxy_mc_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"fused={converted} rows={rows} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
