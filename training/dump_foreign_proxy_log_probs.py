"""Dump projected cross-family proxy log-probs so the standard (T, kappa) grid can run.

``training/eval_proxy_mc_fusion.py`` recalibrates from filename-matched
``proxy_log_probs`` tensors.  The foreign fuser only writes the FUSED posterior,
which cannot be inverted, so the raw projected prior has to be materialised
separately before a sweep.  Written at temperature 1.0 (a plain log-softmax over
the proxy vocabulary, gathered through the first-token map), which is exactly what
``fuse_proxy_logits_with_mc_counts`` consumes, so the sweep and the fuser agree.
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

from foreign_proxy import ForeignProxyRuntime, load_or_build_first_token_map  # noqa: E402

QWEN_HARD_NO_THINK_PREFILL = "<think>\n\n</think>\n\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mc_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--proxy_model_name_or_path", required=True)
    p.add_argument("--proxy_tokenizer_name_or_path", default=None)
    p.add_argument("--target_tokenizer_name_or_path", required=True)
    p.add_argument("--first_token_map", default=None)
    p.add_argument("--dataset_name", default="LLM-LAT/harmful-dataset")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--shared_vocab_size", type=int, default=151669)
    p.add_argument("--floor_logit", type=float, default=-30.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--store_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--torch_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--append_no_think", action="store_true")
    p.add_argument("--qwen_hard_no_think_prefill", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    store_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.torch_dtype
    ]
    target_tok = AutoTokenizer.from_pretrained(
        args.target_tokenizer_name_or_path, local_files_only=args.local_files_only
    )
    proxy_tok = AutoTokenizer.from_pretrained(
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
        args.first_token_map, target_tok, proxy_tok, args.shared_vocab_size
    )
    runtime = ForeignProxyRuntime(
        proxy, proxy_tok, first_token_ids, device, floor_logit=args.floor_logit
    )
    dataset = load_dataset(args.dataset_name)[args.dataset_split]
    names = sorted(n for n in os.listdir(args.mc_dir) if n.endswith(".pt"))
    written = 0
    for name in tqdm(names, desc="Dumping projected proxy log-probs"):
        out_path = os.path.join(args.output_dir, name)
        if os.path.exists(out_path):
            continue
        payload = torch.load(os.path.join(args.mc_dir, name), map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata") or {})
        index = metadata.get("dataset_index")
        if index is None:
            raise ValueError(f"{name} has no dataset_index.")
        question = dataset[int(index)]["prompt"]
        if args.append_no_think:
            question = question + "\n/no_think"
        answer_ids = payload["labels"][0].tolist()
        lead = QWEN_HARD_NO_THINK_PREFILL if args.qwen_hard_no_think_prefill else ""
        prefixes = [
            lead + target_tok.decode(answer_ids[:i], skip_special_tokens=False)
            for i in range(len(answer_ids))
        ]
        rows = runtime.teacher_forced_log_probs(question, prefixes, batch_size=args.batch_size)
        torch.save(
            {
                "proxy_log_probs": rows.to(store_dtype).unsqueeze(0),
                "metadata": {
                    "proxy_log_probs_semantics": "normalised_proxy_log_probs",
                    "proxy_log_probs_temperature": 1.0,
                    "foreign_proxy_projection": "first_token_v1",
                    "proxy_model_name_or_path": args.proxy_model_name_or_path,
                    "shared_vocab_size": int(args.shared_vocab_size),
                },
            },
            out_path,
        )
        written += 1
    print(json.dumps({"records_written": written, "records_total": len(names)}), flush=True)


if __name__ == "__main__":
    main()
