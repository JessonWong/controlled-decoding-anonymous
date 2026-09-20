"""Derive a Monte Carlo cache from an existing exact log-probability cache.

The MC estimator draws K samples from the model's next-token distribution and
smooths the counts. An exact cache already stores that distribution, so the
samples can be drawn from it directly instead of re-running the target model.
This is the same estimator ``pre_logits_exact_openweight.py`` applies to raw
logits -- ``softmax(logits) == exp(log_probs)`` -- so the result is
distributionally identical while costing no GPU time.

Risk-gate fields and labels are copied verbatim, which keeps the derived cache a
drop-in twin of its source for a matched exact-vs-MC comparison.
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.pre_logits_exact_openweight import sampled_ids_to_log_probs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Directory of exact .pt caches.")
    parser.add_argument("--output_dir", required=True, help="Directory for the derived MC cache.")
    parser.add_argument("--samples_per_token", type=int, default=50)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--observed_alpha", type=float, default=0.1)
    parser.add_argument("--floor_mass", type=float, default=1e-4)
    parser.add_argument("--store_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--counts_dtype",
        choices=["int16", "int32"],
        default="int32",
        help="Dtype for the dense mc_counts tensor. int16 halves the on-disk size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_temperature != 1.0 or args.top_p != 1.0:
        raise ValueError(
            "Only temperature=1 / top_p=1 can be derived from a stored distribution; "
            "other settings need the original logits."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    store_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32
    counts_dtype = torch.int16 if args.counts_dtype == "int16" else torch.int32

    names = sorted(name for name in os.listdir(args.input_dir) if name.endswith(".pt"))
    if not names:
        raise ValueError(f"No .pt caches found in {args.input_dir}")

    converted = 0
    total_tokens = 0
    for name in tqdm(names, desc="Deriving MC caches"):
        out_path = os.path.join(args.output_dir, name)
        if os.path.exists(out_path):
            continue
        payload = torch.load(os.path.join(args.input_dir, name), map_location="cpu", weights_only=False)
        log_probs = payload["log_probs"][0].float()
        labels = payload["labels"]
        vocab_size = log_probs.shape[1]

        probs = log_probs.exp()
        # Guard against fp16 storage drift before sampling.
        probs = probs.clamp_min(0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(
            probs,
            num_samples=args.samples_per_token,
            replacement=True,
            generator=generator,
        )
        mc_log_probs, mc_counts = sampled_ids_to_log_probs(
            sampled_token_ids=sampled,
            vocab_size=vocab_size,
            observed_alpha=args.observed_alpha,
            floor_mass=args.floor_mass,
            dtype=store_dtype,
        )

        metadata = dict(payload.get("metadata") or {})
        metadata.update(
            {
                "source": "mc_derived_from_exact",
                "derived_from": args.input_dir,
                "samples_per_token": args.samples_per_token,
                "sample_temperature": args.sample_temperature,
                "top_p": args.top_p,
                "observed_alpha": args.observed_alpha,
                "floor_mass": args.floor_mass,
                "sample_completion_policy": "exact",
            }
        )
        result = {
            "log_probs": mc_log_probs.unsqueeze(0),
            "labels": labels,
            "mc_counts": mc_counts.to(counts_dtype).unsqueeze(0),
            "valid_sample_counts": torch.full_like(labels, args.samples_per_token),
            "metadata": metadata,
        }
        for key in ("risk_gate_mask", "risk_gate_scores", "risk_gate_token_ids"):
            if key in payload:
                result[key] = payload[key]
        torch.save(result, out_path)
        converted += 1
        total_tokens += labels.numel()

    manifest = {
        "configuration": {
            "source": "mc_derived_from_exact",
            "input_dir": args.input_dir,
            "samples_per_token": args.samples_per_token,
            "observed_alpha": args.observed_alpha,
            "floor_mass": args.floor_mass,
            "store_dtype": args.store_dtype,
            "seed": args.seed,
        },
        "manifest_schema_version": 1,
    }
    with open(os.path.join(args.output_dir, "cache_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"converted={converted} tokens={total_tokens} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
