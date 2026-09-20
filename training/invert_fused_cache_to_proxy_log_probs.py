"""Recover the prior q from an already-built fused cache, for (T, kappa) recalibration.

The Dirichlet fusion is invertible given the counts it was built from:

    p_i = (c_i + kappa*q_i) / (N + kappa)   =>   q_i = (p_i*(N + kappa) - c_i) / kappa

so a sweep does not need a second pass over the proxy model.  The recovered q is
the TEMPERED prior, i.e. ``log_softmax(scores / T_build)``, which is exactly what
``eval_proxy_mc_fusion`` consumes when told ``proxy_log_probs_temperature=T_build``:
it rescales by that factor before applying each grid temperature, so the original
tempering is undone rather than compounded.

Accuracy is bounded by the fp16 storage of the fused cache.  Measured on a Gemma
record: row sums land in [0.9975, 1.0046] and a handful of OBSERVED coordinates per
row (<=4 of 151669) go slightly negative from cancellation; those are clamped.  This
is a fast preview of a grid whose authoritative version recomputes q from the proxy.
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

FLOOR = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fused_dir", required=True)
    p.add_argument("--mc_dir", required=True, help="Counts the fused cache was built from.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prior_strength", type=float, required=True, help="kappa used at build time.")
    p.add_argument("--build_temperature", type=float, required=True, help="T used at build time.")
    p.add_argument("--store_dtype", choices=["float16", "float32"], default="float16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    store_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32
    names = sorted(n for n in os.listdir(args.fused_dir) if n.endswith(".pt"))
    stats = {"records": 0, "rows": 0, "clamped": 0, "sum_min": 9e9, "sum_max": -9e9}
    for name in tqdm(names, desc="Inverting fused -> prior"):
        out_path = os.path.join(args.output_dir, name)
        if os.path.exists(out_path):
            continue
        fused = torch.load(os.path.join(args.fused_dir, name), map_location="cpu", weights_only=False)
        mc = torch.load(os.path.join(args.mc_dir, name), map_location="cpu", weights_only=False)
        counts = mc["mc_counts"][0].to(torch.float64)
        p = fused["log_probs"][0].to(torch.float64).exp()
        totals = counts.sum(dim=-1, keepdim=True)
        q = (p * (totals + args.prior_strength) - counts) / args.prior_strength
        sums = q.sum(dim=-1)
        stats["sum_min"] = min(stats["sum_min"], float(sums.min()))
        stats["sum_max"] = max(stats["sum_max"], float(sums.max()))
        stats["clamped"] += int((q < FLOOR).sum())
        q = q.clamp_min(FLOOR)
        log_q = q.log()
        log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
        torch.save(
            {
                "proxy_log_probs": log_q.to(store_dtype).unsqueeze(0),
                "metadata": {
                    "proxy_log_probs_semantics": "normalised_proxy_log_probs",
                    "proxy_log_probs_temperature": float(args.build_temperature),
                    "recovered_from": "fused_cache_inversion_v1",
                    "build_prior_strength": float(args.prior_strength),
                },
            },
            out_path,
        )
        stats["records"] += 1
        stats["rows"] += int(log_q.shape[0])
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
