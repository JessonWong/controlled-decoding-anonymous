"""Train a K-conditioned BiasNet on nested partial-MC replay paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mc_reconstruction import fuse_proxy_logits_with_mc_counts
from modeling_biasnet import (
    BiasNet,
    MC_SAMPLE_COUNT_CONDITIONING_AFFINE,
    MC_SAMPLE_COUNT_CONDITIONING_EXACT_ANCHOR,
    MC_SAMPLE_COUNT_CONDITIONING_MODES,
)
from training.anytime_proxy_mc import (
    AnytimeProxyMCCache,
    load_anytime_cache,
    parse_budgets,
    permute_events,
)


ANYTIME_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an anytime BiasNet from an exact MC50 proxy cache."
    )
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--budgets", default="0,4,8,16,32,50")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--top1_margin", type=float, default=1.0)
    parser.add_argument("--base_wrong_weight", type=float, default=1.0)
    parser.add_argument(
        "--target_mode",
        choices=("teacher_label", "reference_mc50_action"),
        default="teacher_label",
    )
    parser.add_argument("--reference_checkpoint", default=None)
    parser.add_argument(
        "--reference_changed_weight",
        type=float,
        default=3.0,
        help="Weight for rows where reference MC50 changes the deterministic base.",
    )
    parser.add_argument(
        "--full_budget_anchor_weight",
        type=float,
        default=0.0,
        help="Paired K50 reference-action anchor weight for every partial-K batch.",
    )
    parser.add_argument(
        "--freeze_except_mc_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train only the selected scalar-K conditioning parameters.",
    )
    parser.add_argument(
        "--mc_sample_count_conditioning_mode",
        choices=MC_SAMPLE_COUNT_CONDITIONING_MODES,
        default=MC_SAMPLE_COUNT_CONDITIONING_AFFINE,
        help=(
            "The legacy affine mode preserves existing experiments. The "
            "exact-anchor vector is multiplied by one minus the normalized "
            "log sample count and is identically zero at the maximum budget."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay_seed", type=int, default=20260823)
    parser.add_argument(
        "--test_files_from_init_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve the init checkpoint's record-level held-out files.",
    )
    parser.add_argument(
        "--max_train_rows",
        type=int,
        default=None,
        help="Debug-only cap applied after the record split.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning_rate must be finite and positive.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight_decay must be finite and non-negative.")
    if not math.isfinite(args.top1_margin) or args.top1_margin < 0:
        raise ValueError("--top1_margin must be finite and non-negative.")
    if not math.isfinite(args.base_wrong_weight) or args.base_wrong_weight < 1:
        raise ValueError("--base_wrong_weight must be finite and at least one.")
    if (
        not math.isfinite(args.reference_changed_weight)
        or args.reference_changed_weight < 1
    ):
        raise ValueError("--reference_changed_weight must be at least one.")
    if (
        not math.isfinite(args.full_budget_anchor_weight)
        or args.full_budget_anchor_weight < 0
    ):
        raise ValueError("--full_budget_anchor_weight must be non-negative.")
    if args.target_mode == "reference_mc50_action" and not args.reference_checkpoint:
        raise ValueError(
            "reference_mc50_action requires --reference_checkpoint."
        )
    if args.target_mode != "reference_mc50_action" and args.reference_checkpoint:
        raise ValueError(
            "--reference_checkpoint is only valid for reference_mc50_action."
        )
    if (
        args.mc_sample_count_conditioning_mode
        == MC_SAMPLE_COUNT_CONDITIONING_EXACT_ANCHOR
        and not args.freeze_except_mc_conditioning
    ):
        raise ValueError(
            "exact_anchor_count_vector_v1 requires "
            "--freeze_except_mc_conditioning so the Kmax controller remains "
            "identical to the initialization checkpoint."
        )
    if args.max_train_rows is not None and args.max_train_rows <= 0:
        raise ValueError("--max_train_rows must be positive when provided.")


def _test_files_from_checkpoint(checkpoint: Path) -> list[str]:
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    held_out = config.get("held_out_files")
    if not isinstance(held_out, list) or not held_out:
        raise ValueError(
            "The init checkpoint does not declare non-empty held_out_files."
        )
    if not all(isinstance(name, str) and name.endswith(".pt") for name in held_out):
        raise ValueError("The init checkpoint has invalid held_out_files metadata.")
    return sorted(held_out)


def _row_indices_for_records(
    cache: AnytimeProxyMCCache, record_names: Sequence[str]
) -> torch.Tensor:
    wanted = set(record_names)
    record_ids = [
        index for index, name in enumerate(cache.record_names) if name in wanted
    ]
    if len(record_ids) != len(wanted):
        missing = sorted(wanted - set(cache.record_names))
        raise ValueError(f"Requested split records are absent from the cache: {missing}")
    mask = torch.zeros(cache.num_rows, dtype=torch.bool)
    for record_id in record_ids:
        mask |= cache.record_ids.eq(record_id)
    return torch.nonzero(mask, as_tuple=False).flatten()


def _mixed_prefix_counts(
    permuted_events: torch.Tensor,
    budgets: torch.Tensor,
    *,
    vocab_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if permuted_events.shape[0] != budgets.numel():
        raise ValueError("budgets must match the event batch.")
    counts = torch.zeros(
        (permuted_events.shape[0], vocab_size),
        device=permuted_events.device,
        dtype=dtype,
    )
    for budget in torch.unique(budgets).tolist():
        budget = int(budget)
        if budget == 0:
            continue
        row_mask = budgets.eq(budget)
        selected = permuted_events[row_mask, :budget]
        increments = torch.ones_like(selected, dtype=dtype)
        # Boolean advanced indexing returns a copy, so build the selected rows
        # separately and explicitly write them back.
        updated = torch.zeros(
            (selected.shape[0], vocab_size),
            device=permuted_events.device,
            dtype=dtype,
        )
        updated.scatter_add_(1, selected, increments)
        counts[row_mask] = updated
    return counts


def _top1_margin_loss(
    outputs: torch.Tensor, labels: torch.Tensor, margin: float
) -> torch.Tensor:
    top_scores, top_ids = torch.topk(outputs, k=2, dim=-1)
    best_other = torch.where(
        top_ids[:, 0].eq(labels), top_scores[:, 1], top_scores[:, 0]
    )
    target = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
    return F.relu(float(margin) - target + best_other)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _compute_reference_actions(
    *,
    cache: AnytimeProxyMCCache,
    row_indices: torch.Tensor,
    model: BiasNet,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    actions = torch.empty(row_indices.numel(), dtype=torch.long)
    temperature = float(cache.metadata["proxy_temperature"])
    prior_strength = float(cache.metadata["proxy_prior_strength"])
    use_amp = device.type == "cuda"
    for start in range(0, int(row_indices.numel()), batch_size):
        stop = min(start + batch_size, int(row_indices.numel()))
        source_ids = row_indices[start:stop]
        proxy_logits = cache.proxy_logits[source_ids].to(
            device=device, non_blocking=True
        )
        events = cache.events[source_ids].to(device=device, non_blocking=True)
        budgets = torch.full(
            (stop - start,), cache.max_samples, dtype=torch.long, device=device
        )
        counts = _mixed_prefix_counts(
            events,
            budgets,
            vocab_size=cache.vocab_size,
            dtype=torch.float32,
        )
        fused_scores = fuse_proxy_logits_with_mc_counts(
            proxy_logits,
            counts,
            temperature=temperature,
            prior_strength=prior_strength,
            dtype=torch.float32,
        )
        positions = cache.position_ids[source_ids].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            residual = model(
                fused_scores,
                position_ids=(
                    positions
                    if int(getattr(model, "num_position_buckets", 0) or 0) > 0
                    else None
                ),
            )
            outputs = fused_scores + residual
        actions[start:stop] = outputs.argmax(dim=-1).cpu()
    return actions


def _configure_anytime_checkpoint(
    model: BiasNet,
    *,
    cache: AnytimeProxyMCCache,
    budgets: tuple[int, ...],
    train_files: list[str],
    test_files: list[str],
    args: argparse.Namespace,
) -> None:
    model.config.anytime_schema_version = ANYTIME_SCHEMA_VERSION
    model.config.mc_sample_count_conditioning = True
    model.config.mc_sample_count_conditioning_mode = (
        args.mc_sample_count_conditioning_mode
    )
    model.config.mc_max_samples_per_token = cache.max_samples
    model.config.mc_sample_budgets = list(budgets)
    model.config.mc_samples_per_token = None
    model.config.mc_input_representation = "floor_logprob"
    model.config.mc_base_score_representation = "floor_logprob"
    model.config.mc_score_interface = "shared_v1"
    model.config.mc_estimator_schema_version = 1
    model.config.anytime_replay_method = "count_multiset_nested_permutation_v1"
    model.config.anytime_replay_seed = args.replay_seed
    model.config.anytime_train_files = train_files
    model.config.anytime_test_files = test_files
    model.config.anytime_train_epochs = args.epochs
    model.config.anytime_training_seed = args.seed
    model.config.anytime_target_mode = args.target_mode
    model.config.anytime_reference_changed_weight = args.reference_changed_weight
    model.config.anytime_full_budget_anchor_weight = args.full_budget_anchor_weight
    model.config.anytime_freeze_except_mc_conditioning = bool(
        args.freeze_except_mc_conditioning
    )
    if args.reference_checkpoint:
        reference_checkpoint = Path(args.reference_checkpoint).expanduser().resolve()
        model.config.anytime_reference_checkpoint = str(reference_checkpoint)
        model.config.anytime_reference_checkpoint_weights_sha256 = _sha256_file(
            reference_checkpoint / "pytorch_model.bin"
        )
        model.config.anytime_reference_checkpoint_config_sha256 = _sha256_file(
            reference_checkpoint / "config.json"
        )
    model.config.risk_gate_training = "none"
    estimator_field_names = {
        "sample_temperature": "mc_sample_temperature",
        "top_p": "mc_top_p",
        "sample_completion_policy": "mc_completion_policy",
        "observed_alpha": "mc_observed_alpha",
        "floor_mass": "mc_floor_mass",
    }
    for field, value in cache.metadata.items():
        if field == "manifest_configuration_fingerprint":
            model.config.proxy_mc_manifest_configuration_fingerprint = value
        elif value is not None:
            setattr(model.config, estimator_field_names.get(field, field), value)


def _configure_trainable_parameters(
    model: BiasNet, *, freeze_except_mc_conditioning: bool
) -> list[tuple[str, torch.nn.Parameter]]:
    if freeze_except_mc_conditioning:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if (
            model.mc_sample_count_conditioning_mode
            == MC_SAMPLE_COUNT_CONDITIONING_EXACT_ANCHOR
        ):
            model.mc_sample_count_vector.requires_grad = True
        else:
            for parameter in model.mc_sample_count_projection.parameters():
                parameter.requires_grad = True
    else:
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = False
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def main() -> None:
    args = parse_args()
    _validate_args(args)
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    init_checkpoint = Path(args.init_checkpoint).expanduser().resolve()
    if not (init_checkpoint / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"Missing init checkpoint: {init_checkpoint}")

    cache = load_anytime_cache(args.cache_dir)
    budgets = parse_budgets(args.budgets, max_samples=cache.max_samples)
    if args.test_files_from_init_checkpoint:
        test_files = _test_files_from_checkpoint(init_checkpoint)
    else:
        test_files = []
    train_files = sorted(set(cache.record_names) - set(test_files))
    if not train_files:
        raise ValueError("The record split leaves no anytime training files.")
    train_indices = _row_indices_for_records(cache, train_files)
    if args.max_train_rows is not None:
        train_indices = train_indices[: args.max_train_rows]
    train_events = cache.events[train_indices]
    train_row_keys = [cache.row_keys[int(index)] for index in train_indices]
    print(
        json.dumps(
            {
                "stage": "load",
                "records": len(cache.record_names),
                "rows": cache.num_rows,
                "vocab_size": cache.vocab_size,
                "max_samples": cache.max_samples,
                "budgets": budgets,
                "train_records": len(train_files),
                "train_rows": int(train_indices.numel()),
                "test_records": len(test_files),
                "test_files": test_files,
            },
            sort_keys=True,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiasNet.from_pretrained(init_checkpoint, map_location="cpu")
    if getattr(model.config, "mc_input_representation", None) != "floor_logprob":
        raise ValueError("The init checkpoint must use floor_logprob features.")
    if getattr(model.config, "mc_base_score_representation", None) != "floor_logprob":
        raise ValueError("The init checkpoint must use floor_logprob base scores.")
    if model.vocab_size != cache.vocab_size:
        raise ValueError("Checkpoint and cache vocabulary sizes do not match.")
    reference_actions = None
    if args.target_mode == "reference_mc50_action":
        reference_checkpoint = Path(args.reference_checkpoint).expanduser().resolve()
        if not (reference_checkpoint / "pytorch_model.bin").is_file():
            raise FileNotFoundError(
                f"Missing reference checkpoint: {reference_checkpoint}"
            )
        reference_model = BiasNet.from_pretrained(
            reference_checkpoint, map_location="cpu"
        )
        if reference_model.vocab_size != cache.vocab_size:
            raise ValueError("Reference checkpoint and cache vocabulary sizes differ.")
        if (
            getattr(reference_model.config, "mc_input_representation", None)
            != "floor_logprob"
            or getattr(
                reference_model.config, "mc_base_score_representation", None
            )
            != "floor_logprob"
        ):
            raise ValueError("Reference checkpoint must use floor/floor scores.")
        reference_model = reference_model.to(device)
        reference_model.set_up_proj()
        reference_model.eval()
        reference_actions = _compute_reference_actions(
            cache=cache,
            row_indices=train_indices,
            model=reference_model,
            batch_size=args.batch_size,
            device=device,
        )
        print(
            json.dumps(
                {
                    "stage": "reference_actions",
                    "rows": int(reference_actions.numel()),
                    "changes_base": int(
                        reference_actions.ne(cache.base_token_ids[train_indices]).sum()
                    ),
                    "matches_teacher": int(
                        reference_actions.eq(cache.labels[train_indices]).sum()
                    ),
                },
                sort_keys=True,
            )
        )
        del reference_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model.enable_mc_sample_count_conditioning(
        cache.max_samples,
        mode=args.mc_sample_count_conditioning_mode,
    )
    _configure_anytime_checkpoint(
        model,
        cache=cache,
        budgets=budgets,
        train_files=train_files,
        test_files=test_files,
        args=args,
    )
    model = model.to(device)
    model.set_up_proj()
    trainable_named_parameters = _configure_trainable_parameters(
        model,
        freeze_except_mc_conditioning=args.freeze_except_mc_conditioning,
    )
    trainable_parameters = [parameter for _, parameter in trainable_named_parameters]
    if not trainable_parameters:
        raise RuntimeError("The anytime training configuration has no trainable parameters.")
    print(
        json.dumps(
            {
                "stage": "trainable_parameters",
                "conditioning_mode": args.mc_sample_count_conditioning_mode,
                "names": [name for name, _ in trainable_named_parameters],
                "count": sum(parameter.numel() for parameter in trainable_parameters),
            },
            sort_keys=True,
        )
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    temperature = float(cache.metadata["proxy_temperature"])
    prior_strength = float(cache.metadata["proxy_prior_strength"])

    epoch_metrics = []
    num_rows = int(train_indices.numel())
    for epoch in range(args.epochs):
        if args.freeze_except_mc_conditioning:
            model.eval()
        else:
            model.train()
        permuted = permute_events(
            train_events,
            train_row_keys,
            replay_seed=args.replay_seed,
            replicate=epoch,
        )
        # Every row visits every budget once per complete cycle, while each batch
        # remains mixed-K so later epochs cannot catastrophically forget early K.
        row_budgets = torch.tensor(
            [budgets[(row + epoch) % len(budgets)] for row in range(num_rows)],
            dtype=torch.long,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + epoch)
        order = torch.randperm(num_rows, generator=generator)
        loss_sum = 0.0
        anchor_loss_sum = 0.0
        target_correct = 0
        label_correct = 0
        anchor_target_correct = 0
        weight_sum = 0.0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, num_rows, args.batch_size):
            local_ids = order[start : start + args.batch_size]
            source_ids = train_indices[local_ids]
            proxy_logits = cache.proxy_logits[source_ids].to(
                device=device, non_blocking=True
            )
            labels = cache.labels[source_ids].to(device=device, non_blocking=True)
            base_token_ids = cache.base_token_ids[source_ids].to(
                device=device, non_blocking=True
            )
            position_ids = cache.position_ids[source_ids].to(
                device=device, non_blocking=True
            )
            batch_budgets = row_budgets[local_ids].to(device=device)
            batch_events = permuted[local_ids].to(device=device, non_blocking=True)
            batch_targets = (
                reference_actions[local_ids].to(device=device, non_blocking=True)
                if reference_actions is not None
                else labels
            )
            mc_counts = _mixed_prefix_counts(
                batch_events,
                batch_budgets,
                vocab_size=cache.vocab_size,
                dtype=torch.float32,
            )
            fused_scores = fuse_proxy_logits_with_mc_counts(
                proxy_logits,
                mc_counts,
                temperature=temperature,
                prior_strength=prior_strength,
                dtype=torch.float32,
            )
            changed_weight = (
                args.reference_changed_weight
                if reference_actions is not None
                else args.base_wrong_weight
            )
            weights = torch.where(
                base_token_ids.ne(batch_targets),
                torch.full_like(labels, changed_weight, dtype=torch.float32),
                torch.ones_like(labels, dtype=torch.float32),
            )
            full_budgets = torch.full_like(batch_budgets, cache.max_samples)
            full_counts = _mixed_prefix_counts(
                batch_events,
                full_budgets,
                vocab_size=cache.vocab_size,
                dtype=torch.float32,
            )
            full_scores = fuse_proxy_logits_with_mc_counts(
                proxy_logits,
                full_counts,
                temperature=temperature,
                prior_strength=prior_strength,
                dtype=torch.float32,
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                residual = model(
                    fused_scores,
                    position_ids=position_ids,
                    mc_sample_counts=batch_budgets,
                )
                outputs = fused_scores + residual
                per_row_loss = _top1_margin_loss(
                    outputs, batch_targets, args.top1_margin
                )
                full_residual = model(
                    full_scores,
                    position_ids=position_ids,
                    mc_sample_counts=full_budgets,
                )
                full_outputs = full_scores + full_residual
                anchor_per_row_loss = _top1_margin_loss(
                    full_outputs, batch_targets, args.top1_margin
                )
                denominator = weights.sum().clamp_min(1.0)
                partial_loss = (
                    per_row_loss.float() * weights
                ).sum() / denominator
                anchor_loss = (
                    anchor_per_row_loss.float() * weights
                ).sum() / denominator
                loss = partial_loss + args.full_budget_anchor_weight * anchor_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                predictions = outputs.argmax(dim=-1)
                loss_sum += float((per_row_loss.float() * weights).sum().item())
                anchor_loss_sum += float(
                    (anchor_per_row_loss.float() * weights).sum().item()
                )
                target_correct += int((predictions == batch_targets).sum().item())
                label_correct += int((predictions == labels).sum().item())
                anchor_target_correct += int(
                    (full_outputs.argmax(dim=-1) == batch_targets).sum().item()
                )
                weight_sum += float(weights.sum().item())
        metrics = {
            "epoch": epoch + 1,
            "weighted_top1_margin_loss": loss_sum / max(weight_sum, 1.0),
            "weighted_k50_anchor_margin_loss": (
                anchor_loss_sum / max(weight_sum, 1.0)
            ),
            "target_action_accuracy": target_correct / max(num_rows, 1),
            "teacher_label_accuracy": label_correct / max(num_rows, 1),
            "k50_anchor_target_accuracy": (
                anchor_target_correct / max(num_rows, 1)
            ),
            "budget_histogram": {
                str(budget): int(row_budgets.eq(budget).sum().item())
                for budget in budgets
            },
        }
        epoch_metrics.append(metrics)
        print(json.dumps({"stage": "epoch", **metrics}, sort_keys=True))

    output_dir.mkdir(parents=True)
    model.eval()
    model.save_pretrained(output_dir)
    result = {
        "schema_version": ANYTIME_SCHEMA_VERSION,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "init_checkpoint": str(init_checkpoint),
        "output_dir": str(output_dir),
        "budgets": list(budgets),
        "train_files": train_files,
        "test_files": test_files,
        "train_rows": num_rows,
        "args": vars(args),
        "epoch_metrics": epoch_metrics,
    }
    (output_dir / "anytime_training_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": "saved", "output_dir": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
