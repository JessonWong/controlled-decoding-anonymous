"""Utility script for running inference on open-weight causal LLMs.
Provides fine-grained control over the sampling loop so you can inspect and
manipulate logits before every token is committed."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmark_data import PromptRecord, collect_prompt_records as load_prompt_records
from modeling_biasnet import BiasNet
from risk_gate import PrefixRiskGate
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with explicit control over sampling.")
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--tokenizer_name", default=None, help="Optional tokenizer identifier if it differs from the model.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--prompt", default=None, help="Single prompt for quick testing or streaming.")
    parser.add_argument("--prompts", nargs="*", default=None, help="Multiple prompts supplied via the command line.")
    parser.add_argument("--prompt_file", default=None, help="Path to a text file with one prompt per line.")
    parser.add_argument(
        "--benchmark",
        choices=["advbench", "harmbench", "sorrybench"],
        default=None,
        help="Load prompts from a benchmark source instead of --prompt/--prompt_file.",
    )
    parser.add_argument(
        "--benchmark_file",
        default=None,
        help="Local benchmark file (HarmBench CSV or SORRY-Bench JSONL).",
    )
    parser.add_argument(
        "--benchmark_mutation",
        default=None,
        help="SORRY-Bench mutation suffix, for example slang or atbash.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N prompts.")
    parser.add_argument("--system_prompt", default=None, help="Optional system prompt when using chat templates.")
    parser.add_argument("--use_chat_template", action="store_true", help="Apply the tokenizer chat template.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling. Set to 0 for greedy decode.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability mass. Set to 1.0 to disable.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling. Set to 0 to disable.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Penalise previously generated tokens (>1.0).")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Prevent repeating n-grams of this size. 0 disables it.")
    parser.add_argument("--min_length", type=int, default=0, help="Enforce a minimum generation length before EOS can appear.")
    parser.add_argument("--stop_sequence", action="append", default=None, help="Stop generation when this substring is produced. Repeatable.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stream", action="store_true", help="Stream decoded text for a single prompt.")
    parser.add_argument("--print_token_info", action="store_true", help="Print per-step token probabilities for active sequences.")
    parser.add_argument("--token_info_top_k", type=int, default=5, help="Number of tokens to display when printing step info.")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of prompts to process together.")
    parser.add_argument("--device", default=None, help="Torch device specifier, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Model weights dtype.")
    parser.add_argument(
        "--device_map",
        default=None,
        help="Optional device map for model parallelism, e.g. 'auto' or a JSON mapping of module names to devices.",
    )
    parser.add_argument("--output_dir", default=None, help="If set, write each generated completion to this directory.")
    parser.add_argument("--output_json", default=None, help="If set, append all generations into this JSONL file.")
    parser.add_argument("--biasnet_ckpt", default=None, help="Path to a BiasNet checkpoint. If unset, BiasNet is disabled.")
    parser.add_argument(
        "--biasnet_dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Runtime dtype for BiasNet. Use float16 to reduce memory during gated inference.",
    )
    parser.add_argument(
        "--biasnet_logprob_source",
        choices=["exact", "sampled"],
        default="exact",
        help="Input distribution passed to BiasNet. Use 'sampled' for Monte Carlo API-style log-prob estimates.",
    )
    parser.add_argument("--mc_samples_per_token", type=int, default=100)
    parser.add_argument("--mc_sample_temperature", type=float, default=1.0)
    parser.add_argument("--mc_top_p", type=float, default=1.0)
    parser.add_argument("--mc_observed_alpha", type=float, default=0.1)
    parser.add_argument("--mc_floor_mass", type=float, default=1e-4)
    parser.add_argument("--mc_seed", type=int, default=None)
    parser.add_argument("--risk_gate_checkpoint", default=None, help="Path to a prefix-risk checkpoint used to gate BiasNet updates.")
    parser.add_argument("--risk_gate_threshold", type=float, default=0.1, help="Apply BiasNet only when the sampled token's risk is below this value.")
    parser.add_argument("--risk_gate_top_k", type=int, default=50, help="Deprecated compatibility flag; gated generation scores only the sampled token.")
    parser.add_argument("--risk_gate_batch_size", type=int, default=16, help="Batch size for risk-head candidate scoring.")
    parser.add_argument("--risk_gate_max_length", type=int, default=None, help="Override the risk-head max sequence length.")
    parser.add_argument("--risk_gate_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--risk_gate_device", default=None, help="Torch device for the risk gate. Defaults to the generation device.")
    parser.add_argument("--risk_gate_model_name", default=None, help="Override the risk-head backbone model from its checkpoint config.")
    parser.add_argument("--risk_gate_load_in_4bit", action="store_true", help="Load the risk-head backbone in 4-bit.")
    parser.add_argument("--risk_gate_trust_remote_code", action="store_true")
    return parser.parse_args()


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_prompt_records(args: argparse.Namespace) -> List[PromptRecord]:
    return load_prompt_records(
        prompt=args.prompt,
        prompts=args.prompts,
        prompt_file=args.prompt_file,
        benchmark=args.benchmark,
        benchmark_file=args.benchmark_file,
        benchmark_mutation=args.benchmark_mutation,
        limit=args.limit,
    )


def collect_prompts(args: argparse.Namespace) -> List[str]:
    """Compatibility wrapper returning only prompt text."""

    return [record.prompt for record in collect_prompt_records(args)]


def prepare_prompt(raw_prompt: str, tokenizer, use_chat_template: bool, system_prompt: Optional[str]) -> str:
    if not use_chat_template:
        return raw_prompt
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": raw_prompt})
    return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(seq), size):
        yield seq[index : index + size]


def load_model_and_tokenizer(args: argparse.Namespace):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = None
    if args.device_map:
        candidate = args.device_map.strip()
        potential_path = Path(candidate)
        if potential_path.is_file():
            with potential_path.open("r", encoding="utf-8") as handle:
                device_map = json.load(handle)
        else:
            try:
                device_map = json.loads(candidate)
            except json.JSONDecodeError:
                device_map = candidate
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            device_map=device_map,
        )

        def _resolve_primary_device(model_obj) -> torch.device:
            def _coerce_device(value):
                if isinstance(value, torch.device):
                    return value
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
                if isinstance(value, str):
                    if value.lower() == "disk":
                        return None
                    return torch.device(value)
                return None

            if hasattr(model_obj, "hf_device_map") and model_obj.hf_device_map:
                preferred_order = [
                    "model.embed_tokens",
                    "transformer.embed_tokens",
                    "transformer.wte",
                    "gpt_neox.embed_in",
                    "lm_head",
                ]
                for module_name in preferred_order:
                    if module_name in model_obj.hf_device_map:
                        device_candidate = _coerce_device(model_obj.hf_device_map[module_name])
                        if device_candidate is not None:
                            return device_candidate
                for mapped_device in model_obj.hf_device_map.values():
                    device_candidate = _coerce_device(mapped_device)
                    if device_candidate is not None:
                        return device_candidate
            try:
                return next(model_obj.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        device = _resolve_primary_device(model)
    else:
        device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer, device


def load_biasnet(ckpt_path: Optional[str], device: torch.device, dtype_name: str = "float32") -> Optional[BiasNet]:
    if not ckpt_path:
        return None
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
    }
    bias_model = BiasNet.from_pretrained(ckpt_path, map_location="cpu")
    bias_model = bias_model.to(dtype=dtype_map[dtype_name], device=device)
    bias_model.set_up_proj()
    bias_model.eval()
    return bias_model


def load_risk_gate(args: argparse.Namespace, default_device: torch.device) -> Optional[PrefixRiskGate]:
    if not args.risk_gate_checkpoint:
        return None
    gate_device = torch.device(args.risk_gate_device) if args.risk_gate_device else default_device
    return PrefixRiskGate(
        checkpoint=args.risk_gate_checkpoint,
        device=gate_device,
        threshold=args.risk_gate_threshold,
        top_k=args.risk_gate_top_k,
        batch_size=args.risk_gate_batch_size,
        max_length=args.risk_gate_max_length,
        dtype=args.risk_gate_dtype,
        model_name=args.risk_gate_model_name,
        load_in_4bit=args.risk_gate_load_in_4bit,
        trust_remote_code=args.risk_gate_trust_remote_code,
    )


def trim_stop_sequences(text: str, stop_sequences: Optional[List[str]]) -> str:
    if not stop_sequences:
        return text
    earliest: Optional[int] = None
    for stop in stop_sequences:
        if not stop:
            continue
        index = text.find(stop)
        if index != -1 and (earliest is None or index < earliest):
            earliest = index
    if earliest is None:
        return text
    return text[:earliest]


def log_token_info(step: int, probs: torch.Tensor, tokenizer, args: argparse.Namespace, finished_mask: torch.Tensor) -> None:
    top_k = min(args.token_info_top_k, probs.size(-1))
    probs_cpu = probs.detach().cpu()
    finished_cpu = finished_mask.detach().cpu()
    for batch_idx in range(probs_cpu.size(0)):
        if finished_cpu[batch_idx]:
            continue
        values, indices = torch.topk(probs_cpu[batch_idx], k=top_k)
        tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
        pieces = [f"{repr(token)}:{value:.4f}" for token, value in zip(tokens, values.tolist())]
        print(f"[step {step} batch {batch_idx}] " + ", ".join(pieces))


def sample_next_tokens(log_probs: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.temperature is not None and args.temperature <= 0.0:
        return torch.argmax(log_probs, dim=-1)
    probs = torch.exp(log_probs)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_candidate_ids(
    logits: torch.Tensor,
    samples_per_token: int,
    temperature: float,
    top_p: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if samples_per_token <= 0:
        raise ValueError("--mc_samples_per_token must be positive.")
    if temperature <= 0:
        raise ValueError("--mc_sample_temperature must be positive.")
    if top_p <= 0 or top_p > 1:
        raise ValueError("--mc_top_p must be in the interval (0, 1].")

    scaled_logits = logits.float() / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        probs = torch.softmax(sorted_logits, dim=-1)
        sampled_sorted = torch.multinomial(
            probs,
            num_samples=samples_per_token,
            replacement=True,
            generator=generator,
        )
        return sorted_indices.gather(dim=-1, index=sampled_sorted)

    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(
        probs,
        num_samples=samples_per_token,
        replacement=True,
        generator=generator,
    )


def sampled_ids_to_log_probs(
    sampled_token_ids: torch.Tensor,
    vocab_size: int,
    observed_alpha: float,
    floor_mass: float,
    device: torch.device,
) -> torch.Tensor:
    if sampled_token_ids.dim() != 2:
        raise ValueError("sampled_token_ids must have shape [batch_size, samples_per_token].")
    if observed_alpha < 0:
        raise ValueError("--mc_observed_alpha must be non-negative.")
    if floor_mass < 0 or floor_mass >= 1:
        raise ValueError("--mc_floor_mass must be in the interval [0, 1).")

    batch_size, samples_per_token = sampled_token_ids.shape
    rows = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    for row_idx in range(batch_size):
        row_samples = sampled_token_ids[row_idx].detach().cpu().long()
        unique_ids, counts = torch.unique(row_samples, sorted=False, return_counts=True)
        observed_count = unique_ids.numel()
        unseen_count = vocab_size - observed_count
        if unseen_count <= 0:
            unseen_log_prob = -torch.inf
            observed_mass = 1.0
        else:
            unseen_log_prob = torch.log(
                torch.tensor(floor_mass / unseen_count, dtype=torch.float32, device=device)
            )
            observed_mass = 1.0 - floor_mass

        rows[row_idx].fill_(unseen_log_prob)
        denom = float(samples_per_token) + observed_alpha * float(observed_count)
        observed_probs = observed_mass * (counts.float().to(device) + observed_alpha) / denom
        rows[row_idx, unique_ids.to(device)] = observed_probs.log()

    return rows


def estimate_sampled_log_probs(
    logits: torch.Tensor,
    args: argparse.Namespace,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    sampled_token_ids = sample_candidate_ids(
        logits=logits,
        samples_per_token=args.mc_samples_per_token,
        temperature=args.mc_sample_temperature,
        top_p=args.mc_top_p,
        generator=generator,
    )
    return sampled_ids_to_log_probs(
        sampled_token_ids=sampled_token_ids,
        vocab_size=logits.size(-1),
        observed_alpha=args.mc_observed_alpha,
        floor_mass=args.mc_floor_mass,
        device=logits.device,
    )


def apply_bias_model(bias_model: BiasNet, bias_input: torch.Tensor) -> torch.Tensor:
    bias_dtype = next(bias_model.parameters()).dtype
    model_input = bias_input.to(dtype=bias_dtype)
    return model_input + bias_model(model_input)


@torch.no_grad()
def manual_generate(
    model,
    tokenizer,
    prompts: List[str],
    args: argparse.Namespace,
    device: torch.device,
    bias_model: Optional[BiasNet],
    risk_gate: Optional[PrefixRiskGate],
    stream: bool = False,
) -> Optional[List[str]]:
    prepared_prompts = [prepare_prompt(p, tokenizer, args.use_chat_template, args.system_prompt) for p in prompts]
    batch_encoding = tokenizer(prepared_prompts, return_tensors="pt", padding=True)
    prompt_ids = batch_encoding["input_ids"].to(device)
    attention_mask = batch_encoding["attention_mask"].to(device)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define a pad_token or pad_token_id for manual generation.")

    batch_size = prompt_ids.size(0)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
    decoded_cache = ["" for _ in range(batch_size)]
    stop_sequences = [s for s in (args.stop_sequence or []) if s]

    generated_ids = prompt_ids
    generated_attention = attention_mask
    past_key_values = None
    next_input_ids = prompt_ids
    supports_cache: Optional[bool] = None
    mc_generator: Optional[torch.Generator] = None
    if args.biasnet_logprob_source == "sampled":
        mc_seed = args.mc_seed if args.mc_seed is not None else args.seed
        if mc_seed is not None:
            mc_generator = torch.Generator(device=device)
            mc_generator.manual_seed(mc_seed)

    for step in range(args.max_new_tokens):
        if supports_cache is False:
            model_input_ids = generated_ids
        else:
            model_input_ids = next_input_ids
        model_kwargs = {
            "input_ids": model_input_ids,
            "attention_mask": generated_attention,
            "use_cache": True,
        }
        if supports_cache:
            model_kwargs["past_key_values"] = past_key_values
        outputs = model(**model_kwargs)
        if supports_cache is None:
            supports_cache = outputs.past_key_values is not None
        if supports_cache:
            past_key_values = outputs.past_key_values
        else:
            past_key_values = None

        logits_full = outputs.logits.float()
        if supports_cache is False:
            positions = generated_attention.sum(dim=1) - 1
            logits = logits_full[torch.arange(batch_size, device=device), positions]
        elif step == 0:
            prompt_positions = attention_mask.sum(dim=1) - 1
            logits = logits_full[torch.arange(batch_size, device=device), prompt_positions]
        else:
            logits = logits_full[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        next_tokens = sample_next_tokens(log_probs, args)
        final_log_probs = log_probs

        if bias_model is not None:
            if risk_gate is None:
                bias_input = (
                    estimate_sampled_log_probs(logits, args, mc_generator)
                    if args.biasnet_logprob_source == "sampled"
                    else log_probs
                )
                final_log_probs = apply_bias_model(bias_model, bias_input)
                next_tokens = sample_next_tokens(final_log_probs, args)
            else:
                def build_answer_prefix(batch_idx: int, token_id: int) -> str:
                    candidate_ids = generated_tokens[batch_idx] + [token_id]
                    return tokenizer.decode(candidate_ids, skip_special_tokens=False)

                step_mask = risk_gate.build_step_mask(
                    prompts=prompts,
                    token_ids=next_tokens,
                    answer_prefix_builder=build_answer_prefix,
                    finished_mask=finished,
                ).to(device)
                if bool(step_mask.any().item()):
                    bias_input = (
                        estimate_sampled_log_probs(logits, args, mc_generator)
                        if args.biasnet_logprob_source == "sampled"
                        else log_probs
                    )
                    biased_log_probs = apply_bias_model(bias_model, bias_input)
                    biased_tokens = sample_next_tokens(biased_log_probs, args)
                    next_tokens = torch.where(step_mask, biased_tokens, next_tokens)
                    final_log_probs = torch.where(step_mask.unsqueeze(-1), biased_log_probs, log_probs)

        probs = torch.exp(final_log_probs)
        next_tokens = next_tokens.masked_fill(finished, pad_token_id)

        if args.print_token_info:
            log_token_info(step, probs, tokenizer, args, finished)

        for idx in range(batch_size):
            if finished[idx]:
                continue
            token_id = int(next_tokens[idx])
            generated_tokens[idx].append(token_id)
            new_text = tokenizer.decode(generated_tokens[idx], skip_special_tokens=True)
            if stream and idx == 0:
                delta = new_text[len(decoded_cache[idx]) :]
                if delta:
                    print(delta, end="", flush=True)
            decoded_cache[idx] = new_text

        if eos_token_id is not None:
            eos_mask = (next_tokens == eos_token_id) & (~finished)
            finished |= eos_mask

        if stop_sequences:
            for idx, text in enumerate(decoded_cache):
                if finished[idx]:
                    continue
                trimmed = trim_stop_sequences(text, stop_sequences)
                if len(trimmed) < len(text):
                    finished[idx] = True
                    decoded_cache[idx] = trimmed
                    if trimmed:
                        tokenized = tokenizer(trimmed, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                        generated_tokens[idx] = tokenized.tolist()
                    else:
                        generated_tokens[idx] = []

        next_input_ids = next_tokens.unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_input_ids], dim=-1)
        attention_extension = (~finished).unsqueeze(-1).to(generated_attention.dtype)
        generated_attention = torch.cat([generated_attention, attention_extension], dim=-1)

        if finished.all():
            break

    completions: List[str] = []
    for tokens, _ in zip(generated_tokens, decoded_cache):
        clean_tokens = [tok for tok in tokens if tok != pad_token_id]
        if eos_token_id is not None:
            while clean_tokens and clean_tokens[-1] == eos_token_id:
                clean_tokens.pop()
        text = tokenizer.decode(clean_tokens, skip_special_tokens=True)
        text = trim_stop_sequences(text, stop_sequences)
        completions.append(text.strip())
    if stream:
        print()
    return completions


def write_outputs(
    output_dir: Path,
    prompts: List[str],
    completions: List[str],
    start_index: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = start_index
    for prompt, completion in zip(prompts, completions):
        filename = output_dir / f"sample_{index:05d}.txt"
        with filename.open("w", encoding="utf-8") as handle:
            handle.write("PROMPT:\n")
            handle.write(prompt)
            handle.write("\n\nCOMPLETION:\n")
            handle.write(completion)
            handle.write("\n")
        index += 1
    return index


def append_jsonl(
    path: Path,
    prompts: List[str],
    completions: List[str],
    record_metadata: Optional[List[dict]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for index, (prompt, completion) in enumerate(zip(prompts, completions)):
            record = {"prompt": prompt, "completion": completion}
            if record_metadata is not None:
                record.update(record_metadata[index])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    try:
        prompt_records = collect_prompt_records(args)
        prompts = [record.prompt for record in prompt_records]
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    if args.stream and len(prompts) != 1:
        print("Streaming mode requires exactly one prompt.", file=sys.stderr)
        sys.exit(1)
    set_seed(args.seed)
    model, tokenizer, device = load_model_and_tokenizer(args)
    bias_model = load_biasnet(args.biasnet_ckpt, device, args.biasnet_dtype)
    risk_gate = load_risk_gate(args, device) if bias_model is not None else None

    output_dir = Path(args.output_dir) if args.output_dir else None
    json_path = Path(args.output_json) if args.output_json else None
    next_file_index = 0

    if args.stream:
        completions = manual_generate(model, tokenizer, prompts, args, device, bias_model, risk_gate, stream=True)
        if output_dir:
            next_file_index = write_outputs(output_dir, prompts, completions, next_file_index)
        if json_path:
            append_jsonl(
                json_path,
                prompts,
                completions,
                [dict(record.metadata) for record in prompt_records],
            )
        return

    for batch_start in range(0, len(prompt_records), args.batch_size):
        batch_records = prompt_records[batch_start : batch_start + args.batch_size]
        batch = [record.prompt for record in batch_records]
        completions = manual_generate(model, tokenizer, batch, args, device, bias_model, risk_gate, stream=False)
        for prompt, completion in zip(batch, completions):
            print("=== Prompt ===")
            print(prompt)
            print("=== Completion ===")
            print(completion)
            print()
        if output_dir:
            next_file_index = write_outputs(output_dir, batch, completions, next_file_index)
        if json_path:
            append_jsonl(
                json_path,
                batch,
                completions,
                [dict(record.metadata) for record in batch_records],
            )

if __name__ == "__main__":
    main()
