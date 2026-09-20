# Running the existing pipeline

Run all commands from the repository root after installation. These examples
show the source interfaces; they do not claim to recreate a specific reported
table row. Read `REPRODUCIBILITY.md` before selecting experimental settings.

## 1. Prefix-risk model

The gate checkpoint directory must contain `risk_head.pt`,
`risk_head_config.json`, and the tokenizer saved during risk-head training.
The configured backbone must match its hidden dimension, layer indices,
tokenizer, and training configuration. Use the original checkpoint associated
with a result when comparing with it.

The included package provides dataset labeling and training:

```bash
python -m prefix_risk.build_guard_dataset --help
python -m prefix_risk.train --help
python -m prefix_risk.evaluate_checkpoint --help
```

For a new head, use `build_guard_dataset --output-path ...` and then
`train --labeled-jsonl ... --output-dir ...`, choosing explicit backbone and
guard models. This retraining is not equivalent to the missing frozen artifact.

## 2. Cache target samples

Set the following variables to the endpoint, tokenizer, immutable dataset
revision, and gate artifact for your experiment:

```bash
export TARGET_MODEL='your-target-model-id'
export TOKENIZER='your-compatible-tokenizer-id-or-directory'
export DATASET_REVISION='your-immutable-dataset-revision'
export RISK_CHECKPOINT='checkpoints/prefix-risk'
export GATE_THRESHOLD='0.1'
```

The existing OpenRouter sampler can store every reference-answer prefix with
a pinned dataset slice:

```bash
python training/pre_logits_sampled_openrouter.py \
  --model "$TARGET_MODEL" --tokenizer_name "$TOKENIZER" \
  --dataset_revision "$DATASET_REVISION" \
  --start_index 100 --end_index 140 --max_samples 40 \
  --samples_per_token 50 --sample_temperature 1 --top_p 1 \
  --sample_choices_per_request 1 --sample_completion_policy exact \
  --observed_alpha 0.1 --floor_mass 0.0001 \
  --risk_gate_checkpoint "$RISK_CHECKPOINT" --risk_gate_threshold "$GATE_THRESHOLD" \
  --cache_manifest_policy require \
  --output_dir cached_logits/raw
```

Provider routing, prefill support, reasoning rejection, response caching, and
API token budgets must match the target and intended run; they are not inferred
by this example. The example threshold 0.1 follows the manuscript and must be
changed if using a different risk head. Inspect the sampler's `--help` and its output manifest. For
Qwen protocols, record whether hard no-think prefill is enabled. To cache risk
scores, pass the matching `--risk_gate_checkpoint`, or use
`training/rescore_risk_gate_cache.py` on a complete raw cache. Do not enable
`--sample_only_risk_active` when the intended dataset includes every prefix.

The native Gemini path is `training/pre_logits_sampled_gemini.py`; its defaults
and estimator differ as documented in `REPRODUCIBILITY.md`.

## 3. Materialize the global-uniform prior

The raw sampler's observed-support/floor representation is distinct from the
manuscript's uniform Dirichlet representation. The existing conversion step
recovers integer counts, verifies their totals, and then materializes the
selected static prior:

```bash
python scripts/recover_mc_counts.py \
  --raw_dir cached_logits/raw --output_dir cached_logits/counts
python training/materialize_static_mc_prior_cache.py \
  --raw_dir cached_logits/raw --counts_dir cached_logits/counts \
  --output_dir cached_logits/uniform --mode uniform_dirichlet_v1 \
  --held_out_file_count 8 --held_out_split_seed qwen3-dual-rank-v1 \
  --kappas 2
```

The count recovery helper is extracted from the existing job workflow. It
rejects caches that cannot be reconstructed consistently; it does not silently
invent missing observations. Keep the held-out filenames consistent between
materialization and training. With `--kappas 2`, the prior strength is fixed;
use a grid only when deliberately repeating calibration.

## 4. Train BiasNet

For run-time-soft training, the input cache must contain `risk_gate_scores`
from the intended frozen head. Set `VOCAB_SIZE` to the final dimension of the
cache tensors; do not infer it solely from a model name. `GATE_THRESHOLD` must
belong to that same risk-head configuration.

```bash
export VOCAB_SIZE='your-cache-vocabulary-size'
python training/train_biasnet.py \
  --data_dir cached_logits/uniform --output_dir checkpoints/biasnet \
  --held_out_file_count 8 --held_out_split_seed qwen3-dual-rank-v1 \
  --hidden_size 1024 --vocab_size "$VOCAB_SIZE" \
  --lm_head_init none --input_projection_mode count_sketch \
  --count_sketch_hashes 4 --input_hidden_normalization layernorm \
  --epochs 10 --batch_size 32 --learning_rate 0.0003 --weight_decay 0.0001 \
  --ce_loss_weight 1 --margin_loss_weight 0 --top1_margin_loss_weight 0 \
  --risk_gate_training runtime_soft --risk_gate_threshold "$GATE_THRESHOLD" \
  --risk_gate_soft_temperature 0.05 --risk_gate_min_scale 0.01 \
  --risk_gate_warmup_tokens 3 --drop_zero_weight_tokens \
  --mixed_precision --seed 42
```

The saved controller configuration records its representation and prior
contract. Keep `config.json`, model weights, and training metrics together.

## 5. Generate and evaluate

The following example uses the included benign format-check prompts. Supply a
benchmark through `--benchmark` and `--benchmark_file` for benchmark evaluation.

```bash
python inference_openrouter.py \
  --model "$TARGET_MODEL" --tokenizer_name "$TOKENIZER" \
  --biasnet_ckpt checkpoints/biasnet \
  --risk_gate_checkpoint "$RISK_CHECKPOINT" \
  --risk_gate_mode soft --risk_gate_threshold "$GATE_THRESHOLD" \
  --risk_gate_soft_temperature 0.05 --risk_gate_min_scale 0.01 \
  --risk_gate_warmup_tokens 3 \
  --mc_samples_per_token 50 --mc_sample_temperature 1 \
  --sample_choices_per_request 1 --sample_completion_policy exact \
  --mc_static_prior_mode uniform_dirichlet_v1 --mc_static_prior_strength 2 \
  --risk_gate_speculative_draft --risk_gate_speculative_min_base_streak 2 \
  --risk_gate_speculative_draft_tokens 80 --risk_gate_batch_size 8 \
  --temperature 0 --max_new_tokens 80 \
  --prompt_file data/example_prompts.txt --output_json outputs/generation.jsonl
```

Apply the same endpoint/protocol options as the cache run. The two main judge
entrypoints accept the generation JSONL:

```bash
python test/eval_harmful_score.py \
  --input_file outputs/generation.jsonl --output_file outputs/harm_score.jsonl \
  --judge_provider gemini --judge_model gemini-3.5-flash
python test/eval_harmful_info_score.py \
  --input_file outputs/generation.jsonl --output_file outputs/harm_info.jsonl \
  --judge_provider gemini --judge_model gemini-3.5-flash
```

These commands make remote requests when run with real inputs and credentials.
They are documented entrypoints, not operations performed during release
validation. Review the saved API audit fields alongside the scores.
