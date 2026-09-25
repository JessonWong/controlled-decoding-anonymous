# Controlled Decoding Attacks on Black-Box LLMs

Anonymous research code accompanying the manuscript of the same title. This
repository contains sample-based distribution reconstruction, prefix-risk
gating, BiasNet training, speculative execution, prompt-level baseline adapters,
and evaluation tools.

**Release scope.** The implementation, synthetic examples, and offline tests are
included. Trained weights, API caches, benchmark corpora, and generated responses
are external inputs. Some historical experiment settings differ from the stated
manuscript recipe; see [reproduction status](docs/REPRODUCIBILITY.md) before
attempting a result-level replication.

## Contents

- [Method overview](#method-overview)
- [Installation and credentials](#installation-and-credentials)
- [Offline quick start](#offline-quick-start)
- [Data and required artifacts](#data-and-required-artifacts)
- [Training and evaluation workflow](#training-and-evaluation-workflow)
- [Parameter reference](#parameter-reference)
- [Backends, baselines, and ablations](#backends-baselines-and-ablations)
- [Outputs and experiment bookkeeping](#outputs-and-experiment-bookkeeping)
- [Troubleshooting](#troubleshooting)
- [Repository layout and further documentation](#repository-layout-and-further-documentation)

## Method overview

The main setting assumes a target interface that can continue a supplied
assistant prefix. Target weights and numerical next-token probabilities are
unavailable. A locally available tokenizer defines the action vocabulary used
to interpret returned text and construct the next prefix.

At a generation step, the target proposes a base candidate. A frozen local
prefix-risk model scores the prompt together with the candidate continuation.
The gate determines whether to accept that candidate or collect additional
samples and apply the learned BiasNet residual.

| Component | Role | Main implementation |
| --- | --- | --- |
| Sample reconstruction | Convert repeated returned continuations into action counts and estimated probabilities | `mc_reconstruction.py`, `training/pre_logits_sampled_openrouter.py` |
| Prefix-risk gate | Score the evolving candidate prefix and determine residual strength | `risk_gate.py`, `src/prefix_risk/` |
| BiasNet | Transform estimated log probabilities into a learned residual over the same vocabulary | `modeling_biasnet.py` |
| Speculative execution | Request a longer target draft and verify its prefixes locally during bypassed stretches | `inference_openrouter.py`, `inference_gemini_sampled.py` |

The global-uniform reconstruction uses counts `c[v]` from `K` valid actions and
total prior strength `kappa`:

```text
p_hat[v] = (c[v] + kappa / vocabulary_size) / (K + kappa)
controlled_logits = log(p_hat) + gate_scale * BiasNet(log(p_hat))
```

After full-strength warm-up, the soft gate uses
`sigmoid((threshold - risk_score) / gate_temperature)`, with scales at or below
the configured cutoff bypassing intervention. A larger estimated risk therefore
reduces the residual strength. The gate is evaluated on the current candidate
prefix and can reactivate later in a response.

Training uses cached reference-answer prefixes; inference uses generated
prefixes. Speculative verification enforces the gate rule on accepted draft
prefixes. It does not establish preservation of the target sampling distribution
or token-for-token equivalence with repeated single-token API calls.

## Installation and credentials

Use **Python 3.10+**. GPU memory requirements depend on the chosen local risk
backbone, controller vocabulary, precision, and optional local target/proxy
models. Offline unit tests run on CPU. Model-backed training and inference need
the corresponding weights; remote execution additionally needs API access.

Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[api,analysis,dev]'
```

Select a PyTorch build appropriate for your machine. The dependency declarations
are in [pyproject.toml](pyproject.toml); the environment used for the original
release checks is recorded in [validation notes](docs/VALIDATION.md).

| Installation option | Purpose |
| --- | --- |
| `pip install -e .` | Core dependencies and the `prefix_risk` helper package |
| `pip install -e '.[api]'` | Google Gen AI, OpenAI-compatible, and Bedrock SDKs |
| `pip install -e '.[analysis]'` | Additional statistical and learned-gate analysis dependencies |
| `pip install -e '.[dev]'` | pytest and package-build tools |
| `pip install -e '.[quantization]'` | Optional bitsandbytes model loading |

The research entrypoints run directly from this source checkout. Installing the
helper package does not turn every script into a system-wide executable.

Copy the credential template and fill only the entries needed for your run:

```bash
cp .env.example .env
# Edit .env locally, then export its values into the current shell:
set -a
source .env
set +a
```

The scripts read environment variables; they do not automatically load `.env`.

| Variable | Used by |
| --- | --- |
| `OPENROUTER_API_KEY` | OpenRouter sampling, inference, and baseline adapters |
| `GEMINI_API_KEY` | Default native Gemini sampled generation/cache entrypoints |
| `GOOGLE_API_KEY` | Google-based judges; Gemini sampler can select this through `--api_key_env` |
| `OPENAI_API_KEY` | OpenAI-based judges and relevant baseline adapters |
| `AWS_BEARER_TOKEN_BEDROCK` | Bedrock sampling adapter |
| `HF_TOKEN` | Model/dataset access when authentication is required |
| `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | Legacy Vertex-based Gemini cache entrypoint |

## Offline quick start

These checks require the installed dependencies but no API credentials or
pretrained checkpoints:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q
python -c 'from benchmark_data import load_benchmark_records; print(load_benchmark_records("advbench", path="data/example_prompts.txt"))'
python training/train_biasnet.py --help
python inference_openrouter.py --help
```

The original release validation recorded **320 passed tests, 62 passed subtests,
and one skipped test** for excluded machine-specific Slurm wrappers. It also
checked a synthetic count-recovery, prior-materialization, CPU-training, and
checkpoint-reload pipeline. These checks exercise implementation behavior, not
reproduction of the paper's numerical results.

For a pristine release checkout, `python scripts/audit_release.py` additionally
checks anonymous-release contents and the SHA-256 manifest. This is a packaging
check: local `.env` files, downloaded artifacts, configured remotes, or edits can
cause it to report findings even when the research code is working correctly.

## Data and required artifacts

The files under `data/example_*` contain benign synthetic examples. They show
input formats and are not the training/evaluation data used in the manuscript.
See [data preparation](data/README.md) for details.

| Input | Expected contents | Where it is used |
| --- | --- | --- |
| Training records | `prompt` and `rejected` fields from a pinned dataset or compatible JSONL | Target sampling |
| Prefix-risk checkpoint | `risk_head.pt`, `risk_head_config.json`, saved tokenizer, and access to its compatible backbone | Cache risk scores and inference gate |
| Target-compatible tokenizer | Tokenizer ID/directory and its revision | Consistent action IDs across sampling, training, and generation |
| BiasNet checkpoint | `config.json` and `pytorch_model.bin` produced by training | Controlled generation |
| Evaluation prompts | AdvBench CSV/text, HarmBench behavior CSV, or SORRY-Bench JSONL | Generation and benchmarking |

The intended manuscript training slice is `[100, 140)` in
`LLM-LAT/harmful-dataset`, with eight records held out. Pin the dataset revision
and retain the actual held-out filenames. A count alone does not identify the
split. The immutable revision and original result-matched weights are not
supplied by this source release.

The benchmark loader accepts AdvBench CSV files with a `goal` column or one
prompt per line, official HarmBench behavior CSV files, and SORRY-Bench
`question.jsonl`. Supply paths explicitly using `--benchmark_file` in inference
or `--data-file` in the PAIR/LogiBreak adapters. The corpus files are not bundled.

## Training and evaluation workflow

The workflow below exposes the existing OpenRouter interfaces in execution
order. Replace the placeholder values before running it. Target sampling,
generation, and judging make remote requests; the cache conversion steps run
locally. This is a workflow template, not a verified invocation for every
reported table row.

```text
Training records + target API + tokenizer + frozen risk head
    -> raw sample cache and prefix-risk scores
    -> recovered action counts
    -> global-uniform prior cache
    -> trained BiasNet checkpoint

Evaluation prompts + target API + tokenizer + risk head + BiasNet
    -> generated responses and API audits
    -> harmfulness and information judge outputs
```

### 1. Prepare the prefix-risk model

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

### 2. Cache target samples

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
and estimator differ as documented in [reproduction status](docs/REPRODUCIBILITY.md).

### 3. Materialize the global-uniform prior

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

### 4. Train BiasNet

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

### 5. Generate and evaluate

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


## Parameter reference

These values describe the manuscript reference recipe in
[configs/manuscript_recipe.json](configs/manuscript_recipe.json). That JSON file
is documentation; the scripts do not accept it as a shared configuration file.

| Setting | Reference value | Meaning |
| --- | --- | --- |
| Reconstruction samples | 50 | Valid sampled actions per controlled position |
| Sampling temperature / top-p | 1 / 1 | Reconstruction sampling settings |
| Uniform prior strength | 2 | Total prior mass over the vocabulary, not mass per token |
| Controller hidden size | 1,024 | Count-sketch representation width |
| Count-sketch hashes | 4 | Input projection configuration |
| Training | 10 epochs, batch size 32 | Per-target controller training |
| Optimizer settings | Learning rate `3e-4`, weight decay `1e-4` | AdamW settings |
| Held-out records | 8 | Record-level holdout; use the same split seed in materialization and training |
| Gate threshold / temperature | 0.1 / 0.05 | Soft gate parameters tied to the chosen risk head |
| Gate cutoff / warm-up | 0.01 / 3 tokens | Bypass cutoff and initial full-strength positions |
| Speculative activation | 2 bypassed positions | Required consecutive bypasses before requesting a draft |
| Maximum draft / verification batch | 80 tokens / 8 prefixes | Draft length and local risk batching |
| Generation temperature / limit | 0 / 80 local tokens | Controlled decoding settings |
| Training seed | 42 | Seed for the specified training recipe |

Do not substitute the manuscript threshold into an unrelated trained risk head.
Historical jobs include a different threshold and a 41-record cache. Native
Gemini also uses a different reconstruction path in this snapshot. The precise
differences are documented in [reproduction status](docs/REPRODUCIBILITY.md).

## Backends, baselines, and ablations

### Target backends

| Backend | Sampling entrypoint | Generation entrypoint |
| --- | --- | --- |
| OpenRouter | `training/pre_logits_sampled_openrouter.py` | `inference_openrouter.py` |
| Native Gemini | `training/pre_logits_sampled_gemini.py` | `inference_gemini_sampled.py` |
| Local open-weight models | `training/pre_logits_sampled_openweight.py`, `training/pre_logits_exact_openweight.py` | `inference_local_openweight.py`, `inference_opensource.py` |
| Bedrock | `training/pre_logits_sampled_bedrock.py` | See adapter-specific options; the OpenRouter command above is not a native Bedrock command |

For the manuscript reference, local tokenizer coordinates are
`zai-org/GLM-5`, `Qwen/Qwen3-32B`, `moonshotai/Kimi-K2.5`, and
`google/gemma-3-1b-pt` for the respective targets. Gemma coordinates do not imply
access to Gemini's native tokenizer. Vocabulary dimensions and prefix alignment
must be checked against the actual cache. API-side token budgets and local
action-token budgets need not coincide.

### Prompt-level baselines

| Method | Entry point |
| --- | --- |
| FlipAttack | `baselines/flip_attack/generate.py` |
| PAIR | `baselines/pair_attack/run_openrouter.py` |
| LogiBreak | `baselines/logibreak/run_openrouter.py` |
| GPTFuzz, OpenRouter | `GPTFuzz/run_glm5_openrouter.py` |
| GPTFuzz, native Gemini | `GPTFuzz/run_gemini_google.py` |

Each entrypoint provides `--help`. The GPTFuzz OpenRouter runner accepts
configurable target/mutator models despite its historical filename. Baselines
have their own search budgets, routing, and output-selection rules; see
[baseline notes](docs/BASELINES.md) for implementation provenance and input
requirements. Score their final generations with the common judge scripts to
keep evaluation settings explicit.

### Analysis and ablation code

| Analysis | Relevant source |
| --- | --- |
| Uniform versus global-unigram priors | `training/materialize_static_mc_prior_cache.py` |
| Reconstruction and held-out sample diagnostics | `training/eval_proxy_mc_fusion.py` |
| Gate rescoring and controller loss variants | `training/rescore_risk_gate_cache.py`, `training/train_biasnet.py` |
| Numerical log-probability reference caches | `training/pre_logits_exact_openweight.py` |
| Controller intervention plots | `scripts/plot_biasnet_intervention.py`, `scripts/plot_biasnet_intervention_comparison.py` |
| Foreign-tokenizer and empirical-vocabulary experiments | `foreign_proxy.py`, `empirical_vocab/` |

These are source entrypoints for analyses, not bundled result artifacts or a
certified mapping to each ablation-table row. In particular, better held-out
reconstruction likelihood is a different measurement from downstream judge
scores. Preserve each analysis's data split and scoring convention.

## Outputs and experiment bookkeeping

Using the example paths above, the main artifacts are:

| Location | Contents |
| --- | --- |
| `cached_logits/raw/` | Per-record tensors, sampling metadata, and cache manifest |
| `cached_logits/counts/` | Recovered integer counts and `counts_manifest.json` |
| `cached_logits/uniform/` | Prior-materialized tensors and `static_prior_manifest.json` |
| `checkpoints/biasnet/` | Controller configuration, weights, and `training_metrics.json` |
| `outputs/generation.jsonl` | Prompt/completion records with available generation audit information |
| `outputs/harm_score.jsonl` | Harmfulness judge output |
| `outputs/harm_info.jsonl` | Information judge output |

Keep model/provider identifiers, tokenizer and dataset revisions, prefill
settings, cache manifests, held-out filenames, checkpoint configuration, and
judge settings together. API request counts and sampled-action counts measure
different things when retries, batching, or speculative drafts are involved.
The scripts' output metadata is the record of the run actually performed.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Missing `prefix_risk` or another dependency | Activate the intended environment and install the required extras from the repository root. |
| API key not found | Export `.env` into the shell; check the entrypoint's `--api_key_env` or baseline-specific credential option. |
| A benchmark file is missing | Supply the external corpus path explicitly; synthetic examples do not stand in for benchmark data. |
| Tokenizer, vocabulary, or checkpoint incompatibility | Compare cache tensor dimensions and tokenizer identity with the saved controller and risk-head configuration. |
| Missing risk scores during run-time-soft training | Build or rescore the raw cache with the intended frozen head before prior materialization and training. |
| Exact sampling cannot collect enough valid actions | Inspect invalid/empty-response and retry audits, provider continuation support, and provider-side output budgets. |
| Count recovery rejects a cache | Check that it is the original supported floor representation with matching sample counts and smoothing metadata. |
| A materialization directory already exists | Use a fresh output directory; the conversion helpers protect existing artifacts from accidental replacement. |
| Anonymous-release audit fails after local setup | The scanner checks distributable contents; local secrets, model files, edits, and remotes differ from a pristine release. |

## Repository layout and further documentation


| Path | Purpose |
| --- | --- |
| `modeling_biasnet.py` | Residual controller, projections, checkpoint I/O |
| `mc_reconstruction.py` | Sample counts, probability reconstruction, priors |
| `risk_gate.py`, `src/prefix_risk/` | Prefix-risk inference, labeling, and training |
| `training/` | Cache creation/materialization, controller training, analyses |
| `inference_openrouter.py` | OpenRouter decoding, gating, speculative execution |
| `inference_gemini_sampled.py` | Native Gemini sampled decoding adapter |
| `inference_local_openweight.py`, `inference_opensource.py` | Local model controls |
| `empirical_vocab/` | Alternative vocabulary alignment experiments |
| `baselines/`, `GPTFuzz/` | Prompt-level comparison methods |
| `test/eval_*.py`, `test/judge_client.py` | Evaluation and judge clients |
| `test/test_*.py` | Offline regression tests |
| `scripts/` | Portable analysis and release helpers |
| `configs/manuscript_recipe.json` | Stated manuscript settings, for reference |

Auxiliary training modules are retained where they support ablations or imports
of the main implementation. Machine-specific job submissions and exploratory
run directories are excluded. Third-party provenance and license information
are in [THIRD_PARTY.md](THIRD_PARTY.md).



| Document | Read it for |
| --- | --- |
| [Reproduction guide](docs/REPRODUCING.md) | Focused training-to-evaluation command reference |
| [Reproduction status](docs/REPRODUCIBILITY.md) | Missing artifacts and known manuscript/implementation differences |
| [Data preparation](data/README.md) | Local input formats and benchmark handling |
| [Baseline notes](docs/BASELINES.md) | Adapter provenance and dependencies |
| [Validation](docs/VALIDATION.md) | Recorded release checks and observed environment versions |
| [Anonymization](docs/ANONYMIZATION.md) | Export exclusions and manifest checks |
| [Third-party attribution](THIRD_PARTY.md) | Attribution and existing license information |

The bundled GPTFuzz license is retained in [GPTFuzz/LICENSE](GPTFuzz/LICENSE).
No new project-wide license grant is asserted by this anonymous export.
