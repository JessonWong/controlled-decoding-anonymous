# Controlled Decoding Attacks on Black-Box LLMs

Anonymous code accompanying the manuscript of the same title. This source
release contains the existing implementation of sample-based distribution
reconstruction, prefix-risk gating, BiasNet training, speculative execution,
baseline adapters, and evaluation tools.

The target is accessed through a text continuation interface. A local tokenizer
maps returned text to actions; a frozen local risk model decides when to apply
a learned residual controller. The speculative path checks each candidate
prefix against this gate. It does not guarantee preservation of the target
distribution or equivalence to repeated single-token API requests.

## Installation

Use Python 3.10 or newer. Training and model-backed inference require a suitable
PyTorch installation and GPU memory for the selected local models. Run the
following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[api,analysis,dev]'
cp .env.example .env
# Fill the needed credentials locally, then export them:
set -a
source .env
set +a
```

`pip install -e .` installs the `prefix_risk` helper package and dependencies;
the research scripts run directly from this checkout. Optional four-bit model
loading additionally needs `pip install -e '.[quantization]'`. The supported
dependency ranges are in `pyproject.toml`; versions used for the release checks
are recorded in `docs/VALIDATION.md`.

## Start here

The offline tests use small tensors, temporary files, and mocked API clients:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q
python scripts/audit_release.py
python training/train_biasnet.py --help
python inference_openrouter.py --help
```

For the training, generation, and scoring workflow, see
[Reproduction guide](docs/REPRODUCING.md). See
[Data preparation](data/README.md) for expected input formats and
[Baseline implementations](docs/BASELINES.md) for the included adapters.

This is a source release. Model weights, cached API samples, benchmark corpora,
and generated responses are not bundled. The current snapshot also has known
differences between the manuscript recipe and historical experiment scripts;
these are recorded in [Reproduction status](docs/REPRODUCIBILITY.md). The release
checks do not establish reproduction of the manuscript's numerical results.

## Code map

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

