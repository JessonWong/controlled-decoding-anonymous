# GPTFuzz baseline source

This directory bundles the existing GPTFuzz library, upstream seed prompts,
and the project's OpenRouter and native Gemini adapters. The original MIT
license is preserved in `LICENSE`.

Run the API adapters from the repository root:

```bash
python GPTFuzz/run_glm5_openrouter.py --help
python GPTFuzz/run_gemini_google.py --help
```

The OpenRouter adapter accepts configurable target and mutator model names
despite its historical filename. Supply your benchmark CSV explicitly through
`--questions-path`; generated benchmark files are not included. The preparation
helper is `scripts/prepare_gptfuzz_benchmark_first100.py`.

See `../docs/BASELINES.md` for credentials, dependencies, and provenance.

