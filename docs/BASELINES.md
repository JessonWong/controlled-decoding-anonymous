# Baselines

The release includes the source adapters used in this project:

| Method | Entry point | Notes |
| --- | --- | --- |
| FlipAttack | `baselines/flip_attack/generate.py` | Input transforms and provider adapters |
| PAIR | `baselines/pair_attack/run_openrouter.py` | Local reference-style search/selection loop |
| LogiBreak | `baselines/logibreak/run_openrouter.py` | English reformulation and restart adapter |
| GPTFuzz | `GPTFuzz/run_glm5_openrouter.py` | Bundled fuzzer with configurable API target/mutator |
| GPTFuzz, native Gemini | `GPTFuzz/run_gemini_google.py` | Native Google API adapter |

Use each entrypoint's `--help` for its arguments. These are the source adapters
present in the workspace, not a certification of identical settings to all
upstream releases. PAIR uses a local prompt template and internal judge loop;
LogiBreak retains restart candidates for subsequent scoring.

The API runners use environment credentials. PAIR and LogiBreak now default to
`OPENROUTER_API_KEY` and their legacy key-file option is optional. Native Gemini
GPTFuzz checks `GOOGLE_API_KEY` or `GEMINI_API_KEY` before its key-file fallback.
Actual key files are excluded from this repository.

For Azure, provide your deployment URL with `--base-url`; the old account URL
has been replaced with an example hostname.

GPTFuzz includes its original MIT license and seed CSV. The project-specific
benchmark CSVs and trained predictor weights are external inputs. For API-only
GPTFuzz runs, the installation in the main README provides the necessary SDKs;
the upstream local FastChat/vLLM backends need their own optional dependencies.

Keep generation parameters, search budgets, benchmark indices, and all restart
candidates with the result artifacts. Final harmfulness/information scores use
the common evaluation scripts under `test/`.
