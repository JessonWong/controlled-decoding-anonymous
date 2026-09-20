# Data inputs

The included examples contain benign text and exercise file formats only.
They are not training or evaluation data used in the manuscript.

Training samplers generally read `LLM-LAT/harmful-dataset`, with fields `prompt`
and `rejected`. The OpenRouter sampler accepts `--dataset_revision` to pin an
immutable revision and `--dataset_jsonl` for local records with the same fields.
The intended manuscript slice is `[100, 140)`; see the reproduction-status
document before comparing historical artifacts.

Provide benchmark files locally using `--benchmark_file` (inference scripts) or
`--data-file` (PAIR/LogiBreak). The loader in `benchmark_data.py` supports:

| Benchmark | Local format |
| --- | --- |
| AdvBench | CSV with a `goal` column, or one prompt per line |
| HarmBench | Official behavior CSV, including behavior/context fields |
| SORRY-Bench | Official `question.jsonl` |

Use the benchmark versions and ordering corresponding to the intended
experiment. The loader preserves benchmark indices and available identifiers.
Its error messages give upstream locations for missing HarmBench and
SORRY-Bench files. Obtain the data under its upstream terms.

For a tiny local format check:

```bash
python -c 'from benchmark_data import load_benchmark_records; print(load_benchmark_records("advbench", path="data/example_prompts.txt"))'
```

