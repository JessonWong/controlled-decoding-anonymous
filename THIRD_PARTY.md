# Attribution and licensing

BiasNet builds on the JULI residual-control formulation, as discussed in the
manuscript. This anonymous release makes no claim that the underlying BiasNet
formulation is new. Scientific references to prior work are retained.

`GPTFuzz/gptfuzzer/` and the GPTFuzzer seed prompts are taken from the GPTFuzz
checkout used by the project. Its MIT license and original copyright notice
are retained verbatim in `GPTFuzz/LICENSE`. The two API runners are local
adapters; the bundled library also contains adaptations for optional backends.

`baselines/flip_attack/attack.py` identifies its upstream FlipAttack reference
in its module documentation. PAIR and LogiBreak are local adapters; their
presence does not establish equivalence to every setting in the original
implementations. See `docs/BASELINES.md`.

The source workspace did not contain a project-wide license. This export does
not invent a new license grant or replace any upstream terms. Dataset, model,
and dependency licenses remain those of their respective providers. Benchmark
corpora, model/tokenizer snapshots, and trained weights are not redistributed.

