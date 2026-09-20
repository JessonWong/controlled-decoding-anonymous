# Reproduction status

This release preserves the supplied implementation, with anonymity and
portability edits. It does not include a verified mapping from every table row
to its exact cache, checkpoint, and invocation.

## Manuscript settings and historical artifacts

`configs/manuscript_recipe.json` transcribes the manuscript settings. It is a
reference, not an executable configuration accepted by the existing scripts.

The source audit identified the following concrete differences:

* The manuscript specifies risk threshold 0.1, 40 records at indices 100–139,
  and a Llama-3.1-8B prefix-risk model. A historical GLM-5 runtime-soft job uses
  threshold 0.9641336778984433, 41 records, and a separately trained handoff
  risk head. These checkpoints and thresholds are not interchangeable.
* The legacy risk-head artifact in the source workspace identifies
  `meta-llama/Meta-Llama-3-8B-Instruct` as its backbone. It is not evidence for
  the manuscript's Llama-3.1 configuration and is not included in this release.
* Native Gemini sampled inference currently uses the observed-support/floor
  estimator. It does not expose the global-uniform Dirichlet mode provided by
  the OpenRouter path. The Gemini sampler also has different default row
  limits and does not expose a dataset-revision argument.
* The sampler defaults are not a complete manuscript recipe. The intended
  dataset revision, tokenizer revision, provider routing, prefill protocol,
  checkpoint, and generation arguments must be recorded together for a run.

No experimental results were changed or fabricated to resolve these differences.
The unit tests verify code behavior; they do not settle experiment provenance.

## Artifacts needed for a numerical replication

A result-level reproduction needs the exact frozen risk head and compatible
tokenizer, per-target BiasNet checkpoints (or complete training caches), dataset
revisions and held-out filenames, and endpoint/provider settings. These are
external inputs to this source release. The prefix-risk labeling and training
code is included for building new heads, but retraining a head is a new run.

Keep the cache manifests, `static_prior_manifest.json`, controller configuration,
training metrics, generation audit records, and judge outputs together. Hosted
API execution and GPU training were not run as part of preparing this release.

