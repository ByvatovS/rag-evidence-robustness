# Rules for contributors and coding agents

## Scope

- This repository studies **pre-generation** evidence assessment for RAG. The target
  verifier may consume only a question, retrieved evidence, and ordinary retrieval
  metadata. It must not consume a gold answer, supporting-fact labels, a perturbation
  label, or a future generator answer.
- Do not implement an agentic retrieval policy, a chatbot, a UI, or a large-LLM
  training pipeline here. Those are out of scope for Article 1.
- Work in stages: validate data and strong inexpensive baselines before adding a
  proposed method. Do not manufacture a complex method if deduplication or canonical
  ordering resolves the observed effect.

## Scientific integrity and data

- Treat novelty and performance as hypotheses until supported by logged experiments.
  Never invent paper results, licences, code availability, experimental results, or
  hardware measurements.
- Preserve source provenance, parent question IDs, dataset revision, seeds, hashes,
  split IDs, configurations, package versions, and failure conditions.
- Split original questions before generating variants. Every variant of a parent stays
  in its parent split; audit source-document overlap separately.
- A `safe` transformation must preserve the evidence text/facts and provenance.
  `remove_gold` and `inject_conflict` are provisional labels until audited; they may
  not be assumed valid automatically.
- Automatic labels and manual-audit labels must remain distinguishable.

## Engineering constraints

- CPU-only operation is required. CUDA is optional acceleration; unit tests and smoke
  tests must neither need a GPU nor download data or models.
- Put reusable logic in `src/`; notebooks are thin launchers and visualisations.
- All random transformations must accept and record a seed. Prefer deterministic
  standard-library implementations for smoke tests.
- Keep API keys, Drive contents, raw datasets, checkpoints, and large feature caches
  out of Git. Record them through manifests and source identifiers instead.
