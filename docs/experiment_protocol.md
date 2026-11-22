# Pilot protocol: robustness of pre-generation evidence assessment

Status: **draft for review before data/model experiments**. This document is a
screening-pilot protocol, not a claim of results or a completed preregistration.

## Question and hypothesis

Given a question `q` and retrieved evidence `E`, does a verifier's adequacy/conflict
decision remain stable under transformations that preserve text and source provenance,
while responding to audited content-changing interventions? We will first diagnose
this behaviour in inexpensive baselines. Only if a reproducible weakness remains after
deduplication and canonical-order controls will we evaluate a source-aware method.

## Input contract and leakage policy

The verifier receives only `question`, evidence text, source identifiers, and ordinary
retrieval metadata. Gold answers, supporting-fact annotations, transformation names,
and generated answers are prohibited from `features.py`, baseline methods, and any
future proposed method. They may be used only to construct provisional labels and to
audit/evaluate variants.

## Dataset and splits

Start with `hotpotqa/hotpot_qa`, `distractor` configuration. The exact dataset
revision, downloaded file checksums, seed, selected original IDs, and environment will
be written to a manifest before fitting. Split original questions 60/20/20 before
creating variants. All examples with the same `parent_question_id` are restricted to
one split. Audit source-document overlap separately.

## Interventions and label validation

Strict safe interventions are `permute`, exact same-source `duplicate`, and
sentence-preserving `resegment`. `add_distractor` is reported separately as
noise-robustness, not automatically as semantically safe. Content-changing
`remove_gold` and `inject_conflict` are provisional until a blinded audit confirms:

1. the deleted fact is necessary and no alternative evidence path remains; or
2. the injected claim concerns the same entity/relation/time conditions and conflicts.

Record invalid and ambiguous cases; do not silently discard them. Audit a stratified
sample before interpreting model results.

## Baselines and selection rule

Evaluate B0 retrieval-score aggregates, B1 a lightweight classifier, B2 local NLI
only if the resource budget permits, B3 exact deduplication, and B4 canonical order
or permutation averaging. Literature methods are run only under compatible input
contracts; otherwise they are described as non-comparable or explicitly labelled an
approximation. Train on train, choose thresholds and all hyperparameters on dev, and
touch test once after choices are frozen.

## Primary outcomes

- sufficiency/conflict Macro-F1 and classwise precision/recall;
- paired safe `flip_rate`, mean/median/P95 absolute score change;
- recall of audited harmful changes and false alarms on safe variants;
- median/P95 wall-clock inference time, peak RAM/VRAM where applicable, and NLI calls.

Confidence intervals and paired comparisons resample `parent_question_id`, never each
variant independently. Report absolute values and intervals alongside differences.

## Pilot decision gate

Proceed to a proposed method only if all of the following hold:

1. at least one strong simple baseline (especially B3/B4) shows a reproducible,
   non-negligible safe-instability pattern on audited-valid pairs;
2. the pattern remains when score thresholds are selected solely on dev;
3. the harmful-change labels and safe transformations pass audit at a documented rate;
4. a source-aware treatment has a plausible mechanism that differs from B3/B4.

If B3/B4 remove the effect, record that negative result and revise the research
question rather than adding complexity. Any numerical effect-size/non-inferiority
threshold must be fixed after the pilot's label audit and before final test evaluation.
