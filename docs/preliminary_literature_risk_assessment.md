# Preliminary literature risk assessment — 2026-11-22

This is a decision memo based on the available original-paper HTML records. It is not
a claim that every implementation detail or code repository has been verified.

## High-risk overlap

RINSE is a close direct comparator: it judges sufficiency before generation using
question plus retrieved evidence, constructs paired sufficiency contrasts, and warns
that deletion and question/evidence pairing can expose lexical or positional
shortcuts. Its paper states that its three signals operate without an answer and are
combined into a lightweight pre-generation score.

- Source: [Chen et al., RINSE (2026)](https://arxiv.org/abs/2609.37469), especially
  the abstract and sections visible under its pre-generation formulation and paired
  contrast construction.

This means that **a paper whose only contribution is q+E sufficiency prediction or
paired deletion testing is unlikely to be novel enough.** The pilot must treat RINSE
as a direct baseline if compatible code/details are available, or state clearly why a
faithful reproduction is unavailable.

SURE-RAG is not a direct baseline for the target input contract, but it creates a
second overlap risk. It verifies `(question, candidate answer, evidence)`, aggregates
support/refute/neutral relation scores into set-level features, evaluates counterfactual
evidence conditions, runs shortcut checks, and uses group-disjoint question splits.

- Source: [Qiu et al., SURE-RAG (2026)](https://arxiv.org/abs/2605.03534), especially
  Sections I, IV, V, and VI available in the paper HTML.

Therefore, simply adding evidence aggregation, conflict scores, calibration, or
counterfactual evidence is not by itself a contribution. The input-contract distinction
must be maintained: SURE-RAG uses a candidate answer; the target verifier does not.

The ICLR paper on sufficient context establishes the importance of separating context
insufficiency from generator failure, but does not itself establish the requested
source-aware robustness protocol.

- Source: [Joren et al., Sufficient Context (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html).

## Narrow viable hypothesis to test

The defensible remaining niche is not generic sufficiency verification. It is:

> In a **candidate-answer-free**, pre-generation verifier with input only `q + E`, do
> same-fact changes to passage ordering, same-source duplication, and sentence-aligned
> resegmentation create decision instability after strong deduplication/order controls;
> and can provenance-aware aggregation reduce that instability without masking audited
> missing-evidence or conflicting-evidence cases at a measured budget?

The RINSE HTML record does not contain matches for `permutation`, `duplicate`, or
`segmentation`; SURE-RAG's visible counterfactual protocol changes evidence conditions
but also requires an answer. This is evidence of a possible gap, **not proof of novelty**.
It must be confirmed by reading complete methods, appendices, releases, and newer work.

## Go/no-go implication

Proceed through the baseline pilot, but do not commit to SBEV M1/M2 yet. Stop or reframe
if (1) the full audit finds that RINSE or another work already measures this exact
candidate-answer-free invariance protocol, or (2) exact same-source deduplication and
canonical ordering eliminate the instability. A positive pilot then needs a real-retrieval
or second-dataset check before supporting a publication claim.
