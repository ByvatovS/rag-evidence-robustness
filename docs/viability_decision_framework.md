# What would justify continuing this topic?

This project does not assume a positive answer. A justified go/no-go decision needs
two independent kinds of evidence.

## Literature viability

The claim remains viable only if a primary-source audit shows that no prior work has
already established the same combination of: (a) pre-generation `q + E` input,
(b) source-aware handling of repeated evidence, (c) paired fact-preserving and
fact-changing interventions, and (d) a robustness/sensitivity/cost comparison against
strong simple controls. Partial overlap is expected and must be acknowledged.

## Empirical viability

The pilot should show an audited-valid, statistically quantified failure mode for at
least one credible baseline after exact deduplication and order controls. A proposed
method is justified only if it improves safe-pair stability without simply becoming
insensitive to removal/conflict, and if its compute cost is reported.

## Outcomes

| Observation | Decision |
| --- | --- |
| Directly overlapping work already provides the same protocol and result | Reframe the question or stop this line. |
| Transformations or labels fail audit | Improve data construction; do not make model claims. |
| B3/B4 solve the issue | Publish/report a negative diagnostic if useful; do not add SBEV complexity. |
| Baseline failure is reproducible and SBEV improves the trade-off | Proceed to ablations and external validation. |
| Effect only appears on one synthetic setting | Treat as preliminary; require a retrieval or second-data check. |
