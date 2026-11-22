# Manual audit guide for evidence transformations

Audit each original/variant pair without seeing a method prediction. Mark a short
rationale and one of `valid`, `invalid`, or `ambiguous` for the transformation.

## Safe transformations

- `permute`: all passages occur exactly once with unchanged text, source ID, and
  sentence IDs; only order changes.
- `duplicate`: copied text is exact and retains the original `source_document_id`.
  It is not an independent corroborating source.
- `resegment`: every original sentence and identifier occurs exactly once; text is not
  rewritten, deleted, or moved across source documents.

## Content-changing transformations

- `remove_gold`: decide whether the removed item is actually necessary for answering
  the question from remaining evidence. Mark invalid if an alternative evidence path
  remains.
- `inject_conflict`: confirm the new assertion addresses the same entity, relation,
  scope, and time. A merely different number or related topic is not enough.

## Labels for the resulting context

Record `sufficient` (yes/no/ambiguous), `conflict` (yes/no/ambiguous), audit status,
and a one- or two-sentence rationale. Keep automatic and reviewed labels separate.
