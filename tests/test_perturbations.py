from __future__ import annotations

from evidence_robustness.perturbations import all_sentence_ids, duplicate, normalised_sentences, permute, resegment
from evidence_robustness.smoke import make_synthetic_examples


def test_permute_preserves_passages_and_records_parent() -> None:
    original = make_synthetic_examples(1, 12)[0]
    transformed = permute(original, 101)
    assert {passage.passage_id for passage in transformed.evidence} == {
        passage.passage_id for passage in original.evidence
    }
    assert transformed.transformation.parent_example_id == original.example_id
    assert transformed.transformation.random_seed == 101


def test_duplicate_preserves_source_provenance() -> None:
    original = make_synthetic_examples(1, 12)[0]
    transformed = duplicate(original, 101)
    copied = transformed.evidence[-1]
    assert copied.passage_id.endswith("::duplicate")
    assert copied.source_document_id in {passage.source_document_id for passage in original.evidence}
    assert copied.text in {passage.text for passage in original.evidence}


def test_resegment_preserves_every_sentence_and_id() -> None:
    original = make_synthetic_examples(1, 12)[0]
    transformed = resegment(original, 101)
    assert normalised_sentences(transformed) == normalised_sentences(original)
    assert all_sentence_ids(transformed) == all_sentence_ids(original)
    assert {passage.source_document_id for passage in transformed.evidence} == {
        passage.source_document_id for passage in original.evidence
    }
