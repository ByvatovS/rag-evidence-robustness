from __future__ import annotations

import pytest

from evidence_robustness.schema import EvidenceExample, EvidencePassage
from evidence_robustness.smoke import make_synthetic_examples


def test_synthetic_example_satisfies_contract() -> None:
    example = make_synthetic_examples(1, 7)[0]
    assert example.parent_question_id == example.example_id
    assert example.transformation.kind == "original"


def test_empty_passage_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        EvidencePassage(passage_id="p", source_document_id="s", text="  ")


def test_transformed_example_requires_parent_reference() -> None:
    original = make_synthetic_examples(1, 7)[0]
    with pytest.raises(ValueError, match="parent_example_id"):
        EvidenceExample(
            example_id="bad",
            parent_question_id=original.parent_question_id,
            split=original.split,
            question=original.question,
            evidence=original.evidence,
            transformation=original.transformation.__class__(kind="permute", family="safe"),
        )
