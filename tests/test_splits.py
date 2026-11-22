from __future__ import annotations

import pytest

from evidence_robustness.data import assert_grouped_splits, deterministic_split
from evidence_robustness.smoke import make_synthetic_examples


def test_split_assignment_is_order_independent_and_deterministic() -> None:
    ids = [f"question-{number}" for number in range(100)]
    first = {item: deterministic_split(item, 9) for item in ids}
    second = {item: deterministic_split(item, 9) for item in reversed(ids)}
    assert first == second
    assert set(first.values()) == {"train", "dev", "test"}


def test_grouped_split_leakage_is_rejected() -> None:
    example = make_synthetic_examples(1, 7)[0]
    leaked = example.__class__(
        example_id="leaked",
        parent_question_id=example.parent_question_id,
        split="test" if example.split != "test" else "train",
        question=example.question,
        evidence=example.evidence,
    )
    with pytest.raises(ValueError, match="split leakage"):
        assert_grouped_splits([example, leaked])
