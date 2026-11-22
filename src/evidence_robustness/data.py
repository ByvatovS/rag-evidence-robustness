"""Small deterministic data utilities that are safe to use in tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from .schema import EvidenceExample, EvidenceLabels, EvidencePassage, TransformationMetadata


def deterministic_split(parent_question_id: str, seed: int = 20261122) -> str:
    """Assign a parent question to train/dev/test without depending on input order."""
    digest = hashlib.sha256(f"{seed}:{parent_question_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "dev"
    return "test"


def assert_grouped_splits(examples: Iterable[EvidenceExample]) -> None:
    """Raise when variants of one original question appear in different splits."""
    parent_to_split: dict[str, str] = {}
    for example in examples:
        prior = parent_to_split.setdefault(example.parent_question_id, example.split)
        if prior != example.split:
            raise ValueError(
                f"split leakage: parent_question_id {example.parent_question_id!r} occurs in {prior!r} and {example.split!r}"
            )


def example_from_dict(payload: dict[str, object]) -> EvidenceExample:
    """Load an example written by :meth:`EvidenceExample.to_dict` from JSONL."""
    evidence = tuple(
        EvidencePassage(
            **{
                **passage,
                "sentence_ids": None
                if passage.get("sentence_ids") is None
                else tuple(sentence_id for sentence_id in passage["sentence_ids"]),
                "sentence_texts": None
                if passage.get("sentence_texts") is None
                else tuple(str(sentence) for sentence in passage["sentence_texts"]),
            }
        )
        for raw_passage in payload["evidence"]  # type: ignore[index]
        for passage in [dict(raw_passage)]
    )
    labels = EvidenceLabels(**dict(payload.get("labels", {})))
    transformation = TransformationMetadata(**dict(payload.get("transformation", {})))
    return EvidenceExample(
        example_id=str(payload["example_id"]),
        parent_question_id=str(payload["parent_question_id"]),
        split=str(payload["split"]),  # type: ignore[arg-type]
        question=str(payload["question"]),
        evidence=evidence,
        labels=labels,
        transformation=transformation,
        dataset_name=str(payload.get("dataset_name", "unknown")),
        dataset_revision=str(payload.get("dataset_revision", "unknown")),
        construction_metadata=dict(payload.get("construction_metadata", {})),
    )


def write_jsonl(path: Path, examples: Iterable[EvidenceExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True, default=list))
            handle.write("\n")


def read_jsonl(path: Path) -> list[EvidenceExample]:
    with path.open(encoding="utf-8") as handle:
        return [example_from_dict(json.loads(line)) for line in handle if line.strip()]
