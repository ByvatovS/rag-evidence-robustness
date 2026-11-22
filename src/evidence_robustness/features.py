"""Inexpensive verifier features; these functions intentionally ignore all labels."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace

from .schema import EvidenceExample, EvidencePassage

_TOKEN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


def tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text) if len(token) > 1}


def evidence_view(example: EvidenceExample, mode: str) -> tuple[EvidencePassage, ...]:
    """Create raw, exact-same-source-deduplicated, or canonical evidence views."""
    passages = list(example.evidence)
    if mode == "raw":
        return tuple(passages)
    if mode == "dedup":
        seen: set[tuple[str, str]] = set()
        unique: list[EvidencePassage] = []
        for passage in passages:
            key = (passage.source_document_id, passage.text)
            if key not in seen:
                seen.add(key)
                unique.append(passage)
        return tuple(unique)
    if mode == "canonical":
        return tuple(sorted(passages, key=lambda passage: (passage.source_document_id, passage.passage_id)))
    raise ValueError(f"unknown evidence view: {mode}")


def feature_dict(example: EvidenceExample, mode: str) -> dict[str, float]:
    """Return no-oracle lexical and provenance features for a baseline classifier."""
    passages = evidence_view(example, mode)
    question_tokens = tokens(example.question)
    overlaps: list[float] = []
    for passage in passages:
        overlap = len(question_tokens & tokens(passage.text)) / max(len(question_tokens), 1)
        overlaps.append(overlap)
    source_counts = Counter(passage.source_document_id for passage in passages)
    text_counts = Counter(passage.text for passage in passages)
    return {
        "max_question_overlap": max(overlaps, default=0.0),
        "mean_question_overlap": sum(overlaps) / max(len(overlaps), 1),
        "passage_count": float(len(passages)),
        "source_count": float(len(source_counts)),
        "duplicate_passages": float(sum(count - 1 for count in text_counts.values() if count > 1)),
        "same_source_repeats": float(sum(count - 1 for count in source_counts.values() if count > 1)),
        "mean_passage_tokens": sum(len(tokens(passage.text)) for passage in passages) / max(len(passages), 1),
        "negation_count": float(sum(len(re.findall(r"\b(?:no|not|never|none|without)\b", passage.text.lower())) for passage in passages)),
        "synthetic_passage_count": float(sum(passage.is_synthetic for passage in passages)),
    }


def feature_names() -> tuple[str, ...]:
    return tuple(feature_dict(_minimal_example(), "raw").keys())


def _minimal_example() -> EvidenceExample:
    """Private fixed object used only to derive the feature column order."""
    from .schema import EvidenceExample, EvidencePassage

    return EvidenceExample(
        example_id="feature-schema",
        parent_question_id="feature-schema",
        split="train",
        question="placeholder question",
        evidence=(EvidencePassage("feature-schema-p", "feature-schema-s", "placeholder evidence."),),
    )
