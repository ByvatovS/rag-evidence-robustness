"""Deterministic, provenance-preserving evidence transformations.

Only safe transformations are implemented here.  Content-changing variants require
dataset-specific validation and an audit, so they intentionally remain out of the
Stage-0 smoke path.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace

from .schema import EvidenceExample, EvidencePassage, TransformationMetadata

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _metadata(example: EvidenceExample, kind: str, seed: int) -> TransformationMetadata:
    return TransformationMetadata(
        kind=kind,  # type: ignore[arg-type]
        family="safe",
        random_seed=seed,
        parent_example_id=example.example_id,
        rationale="Deterministic Stage-0 safe transformation; text provenance is retained.",
    )


def _replace(example: EvidenceExample, kind: str, seed: int, evidence: tuple[EvidencePassage, ...]) -> EvidenceExample:
    return replace(
        example,
        example_id=f"{example.example_id}::{kind}",
        evidence=evidence,
        transformation=_metadata(example, kind, seed),
    )


def permute(example: EvidenceExample, seed: int) -> EvidenceExample:
    """Shuffle passages deterministically, retaining every passage unchanged."""
    passages = list(example.evidence)
    random.Random(seed).shuffle(passages)
    return _replace(example, "permute", seed, tuple(passages))


def duplicate(example: EvidenceExample, seed: int) -> EvidenceExample:
    """Copy one passage exactly; its source ID intentionally remains unchanged."""
    rng = random.Random(seed)
    selected = example.evidence[rng.randrange(len(example.evidence))]
    copy = replace(selected, passage_id=f"{selected.passage_id}::duplicate")
    return _replace(example, "duplicate", seed, (*example.evidence, copy))


def _sentences(passage: EvidencePassage) -> list[str]:
    if passage.sentence_texts is not None:
        return list(passage.sentence_texts)
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(passage.text.strip()) if part.strip()]
    if not parts:
        raise ValueError(f"could not segment passage {passage.passage_id}")
    return parts


def resegment(example: EvidenceExample, seed: int) -> EvidenceExample:
    """Split passages into sentence-aligned chunks without losing or rewriting text.

    The seed is recorded for uniform provenance even though chunk boundaries are
    deterministic. A smoke fixture supplies sentence IDs, enabling an exact audit.
    """
    segments: list[EvidencePassage] = []
    for passage in example.evidence:
        sentences = _sentences(passage)
        sentence_ids = passage.sentence_ids
        if sentence_ids is not None and len(sentence_ids) != len(sentences):
            raise ValueError(
                f"passage {passage.passage_id} has {len(sentences)} sentences but {len(sentence_ids)} sentence IDs"
            )
        # Pair adjacent sentences. This changes boundaries while retaining source identity.
        for offset in range(0, len(sentences), 2):
            ids = None if sentence_ids is None else tuple(sentence_ids[offset : offset + 2])
            segments.append(
                EvidencePassage(
                    passage_id=f"{passage.passage_id}::segment{offset // 2}",
                    source_document_id=passage.source_document_id,
                    source_title=passage.source_title,
                    text=" ".join(sentences[offset : offset + 2]),
                    sentence_ids=ids,
                    sentence_texts=tuple(sentences[offset : offset + 2]),
                    is_synthetic=passage.is_synthetic,
                )
            )
    return _replace(example, "resegment", seed, tuple(segments))


def normalised_sentences(example: EvidenceExample) -> tuple[str, ...]:
    """Expose sentence content for transformation validation, not verifier features."""
    return tuple(sentence for passage in example.evidence for sentence in _sentences(passage))


def all_sentence_ids(example: EvidenceExample) -> tuple[int, ...]:
    """Return IDs in evidence order; requires IDs to be present for every passage."""
    ids: list[int] = []
    for passage in example.evidence:
        if passage.sentence_ids is None:
            raise ValueError("cannot audit resegmentation without sentence IDs")
        ids.extend(passage.sentence_ids)
    return tuple(ids)
