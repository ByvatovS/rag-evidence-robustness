"""Offline synthetic data and invariant checks for the first project milestone."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from .data import assert_grouped_splits, deterministic_split
from .perturbations import all_sentence_ids, duplicate, normalised_sentences, permute, resegment
from .schema import EvidenceExample, EvidenceLabels, EvidencePassage, TransformationMetadata


def make_synthetic_examples(count: int, seed: int) -> list[EvidenceExample]:
    if count <= 0:
        raise ValueError("examples must be positive")
    examples: list[EvidenceExample] = []
    for number in range(count):
        parent_id = f"synthetic-{number:03d}"
        evidence = (
            EvidencePassage(
                passage_id=f"{parent_id}-a",
                source_document_id=f"source-{number:03d}-a",
                source_title=f"Synthetic source A {number}",
                text=f"Entity {number} has property alpha. Entity {number} is in group beta.",
                sentence_ids=(number * 10, number * 10 + 1),
                sentence_texts=(f"Entity {number} has property alpha.", f"Entity {number} is in group beta."),
                is_synthetic=True,
            ),
            EvidencePassage(
                passage_id=f"{parent_id}-b",
                source_document_id=f"source-{number:03d}-b",
                source_title=f"Synthetic source B {number}",
                text=f"Group beta is connected to region gamma. Region gamma was recorded in year {2000 + number}.",
                sentence_ids=(number * 10 + 2, number * 10 + 3),
                sentence_texts=(
                    f"Group beta is connected to region gamma.",
                    f"Region gamma was recorded in year {2000 + number}.",
                ),
                is_synthetic=True,
            ),
        )
        examples.append(
            EvidenceExample(
                example_id=parent_id,
                parent_question_id=parent_id,
                split=deterministic_split(parent_id, seed),  # type: ignore[arg-type]
                question=f"Which region is linked to entity {number}?",
                evidence=evidence,
                labels=EvidenceLabels(sufficient=1, conflict=0, review_status="auto"),
                transformation=TransformationMetadata(random_seed=seed),
                dataset_name="synthetic-smoke",
                dataset_revision="v1",
            )
        )
    return examples


def _canonical_payload(examples: list[EvidenceExample]) -> str:
    return json.dumps([example.to_dict() for example in examples], ensure_ascii=False, sort_keys=True, default=list)


def run_smoke(count: int, seed: int) -> dict[str, object]:
    originals = make_synthetic_examples(count, seed)
    variants: list[EvidenceExample] = []
    for index, original in enumerate(originals):
        transformation_seed = seed + index
        variants.extend(
            [
                permute(original, transformation_seed),
                duplicate(original, transformation_seed),
                resegment(replace(original, transformation=TransformationMetadata(random_seed=transformation_seed)), transformation_seed),
            ]
        )
    assert_grouped_splits([*originals, *variants])
    for original, offset in zip(originals, range(0, len(variants), 3), strict=True):
        permuted, duplicated, resegmented = variants[offset : offset + 3]
        if set(p.passage_id for p in original.evidence) != set(p.passage_id for p in permuted.evidence):
            raise AssertionError("permutation changed passage provenance")
        if duplicated.evidence[-1].source_document_id not in {p.source_document_id for p in original.evidence}:
            raise AssertionError("duplicate was incorrectly made into an independent source")
        if normalised_sentences(original) != normalised_sentences(resegmented):
            raise AssertionError("resegmentation changed sentence text")
        if all_sentence_ids(original) != all_sentence_ids(resegmented):
            raise AssertionError("resegmentation changed sentence IDs")
    payload = _canonical_payload([*originals, *variants])
    return {
        "status": "ok",
        "seed": seed,
        "original_examples": len(originals),
        "safe_variants": len(variants),
        "total_examples": len(originals) + len(variants),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "network_used": False,
        "gpu_required": False,
    }
