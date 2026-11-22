"""HotpotQA normalisation and auditable perturbation construction for the pilot.

This module is intentionally separated from verifier features: construction metadata
contains weak-label material and must never enter a model input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .data import deterministic_split, read_jsonl, write_jsonl
from .perturbations import duplicate, permute, resegment
from .schema import EvidenceExample, EvidenceLabels, EvidencePassage, TransformationMetadata


def _require_datasets() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - depends on optional Colab extra
        raise RuntimeError("Pilot commands require `pip install -e '.[pilot]'`.") from error
    return load_dataset


def _stable_sample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    def rank(row: dict[str, Any]) -> str:
        identifier = str(row.get("id", ""))
        return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()

    return sorted(rows, key=rank)[:limit]


def _context_pairs(context: Any) -> list[tuple[str, list[str]]]:
    """Handle the documented HotpotQA context mapping and fail loudly otherwise."""
    if isinstance(context, dict):
        titles = context.get("title", context.get("titles"))
        sentences = context.get("sentences")
        if isinstance(titles, list) and isinstance(sentences, list) and len(titles) == len(sentences):
            return [(str(title), [str(sentence) for sentence in document]) for title, document in zip(titles, sentences)]
    if isinstance(context, list):
        pairs: list[tuple[str, list[str]]] = []
        for item in context:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                break
            pairs.append((str(item[0]), [str(sentence) for sentence in item[1]]))
        if pairs:
            return pairs
    raise ValueError(f"Unsupported HotpotQA context structure: {type(context)!r}")


def _supporting_facts(value: Any) -> list[tuple[str, int]]:
    if not isinstance(value, dict):
        return []
    titles = value.get("title", value.get("titles", []))
    sent_ids = value.get("sent_id", value.get("sentence_ids", []))
    if not isinstance(titles, list) or not isinstance(sent_ids, list):
        return []
    return [(str(title), int(sentence_id)) for title, sentence_id in zip(titles, sent_ids)]


def normalise_hotpot_row(row: dict[str, Any], *, seed: int, revision: str) -> EvidenceExample:
    question_id = str(row["id"])
    passages: list[EvidencePassage] = []
    for position, (title, sentences) in enumerate(_context_pairs(row["context"])):
        # HotpotQA occasionally has an empty sentence slot. Preserve original sentence
        # indices for supporting-fact provenance while excluding empty text from a
        # passage contract that rightly forbids empty sentences.
        indexed_sentences = [(index, sentence.strip()) for index, sentence in enumerate(sentences) if sentence.strip()]
        if not indexed_sentences:
            continue
        sentence_ids, sentence_texts = zip(*indexed_sentences, strict=True)
        passages.append(
            EvidencePassage(
                passage_id=f"{question_id}::doc{position}",
                source_document_id=title,
                source_title=title,
                text=" ".join(sentence_texts),
                sentence_ids=tuple(sentence_ids),
                sentence_texts=tuple(sentence_texts),
            )
        )
    if not passages:
        raise ValueError(f"No non-empty context passages in {question_id}")
    supporting_facts = _supporting_facts(row.get("supporting_facts"))
    return EvidenceExample(
        example_id=question_id,
        parent_question_id=question_id,
        split=deterministic_split(question_id, seed),  # type: ignore[arg-type]
        question=str(row["question"]),
        evidence=tuple(passages),
        labels=EvidenceLabels(
            sufficient=1,
            conflict=0,
            review_status="auto",
            label_rationale="Weak label: original HotpotQA distractor context; requires audit.",
        ),
        transformation=TransformationMetadata(random_seed=seed),
        dataset_name="hotpotqa/hotpot_qa:distractor",
        dataset_revision=revision,
        # This construction-only material must never be passed to a verifier.
        construction_metadata={
            "supporting_facts": [{"title": title, "sentence_id": sentence_id} for title, sentence_id in supporting_facts],
            "gold_answer": str(row.get("answer", "")),
        },
    )


def prepare_hotpot_pilot(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Download and normalise a deterministic small HotpotQA sample in Colab."""
    load_dataset = _require_datasets()
    dataset_config = config["dataset"]
    seed = int(config["seed"])
    revision = dataset_config.get("revision")
    dataset = load_dataset(
        dataset_config["name"], dataset_config["config"], split="train", revision=revision
    )
    rows = _stable_sample([dict(row) for row in dataset], int(dataset_config["max_original_questions"]), seed)
    effective_revision = str(revision or getattr(dataset, "_fingerprint", "unversioned"))
    examples = [normalise_hotpot_row(row, seed=seed, revision=effective_revision) for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "pilot_originals.jsonl"
    write_jsonl(examples_path, examples)
    source_ids = [example.example_id for example in examples]
    manifest = {
        "dataset": dataset_config["name"],
        "config": dataset_config["config"],
        "requested_revision": revision,
        "effective_revision_or_fingerprint": effective_revision,
        "seed": seed,
        "count": len(examples),
        "selected_id_sha256": hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest(),
        "split_counts": {
            split: sum(example.split == split for example in examples) for split in ("train", "dev", "test")
        },
        "warning": "Original sufficiency/conflict labels are weak labels and require audit.",
    }
    (output_dir / "pilot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _remove_supporting_sentence(example: EvidenceExample, seed: int) -> EvidenceExample | None:
    supports = example.construction_metadata.get("supporting_facts", [])
    if not isinstance(supports, list):
        return None
    for support in supports:
        if not isinstance(support, dict):
            continue
        title, sentence_id = str(support.get("title", "")), support.get("sentence_id")
        if not isinstance(sentence_id, int):
            continue
        for index, passage in enumerate(example.evidence):
            if passage.source_document_id != title or passage.sentence_ids is None or passage.sentence_texts is None:
                continue
            if sentence_id not in passage.sentence_ids or len(passage.sentence_ids) <= 1:
                continue
            position = passage.sentence_ids.index(sentence_id)
            new_ids = tuple(value for offset, value in enumerate(passage.sentence_ids) if offset != position)
            new_sentences = tuple(value for offset, value in enumerate(passage.sentence_texts) if offset != position)
            new_passage = replace(passage, text=" ".join(new_sentences), sentence_ids=new_ids, sentence_texts=new_sentences)
            evidence = list(example.evidence)
            evidence[index] = new_passage
            return replace(
                example,
                example_id=f"{example.example_id}::remove_gold",
                evidence=tuple(evidence),
                labels=EvidenceLabels(
                    sufficient=0,
                    conflict=example.labels.conflict,
                    review_status="auto",
                    label_rationale="Weak label: one HotpotQA supporting sentence removed; audit for alternative evidence paths.",
                ),
                transformation=TransformationMetadata(
                    kind="remove_gold",
                    family="content_changing",
                    random_seed=seed,
                    parent_example_id=example.example_id,
                    rationale=f"Removed supporting sentence {sentence_id} from source {title}; validity is unreviewed.",
                ),
                construction_metadata={**example.construction_metadata, "removed_supporting_fact": support},
            )
    return None


def _inject_conflict(example: EvidenceExample, seed: int) -> EvidenceExample | None:
    answer = str(example.construction_metadata.get("gold_answer", "")).strip()
    if not answer:
        return None
    conflict_text = (
        f"A synthetic conflicting source explicitly asserts that the answer to the question "
        f"'{example.question}' is not '{answer}'."
    )
    conflict = EvidencePassage(
        passage_id=f"{example.example_id}::synthetic_conflict",
        source_document_id=f"synthetic-conflict::{example.example_id}",
        source_title="Synthetic conflicting source",
        text=conflict_text,
        sentence_ids=(0,),
        sentence_texts=(conflict_text,),
        is_synthetic=True,
    )
    return replace(
        example,
        example_id=f"{example.example_id}::inject_conflict",
        evidence=(*example.evidence, conflict),
        labels=EvidenceLabels(
            sufficient=example.labels.sufficient,
            conflict=1,
            review_status="auto",
            label_rationale="Weak synthetic conflict label; manual audit required for entity/relation/time validity.",
        ),
        transformation=TransformationMetadata(
            kind="inject_conflict",
            family="content_changing",
            random_seed=seed,
            parent_example_id=example.example_id,
            rationale="Added a synthetic explicitly incompatible answer claim; unreviewed.",
        ),
    )


def build_pilot_perturbations(config: dict[str, Any], input_dir: Path, output_dir: Path) -> dict[str, Any]:
    seed = int(config["seed"])
    originals = read_jsonl(input_dir / "pilot_originals.jsonl")
    variants: list[EvidenceExample] = []
    skipped_remove_gold = 0
    skipped_conflict = 0
    for index, original in enumerate(originals):
        local_seed = seed + index
        variants.extend([permute(original, local_seed), duplicate(original, local_seed), resegment(original, local_seed)])
        removed = _remove_supporting_sentence(original, local_seed)
        conflict = _inject_conflict(original, local_seed)
        if removed is None:
            skipped_remove_gold += 1
        else:
            variants.append(removed)
        if conflict is None:
            skipped_conflict += 1
        else:
            variants.append(conflict)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "pilot_variants.jsonl"
    write_jsonl(output, variants)
    manifest = {
        "seed": seed,
        "originals": len(originals),
        "variants": len(variants),
        "safe_variants": sum(example.transformation.family == "safe" for example in variants),
        "content_changing_variants": sum(example.transformation.family == "content_changing" for example in variants),
        "skipped_remove_gold": skipped_remove_gold,
        "skipped_inject_conflict": skipped_conflict,
        "warning": "Content-changing labels are provisional and must be audited before publication claims.",
    }
    (output_dir / "perturbation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
