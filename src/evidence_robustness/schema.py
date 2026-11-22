"""Data contracts for evidence-assessment experiments.

The contracts deliberately keep labels separate from passages.  Labels may be used
for training/evaluation but are never part of the verifier input contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Split = Literal["train", "dev", "test"]
TransformationKind = Literal[
    "original", "permute", "duplicate", "resegment", "remove_gold", "inject_conflict", "add_distractor"
]
TransformationFamily = Literal["original", "safe", "content_changing"]


@dataclass(frozen=True)
class EvidencePassage:
    passage_id: str
    source_document_id: str
    text: str
    source_title: str | None = None
    sentence_ids: tuple[int, ...] | None = None
    sentence_texts: tuple[str, ...] | None = None
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        if not self.passage_id or not self.source_document_id:
            raise ValueError("passage_id and source_document_id must be non-empty")
        if not self.text.strip():
            raise ValueError("evidence passage text must be non-empty")
        if self.sentence_ids is not None and len(set(self.sentence_ids)) != len(self.sentence_ids):
            raise ValueError("sentence_ids must be unique within a passage")
        if self.sentence_texts is not None:
            if not self.sentence_texts or any(not sentence.strip() for sentence in self.sentence_texts):
                raise ValueError("sentence_texts must contain non-empty text")
            if self.sentence_ids is not None and len(self.sentence_ids) != len(self.sentence_texts):
                raise ValueError("sentence_ids and sentence_texts must have equal length")


@dataclass(frozen=True)
class EvidenceLabels:
    sufficient: int | None = None
    conflict: int | None = None
    review_status: Literal["auto", "reviewed", "ambiguous"] = "auto"
    label_rationale: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("sufficient", self.sufficient), ("conflict", self.conflict)):
            if value not in (0, 1, None):
                raise ValueError(f"{name} must be 0, 1, or None")


@dataclass(frozen=True)
class TransformationMetadata:
    kind: TransformationKind = "original"
    family: TransformationFamily = "original"
    random_seed: int = 0
    parent_example_id: str | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        expected = {
            "original": "original",
            "permute": "safe",
            "duplicate": "safe",
            "resegment": "safe",
            "remove_gold": "content_changing",
            "inject_conflict": "content_changing",
            "add_distractor": "content_changing",
        }[self.kind]
        if self.family != expected:
            raise ValueError(f"{self.kind} must have transformation family {expected}")


@dataclass(frozen=True)
class EvidenceExample:
    example_id: str
    parent_question_id: str
    split: Split
    question: str
    evidence: tuple[EvidencePassage, ...]
    labels: EvidenceLabels = field(default_factory=EvidenceLabels)
    transformation: TransformationMetadata = field(default_factory=TransformationMetadata)
    dataset_name: str = "synthetic"
    dataset_revision: str = "local"
    construction_metadata: dict[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.example_id or not self.parent_question_id or not self.question.strip():
            raise ValueError("example_id, parent_question_id, and question must be non-empty")
        if not self.evidence:
            raise ValueError("an example must contain at least one evidence passage")
        ids = [passage.passage_id for passage in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("passage_id values must be unique within an example")
        if self.transformation.kind == "original" and self.transformation.parent_example_id is not None:
            raise ValueError("original examples cannot have parent_example_id")
        if self.transformation.kind != "original" and not self.transformation.parent_example_id:
            raise ValueError("transformed examples must identify their parent_example_id")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation for manifests and smoke output."""
        return asdict(self)


@dataclass(frozen=True)
class VerifierOutput:
    sufficiency_score: float
    conflict_score: float
    uncertainty_score: float | None
    evidence_trace: tuple[dict[str, object], ...]
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        for name, value in (("sufficiency_score", self.sufficiency_score), ("conflict_score", self.conflict_score)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.uncertainty_score is not None and not 0.0 <= self.uncertainty_score <= 1.0:
            raise ValueError("uncertainty_score must be None or in [0, 1]")
