"""Cheap diagnostic baselines used before any proposed verifier is attempted."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .features import feature_dict, feature_names
from .schema import EvidenceExample


def _require_sklearn() -> Any:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Baseline commands require `pip install -e '.[pilot]'`.") from error
    return LogisticRegression


def _matrix(examples: list[EvidenceExample], mode: str) -> list[list[float]]:
    columns = feature_names()
    return [[feature_dict(example, mode)[column] for column in columns] for example in examples]


@dataclass
class ConstantClassifier:
    value: int

    def predict_proba(self, matrix: list[list[float]]) -> list[list[float]]:
        return [[1.0 - self.value, float(self.value)] for _ in matrix]


@dataclass
class BaselineBundle:
    mode: str
    sufficiency_model: Any
    conflict_model: Any
    b0_sufficiency_threshold: float

    def scores(self, example: EvidenceExample) -> tuple[float, float, float]:
        row = _matrix([example], self.mode)
        sufficiency = float(self.sufficiency_model.predict_proba(row)[0][1])
        conflict = float(self.conflict_model.predict_proba(row)[0][1])
        b0_score = feature_dict(example, self.mode)["max_question_overlap"]
        return sufficiency, conflict, b0_score


def _fit_outcome(examples: list[EvidenceExample], mode: str, outcome: str) -> Any:
    LogisticRegression = _require_sklearn()
    labelled = [example for example in examples if getattr(example.labels, outcome) is not None]
    labels = [int(getattr(example.labels, outcome)) for example in labelled]
    if not labelled:
        return ConstantClassifier(0)
    if len(set(labels)) == 1:
        return ConstantClassifier(labels[0])
    return LogisticRegression(max_iter=500, random_state=0, class_weight="balanced").fit(_matrix(labelled, mode), labels)


def _best_b0_threshold(dev_examples: list[EvidenceExample], mode: str) -> float:
    try:
        from sklearn.metrics import f1_score
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Baseline commands require `pip install -e '.[pilot]'`.") from error
    labelled = [example for example in dev_examples if example.labels.sufficient is not None]
    if not labelled:
        return 0.5
    scores = [feature_dict(example, mode)["max_question_overlap"] for example in labelled]
    candidates = sorted({0.0, 1.0, *scores})
    labels = [int(example.labels.sufficient) for example in labelled]
    return max(candidates, key=lambda threshold: f1_score(labels, [score >= threshold for score in scores], zero_division=0))


def fit_diagnostic_baselines(train_examples: list[EvidenceExample], dev_examples: list[EvidenceExample]) -> dict[str, BaselineBundle]:
    """Fit B1/B3/B4 classifiers and choose B0's threshold only on development data."""
    bundles: dict[str, BaselineBundle] = {}
    for name, mode in (("B1", "raw"), ("B3", "dedup"), ("B4", "canonical")):
        bundles[name] = BaselineBundle(
            mode=mode,
            sufficiency_model=_fit_outcome(train_examples, mode, "sufficient"),
            conflict_model=_fit_outcome(train_examples, mode, "conflict"),
            b0_sufficiency_threshold=_best_b0_threshold(dev_examples, mode),
        )
    return bundles
