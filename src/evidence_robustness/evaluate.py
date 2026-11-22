"""Fit and evaluate the diagnostic, pre-proposed-method pilot baselines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .baselines import BaselineBundle, fit_diagnostic_baselines
from .data import read_jsonl
from .metrics import paired_robustness_summary


def _require_joblib_and_metrics() -> tuple[Any, Any]:
    try:
        import joblib
        from sklearn.metrics import f1_score
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Pilot commands require `pip install -e '.[pilot]'`.") from error
    return joblib, f1_score


def _all_examples(workdir: Path) -> list[Any]:
    originals = read_jsonl(workdir / "processed" / "pilot_originals.jsonl")
    variants = read_jsonl(workdir / "perturbations" / "pilot_variants.jsonl")
    return [*originals, *variants]


def fit_baselines(workdir: Path) -> dict[str, Any]:
    joblib, _ = _require_joblib_and_metrics()
    examples = _all_examples(workdir)
    train = [example for example in examples if example.split == "train"]
    dev = [example for example in examples if example.split == "dev"]
    if not train or not dev:
        raise ValueError("pilot needs non-empty train and dev splits; increase sample size")
    bundles = fit_diagnostic_baselines(train, dev)
    output = workdir / "models" / "diagnostic_baselines.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundles, output)
    metadata = {
        "train_examples": len(train),
        "dev_examples": len(dev),
        "models": sorted(bundles),
        "b0_threshold": bundles["B1"].b0_sufficiency_threshold,
        "warning": "All labels are weak labels until the manual audit is completed.",
    }
    (workdir / "models" / "diagnostic_baselines_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def _prediction_row(model: str, example: Any, sufficiency_score: float, conflict_score: float, threshold: float, elapsed_ms: float) -> dict[str, Any]:
    return {
        "model": model,
        "example_id": example.example_id,
        "parent_question_id": example.parent_question_id,
        "split": example.split,
        "transformation_kind": example.transformation.kind,
        "transformation_family": example.transformation.family,
        "sufficiency_label": example.labels.sufficient,
        "conflict_label": example.labels.conflict,
        "review_status": example.labels.review_status,
        "sufficiency_score": sufficiency_score,
        "conflict_score": conflict_score,
        "sufficiency_prediction": int(sufficiency_score >= threshold),
        "conflict_prediction": int(conflict_score >= 0.5),
        "threshold": threshold,
        "wall_time_ms": elapsed_ms,
        "nli_calls": 0,
    }


def evaluate_baselines(workdir: Path) -> dict[str, Any]:
    joblib, f1_score = _require_joblib_and_metrics()
    model_path = workdir / "models" / "diagnostic_baselines.joblib"
    if not model_path.is_file():
        raise FileNotFoundError("Run `fit-baselines` before `evaluate`.")
    bundles: dict[str, BaselineBundle] = joblib.load(model_path)
    test_examples = [example for example in _all_examples(workdir) if example.split == "test"]
    if not test_examples:
        raise ValueError("pilot has no test examples; increase sample size")
    rows: list[dict[str, Any]] = []
    for example in test_examples:
        # B0 is a retrieval-overlap baseline; it deliberately has no conflict detector.
        b0_start = time.perf_counter()
        _suff, _conflict, b0_score = bundles["B1"].scores(example)
        rows.append(
            _prediction_row(
                "B0", example, b0_score, 0.0, bundles["B1"].b0_sufficiency_threshold, (time.perf_counter() - b0_start) * 1000
            )
        )
        for name, bundle in bundles.items():
            start = time.perf_counter()
            sufficiency, conflict, _ = bundle.scores(example)
            rows.append(
                _prediction_row(
                    name,
                    example,
                    sufficiency,
                    conflict,
                    0.5,
                    (time.perf_counter() - start) * 1000,
                )
            )
    quality: dict[str, dict[str, float | None]] = {}
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        suff = [row for row in model_rows if row["sufficiency_label"] is not None]
        conflict = [row for row in model_rows if row["conflict_label"] is not None]
        quality[model] = {
            "sufficiency_macro_f1_weak_label": float(
                f1_score([row["sufficiency_label"] for row in suff], [row["sufficiency_prediction"] for row in suff], average="macro", zero_division=0)
            ) if suff else None,
            "conflict_macro_f1_weak_label": float(
                f1_score([row["conflict_label"] for row in conflict], [row["conflict_prediction"] for row in conflict], average="macro", zero_division=0)
            ) if conflict else None,
            "median_wall_time_ms": sorted(row["wall_time_ms"] for row in model_rows)[len(model_rows) // 2],
        }
    summary = {
        "split": "test",
        "test_examples": len(test_examples),
        "quality_weak_labels": quality,
        "paired_weak_label_summary": paired_robustness_summary(rows),
        "warnings": [
            "Results use automatically constructed weak labels until audited.",
            "Synthetic conflict evidence is a stress test, not a natural-conflict benchmark.",
            "This pilot supplies descriptive statistics; do not claim publication-grade significance yet.",
        ],
    }
    reports = workdir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "pilot_predictions_test.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    (reports / "pilot_baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
