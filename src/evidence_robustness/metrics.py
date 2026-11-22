"""Paired pilot metrics grouped by original question rather than by variant."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def paired_robustness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise predictions emitted by the diagnostic pilot.

    This is descriptive screening output. Publication confidence intervals are added
    only after labels and protocol are frozen.
    """
    by_model_parent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model_parent[(row["model"], row["parent_question_id"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for (model, _parent), group in by_model_parent.items():
        summary = result.setdefault(model, {"safe_pairs": 0, "safe_flips": 0, "harmful_remove_gold": 0, "remove_gold_detected": 0, "harmful_conflicts": 0, "conflicts_detected": 0})
        original = next((row for row in group if row["transformation_kind"] == "original"), None)
        if original is None:
            continue
        original_decision = original["sufficiency_prediction"]
        for row in group:
            kind = row["transformation_kind"]
            if row["transformation_family"] == "safe":
                summary["safe_pairs"] += 1
                summary["safe_flips"] += int(row["sufficiency_prediction"] != original_decision)
            elif kind == "remove_gold":
                summary["harmful_remove_gold"] += 1
                summary["remove_gold_detected"] += int(row["sufficiency_prediction"] == 0)
            elif kind == "inject_conflict":
                summary["harmful_conflicts"] += 1
                summary["conflicts_detected"] += int(row["conflict_prediction"] == 1)
    for summary in result.values():
        summary["flip_rate_safe"] = summary["safe_flips"] / max(summary["safe_pairs"], 1)
        summary["recall_remove_gold_weak_label"] = summary["remove_gold_detected"] / max(summary["harmful_remove_gold"], 1)
        summary["recall_conflict_weak_label"] = summary["conflicts_detected"] / max(summary["harmful_conflicts"], 1)
    return result
