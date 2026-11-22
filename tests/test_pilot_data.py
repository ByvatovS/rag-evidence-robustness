from __future__ import annotations

from evidence_robustness.data import read_jsonl, write_jsonl
from evidence_robustness.pilot_data import build_pilot_perturbations, normalise_hotpot_row


def _hotpot_like_row() -> dict[str, object]:
    return {
        "id": "hotpot-like-1",
        "question": "Where did Ada work?",
        "answer": "Lab A",
        "context": {
            "title": ["Ada", "Lab A"],
            "sentences": [
                ["Ada worked at Lab A.", "Ada was a scientist."],
                ["Lab A is in City B.", "City B is on Earth."],
            ],
        },
        "supporting_facts": {"title": ["Ada", "Lab A"], "sent_id": [0, 0]},
    }


def test_normalisation_keeps_supporting_metadata_outside_evidence() -> None:
    example = normalise_hotpot_row(_hotpot_like_row(), seed=3, revision="test")
    assert example.labels.sufficient == 1
    assert example.evidence[0].sentence_texts == ("Ada worked at Lab A.", "Ada was a scientist.")
    assert example.construction_metadata["gold_answer"] == "Lab A"
    assert "Lab A" not in example.evidence[0].source_document_id


def test_json_roundtrip_and_pilot_variants(tmp_path) -> None:
    original = normalise_hotpot_row(_hotpot_like_row(), seed=3, revision="test")
    processed = tmp_path / "processed"
    perturbations = tmp_path / "perturbations"
    write_jsonl(processed / "pilot_originals.jsonl", [original])
    loaded = read_jsonl(processed / "pilot_originals.jsonl")
    assert loaded == [original]
    manifest = build_pilot_perturbations({"seed": 3}, processed, perturbations)
    assert manifest["safe_variants"] == 3
    assert manifest["content_changing_variants"] == 2
    kinds = {example.transformation.kind for example in read_jsonl(perturbations / "pilot_variants.jsonl")}
    assert {"permute", "duplicate", "resegment", "remove_gold", "inject_conflict"} <= kinds
