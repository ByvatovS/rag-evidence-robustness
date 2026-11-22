# Experiment log

This log records actual executions only. It is not evidence of model quality or
scientific novelty.

## 2026-11-22 — Stage-0 local CPU validation

- Environment observed: Windows host, Python 3.11.6, PyYAML 6.0.3.
- Installation command: `python -m pip install -e .`
- Test command: `python -m pytest -q`
- Test result after adding the offline pilot-data tests: **11 passed in 0.08 s**.
- Smoke command: `python -m evidence_robustness.cli smoke --config configs/smoke.yaml`
- Smoke result: `status=ok`, `original_examples=10`, `safe_variants=30`,
  `total_examples=40`, `seed=20261122`, `network_used=false`,
  `gpu_required=false`.
- Deterministic smoke artifact SHA256:
  `a0f46f645fedfd511075f6d2e3695014729c884b061e70643496f8cd283d308f`.
- Notebook JSON validation command:
  `python -c "import json, pathlib; [json.loads(path.read_text(encoding='utf-8')) for path in pathlib.Path('notebooks').glob('*.ipynb')]; print('notebooks: valid JSON')"`
- Notebook JSON validation result: `notebooks: valid JSON`.

## 2026-11-22 — literature risk check

- Consulted original HTML records for RINSE, SURE-RAG, and Sufficient Context.
- Outcome: RINSE is a direct `q + E` sufficiency comparator with paired contrast
  construction; SURE-RAG has an answer-conditioned contract and already uses
  set-level aggregation, shortcut checks, and counterfactual evidence conditions.
- No claim of complete reproduction, code availability, or novelty was made. See
  `docs/preliminary_literature_risk_assessment.md` and `docs/literature_matrix.csv`.

Not run in this environment: Google Colab, Google Drive mounting, HotpotQA download,
GPU inference, NLI, baseline fitting, or proposed-method evaluation.
