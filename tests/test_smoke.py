from __future__ import annotations

from evidence_robustness.smoke import run_smoke


def test_smoke_is_offline_and_deterministic() -> None:
    first = run_smoke(count=10, seed=20261122)
    second = run_smoke(count=10, seed=20261122)
    assert first == second
    assert first["status"] == "ok"
    assert first["total_examples"] == 40
    assert first["network_used"] is False
