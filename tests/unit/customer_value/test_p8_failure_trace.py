"""Safe originating trace capture; synthetic failures, development metadata and temp paths only."""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from decision_intelligence.common.failure_trace import TracedTaskError, failure_trace
from decision_intelligence.customer_value import predictive_cases as pc


def _script() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "generate_m11_evidence.py"
    spec = importlib.util.spec_from_file_location("generate_m11_evidence_trace", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


GEN = _script()
SECRET = "synthetic-outcome-must-not-be-logged"


def _boom(seed: int) -> dict[str, Any]:
    secret_local = SECRET
    try:
        raise ZeroDivisionError(secret_local)
    except ZeroDivisionError as exc:
        raise ValueError(secret_local) from exc


def test_chained_locations_exclude_messages_locals_and_source(monkeypatch: pytest.MonkeyPatch
                                                             ) -> None:
    monkeypatch.setattr(pc, "r2_draw", _boom)
    with pytest.raises(TracedTaskError) as caught:
        pc.run_task(pc.Task("R2", 7001))
    # Round-trip only this synthetic exception, never an evidence pickle.
    transported = pickle.loads(pickle.dumps(caught.value))
    trace = failure_trace(transported)
    assert [e["type"] for e in trace] == ["TracedTaskError", "ValueError", "ZeroDivisionError"]
    assert trace[1]["relation"] == "cause"
    assert trace[-1]["relation"] == "cause"
    assert any(f["function"] == "_boom" for e in trace for f in e["frames"])
    assert SECRET not in json.dumps(trace) + str(transported)
    assert "secret_local" not in json.dumps(trace)
    assert all(set(f) == {"file", "line", "function"} for e in trace for f in e["frames"])


def test_actual_spawn_transport_retains_origin_without_remote_text() -> None:
    # Invalid study fails before any scenario/RNG use; no final-mode execution or reservation.
    with pytest.raises(TracedTaskError) as caught:
        GEN.run_tasks([pc.Task("INVALID_TRACE_FIXTURE", 7002)], 2)
    trace = failure_trace(caught.value)
    assert [e["type"] for e in trace] == ["TracedTaskError", "ValueError"]
    assert any(f["file"] == "predictive_cases.py" and f["function"] == "_dispatch"
               for e in trace for f in e["frames"])
    assert "RemoteTraceback" not in json.dumps(trace)
    assert "unknown study" not in json.dumps(trace) + str(caught.value)


def test_failure_ledger_and_cli_do_not_log_exception_payloads(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
        ) -> None:
    def fail(task: pc.Task) -> dict[str, Any]:
        return _boom(task.seed)

    with pytest.raises(ValueError):
        GEN.execute(GEN.dry_run_plan(draws=1, refits=1), tmp_path / "run", 1, fn=fail,
                    registry=tmp_path / "registry", evidence_root=tmp_path)
    text = (tmp_path / "run" / "ledger.json").read_text()
    ledger = json.loads(text)
    assert ledger["status"] == "failed" and ledger["phase"] == "main"
    assert ledger["error"] == "ValueError"
    assert [e["type"] for e in ledger["failure_trace"]] == ["ValueError", "ZeroDivisionError"]
    assert SECRET not in text and not (tmp_path / "run" / "raw").exists()

    def fail_execute(*args: Any, **kwargs: Any) -> Path:
        raise ValueError(SECRET)

    monkeypatch.setattr(GEN, "execute", fail_execute)
    assert GEN.main(["--mode", "dry-run"]) == 1
    assert SECRET not in capsys.readouterr().err


def test_suppressed_context_and_cyclic_chain_are_safe() -> None:
    try:
        try:
            raise ValueError(SECRET)
        except ValueError:
            raise RuntimeError(SECRET) from None
    except RuntimeError as exc:
        assert [e["type"] for e in failure_trace(exc)] == ["RuntimeError"]
        exc.__cause__ = exc
        assert len(failure_trace(exc)) == 1
