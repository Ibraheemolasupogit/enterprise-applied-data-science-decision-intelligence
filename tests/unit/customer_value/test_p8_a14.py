"""M11 P8, proposed amendment A14: the zero-dropout limit in E4's outcome simulation, the
small-a fallback of the accelerated E4 rule, non-finite inputs, task-identified failures and
the one authorised new attempt (``final-a14``). Development seeds, fake draws and temporary
registries only; no reserved seed is generated or used."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from decision_intelligence.customer_value import btyd
from decision_intelligence.customer_value import predictive_cases as pc
from decision_intelligence.customer_value import predictive_evaluation as ev
from decision_intelligence.customer_value import predictive_protocol as pp

REPO = Path(__file__).resolve().parents[3]


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_m11_evidence_a14", REPO / "scripts" / "generate_m11_evidence.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_m11_evidence_a14"] = mod
    spec.loader.exec_module(mod)
    return mod


GEN = _script()


# --- zero-dropout limit ---------------------------------------------------------------------
def test_zero_dropout_probability_is_the_no_dropout_limit_and_keeps_the_stream() -> None:
    p = np.array([0.4, 0.0, 0.02, 1.0, 0.0, 1e-310])
    k = ev.dropout_purchase_limit(p, np.random.default_rng(6701))
    assert k[1] == k[4] == ev.NO_DROPOUT == np.iinfo(np.int64).max
    ref = np.random.default_rng(6701).geometric(np.where(p == 0.0, 1.0, p))
    keep = p > 0.0
    np.testing.assert_array_equal(k[keep], ref[keep])  # every positive-p draw unchanged
    assert k[5] == np.iinfo(np.int64).max  # NumPy's own value for tiny positive p
    a, b = np.random.default_rng(6702), np.random.default_rng(6702)
    ev.dropout_purchase_limit(p, a)
    b.geometric(np.full(p.size, 0.5))
    assert a.uniform() == b.uniform()  # the stream position is unchanged


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_dropout_probabilities_raise(bad: float) -> None:
    with pytest.raises(ValueError, match="outside"):
        ev.dropout_purchase_limit(np.array([0.3, bad]), np.random.default_rng(6703))


@pytest.fixture(scope="module")
def e4() -> btyd.E4Prediction:
    return btyd.predict(pc.purchases_scenario(6704, 0.5), 30, pp.PRIMARY_HORIZON)


def test_tiny_a_draws_that_underflow_now_simulate(e4: btyd.E4Prediction) -> None:
    rng = np.random.default_rng(6705)
    assert np.any(rng.beta(np.full(10_000, 1e-3), 30.0) == 0.0)  # the failure mechanism
    draws = {}
    for key, (_, bf, _, gf) in e4.fits.used.items():
        lb = np.tile(bf.log_params, (4, 1))
        lb[:, 2] = np.log(1e-3)  # a = 0.001 in every draw: Beta draws underflow to 0
        draws[key] = (lb, np.tile(gf.log_params, (4, 1)))
    sim = pc.e4_simulate_fast(e4, 4, np.random.default_rng(6706), param_draws=draws)
    assert np.all(np.isfinite(sim.outcome)) and np.all(sim.outcome >= 0.0)
    assert np.all(np.isfinite(sim.expected))
    old = ev.e4_simulate(e4, 3, np.random.default_rng(6707))  # P7 path: same rule, still runs
    assert np.all(np.isfinite(old.outcome))


def test_non_finite_parameter_draws_raise() -> None:
    x = np.array([0.0, 2.0])
    with pytest.raises(ValueError, match="finite and positive"):
        pc.e4_window_purchases(np.array([[0.5, np.nan, 0.7, 2.0]]), x, x, x + 50, 5.0, 50.0)
    with pytest.raises(ValueError, match="finite and positive"):
        pc.e4_window_purchases(np.array([[0.5, 8.0, 0.0, 2.0]]), x, x, x + 50, 5.0, 50.0)


# --- accelerated rule: small-a fallback -----------------------------------------------------------
# Below E4_ACCELERATED_MIN_A the original per-count nodes are used with the telescoped window
# integrand (A(e_0) - A(e_H)) / p, which is finite at p = 0; the frozen btyd code differences
# per-edge integrals of A(e) / p, which cancel catastrophically as a -> 0. Against a singularity-
# free adaptive reference (P8 A14 review) the A14 path was 2-4 times more accurate (a = 1e-6:
# 2.2e-6 vs 4.3e-6; 1e-3: 2.4e-9 vs 8.9e-9; 0.01: 3.8e-10 vs 1.3e-9), so the bounds below are
# the frozen code's own error envelope; for a >= 0.4 the A12 bound (2e-11) holds.
@pytest.mark.parametrize("a, rtol", [(1e-6, 5e-6), (1e-3, 2e-8), (0.01, 3e-9), (0.1, 2e-10),
                                     (0.39, 2e-11), (0.4, 2e-11), (1.0, 2e-11), (5.0, 2e-11)])
def test_small_a_uses_the_original_rule_and_large_a_meets_the_a12_bound(a: float,
                                                                         rtol: float) -> None:
    x = np.arange(0, 50, dtype=float)
    t_x = np.minimum(x * 1.5, 100.0)
    T = np.full(x.size, 104.0)
    edges = np.linspace(8.7, 60.8, 13)
    for r, alpha, b in ((0.3, 5.0, 0.5), (0.8, 12.0, 3.0), (2.0, 30.0, 8.0)):
        got, _ = pc.e4_window_purchases(np.array([[r, alpha, a, b]]), x, t_x, T,
                                        float(edges[0]), float(edges[-1]))
        ref = btyd.bgnbd_conditional((r, alpha, a, b), x, t_x, T, edges).purchases.sum(axis=1)
        np.testing.assert_allclose(got[0], ref, rtol=rtol)
    assert pc.E4_ACCELERATED_MIN_A == 0.4


# --- task-identified failures -----------------------------------------------------------------
def test_a_failing_task_names_its_study_and_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(seed: int, scale: float) -> dict[str, Any]:
        raise ValueError("p <= 0, p > 1 or p contains NaNs")

    monkeypatch.setattr(pc, "r3_draw", boom)
    with pytest.raises(RuntimeError, match=r"task R3 seed 6708 scale 0\.5 refits 0: "
                                           r"ValueError$"):
        pc.run_task(pc.Task("R3", 6708, 0.5))


# --- the one authorised new attempt -----------------------------------------------------------
class Recorder:
    def __init__(self) -> None:
        self.calls: list[pc.Task] = []

    def __call__(self, task: pc.Task) -> dict[str, Any]:
        self.calls.append(task)
        return {"task": task, "seed": task.seed, "seconds": 0.0}


def _stub_report(plan: Any, results: dict[str, Any], perf: dict[str, Any], run_dir: Path,
                 output: Path | None) -> list[Path]:
    out = run_dir / "record.md"
    out.write_text("stub")
    return [out]


def _fake_final() -> Any:
    return GEN.SeedPlan("final", 6000, 6001, True, {"R1": [6002], "R2": [6002], "R3": [6002],
                                                    "REFIT": [6003]}, (0.5, 1.0, 2.0), 2)


@pytest.fixture()
def failed_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """A temporary repository state like the real one: a failed final ledger and the 'final'
    reservation."""
    monkeypatch.setattr(GEN, "git_dirty_paths", lambda repo=None: [])
    monkeypatch.setattr(GEN, "resolve_fresh_seed", lambda plan, **kw: 6009)  # never reserved
    monkeypatch.setattr(GEN, "a15_gate_applies", lambda: False)  # historical replay only
    monkeypatch.setattr(pp, "PENDING_BEFORE_MAIN", ())  # historical fake A14 only
    ledger = tmp_path / "m11-p8-final" / "ledger.json"
    ledger.parent.mkdir()
    ledger.write_text(json.dumps({"mode": "final", "status": "failed",
                                  "phase": "repeated_sampling"}))
    GEN.AttemptRegistry(tmp_path / "registry").reserve("final", {"run_dir": str(ledger.parent)})
    return ledger, hashlib.sha256(ledger.read_bytes()).hexdigest()


def _attempt(tmp_path: Path, run: str, rec: Recorder | None = None) -> Path:
    out: Path = GEN.execute(_fake_final(), tmp_path / run, 1, fn=rec or Recorder(),
                            report_fn=_stub_report, registry=tmp_path / "registry",
                            evidence_root=tmp_path, expect_manifest=pp.protocol_digest(),
                            expect_code=pp.protocol_code_digest())
    return out


def test_without_an_authorisation_no_new_final_attempt_is_possible(
        tmp_path: Path, failed_attempt: tuple[Path, str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The state before A14 (key 'final', no listed failed ledger) refuses any new attempt."""
    monkeypatch.setattr(pp, "P8_FINAL_ATTEMPT_KEY", "final")
    monkeypatch.setattr(pp, "P8_FAILED_ATTEMPTS", ())
    assert GEN.final_attempt_key() == "final" and GEN.authorised_failed_attempts() == {}
    rec = Recorder()
    with pytest.raises(GEN.RunRefused, match="already attempted|protocol invalid"):
        _attempt(tmp_path, "m11-p8-final-a14", rec)
    assert rec.calls == []


def test_a14_authorises_exactly_one_new_attempt_on_the_same_seeds(
        tmp_path: Path, failed_attempt: tuple[Path, str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    ledger, sha = failed_attempt
    monkeypatch.setattr(pp, "P8_FINAL_ATTEMPT_KEY", "final-a14", raising=False)
    monkeypatch.setattr(pp, "P8_FAILED_ATTEMPTS", (("m11-p8-final/ledger.json", sha),),
                        raising=False)
    rec = Recorder()
    _attempt(tmp_path, "m11-p8-final-a14", rec)
    plan = _fake_final()
    assert rec.calls[0].seed == plan.main  # the same plan (seeds), not replacements
    led = json.loads((tmp_path / "m11-p8-final-a14" / "ledger.json").read_text())
    assert led["reservation"] == "final-a14" and led["status"] == "complete"
    assert ledger.read_text().startswith('{"mode": "final"')  # failed evidence untouched
    keys = [json.loads(line)["key"]
            for line in (tmp_path / "registry" / "attempts.log").read_text().splitlines()]
    assert keys == ["final", "final-a14"]
    for other in ("m11-p8-final-a14", "elsewhere"):  # no third attempt, any directory
        again = Recorder()
        with pytest.raises(GEN.RunRefused):
            _attempt(tmp_path, other, again)
        assert again.calls == []


@pytest.mark.parametrize("tamper", ["altered_failed_ledger", "extra_final_ledger"])
def test_a14_refuses_unlisted_or_altered_final_ledgers(tmp_path: Path,
                                                       failed_attempt: tuple[Path, str],
                                                       monkeypatch: pytest.MonkeyPatch,
                                                       tamper: str) -> None:
    ledger, sha = failed_attempt
    monkeypatch.setattr(pp, "P8_FINAL_ATTEMPT_KEY", "final-a14", raising=False)
    monkeypatch.setattr(pp, "P8_FAILED_ATTEMPTS", (("m11-p8-final/ledger.json", sha),),
                        raising=False)
    if tamper == "altered_failed_ledger":
        ledger.write_text(ledger.read_text() + " ")
    else:
        extra = tmp_path / "somewhere" / "ledger.json"
        extra.parent.mkdir()
        extra.write_text(json.dumps({"mode": "final"}))
    rec = Recorder()
    with pytest.raises(GEN.RunRefused, match="already attempted"):
        _attempt(tmp_path, "m11-p8-final-a14", rec)
    assert rec.calls == [] and not (tmp_path / "m11-p8-final-a14").exists()


def test_historical_final_attempts_are_consumed_and_no_retry_is_authorised() -> None:
    """Check committed failure and reservation metadata without attempting a reservation."""
    histories = {
        "final": ("m11-p8-final",
                  "cd7c8391347df2f9a2cf802161f35a0cb6a9a889b29c8fce3825a22ebe5902a9"),
        "final-a14": ("m11-p8-final-a14",
                      "74a0cdd66175ab19f5ecf0dddef08114230713897d5f968d6295df390b678008"),
    }
    reg = GEN.AttemptRegistry(GEN.REGISTRY_DIR)
    assert reg.validate() == [] and not reg.lock.exists()
    logged = [json.loads(line)["key"] for line in reg.log.read_text().splitlines()]
    final_keys = {
        path.stem for path in reg.root.glob("*.json")
        if json.loads(path.read_bytes()).get("mode") == "final"
    }
    assert final_keys == histories.keys()
    for key, (directory, expected_hash) in histories.items():
        assert logged.count(key) == 1
        reservation = (reg.root / f"{key}.json").read_bytes()
        assert json.loads(reservation)["key"] == key
        ledger = (GEN.EVIDENCE / directory / "ledger.json").read_bytes()
        assert hashlib.sha256(ledger).hexdigest() == expected_hash
        record = json.loads(ledger)
        assert (record["mode"], record["status"], record["phase"]) == (
            "final", "failed", "repeated_sampling")
        assert record["reservation"] == key
        assert record["reservation_sha256"] == hashlib.sha256(reservation).hexdigest()

    # Both keys already exist in the validated registry and can never be reserved again.
    # A15 recognises both historical failures and still retains an independent execution hold.
    assert GEN.final_attempt_key() == "final-a14" and GEN.final_attempt_key() in final_keys
    assert GEN.authorised_failed_attempts() == {
        f"{directory}/ledger.json": sha for directory, sha in histories.values()}
    assert GEN.unauthorised_final_attempts(GEN.EVIDENCE) == []
    with pytest.raises(GEN.RunRefused, match="protocol items are pending"):
        GEN.verify_protocol(_fake_final(), pp.protocol_digest(), pp.protocol_code_digest())
