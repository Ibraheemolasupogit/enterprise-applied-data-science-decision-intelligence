"""Replay the archived orchestration with fake draws and temporary metadata only."""

from pathlib import Path
from types import ModuleType
from typing import Any


def execute_archived(g: ModuleType, plan: Any, run_dir: Path, workers: int, *,
                     fn: Any, report_fn: Any, registry: Path, evidence_root: Path,
                     expect_manifest: str, expect_code: str) -> Path:
    g.verify_protocol(plan.mode, expect_manifest, expect_code)
    g.pp.require_launch_approval()
    key = g.reservation_key(plan)
    reg = g.AttemptRegistry(registry)
    preflight = None
    if g.a15_gate_applies():
        if key != g.A15_ATTEMPT_KEY:
            raise g.RunRefused(
                "A15 requires a separately authorised, new final-a15 reservation key")
        g.require_final_target(evidence_root, registry, run_dir, None)
        if not reg.root.is_dir():
            raise g.RunRefused("canonical registry is missing")

        def verify_candidate() -> None:
            if run_dir.exists() or run_dir.is_symlink():
                raise g.RunRefused("final-a15 output directory already exists and cannot be reused")
            g.verify_a15_history(evidence_root, reg)
            g.pp.require_launch_approval(require_clean=False)

        preflight = verify_candidate
    elif g.unauthorised_final_attempts(evidence_root):
        raise g.RunRefused("a final run was already attempted")
    entry = reg.reserve(key, {"run_dir": str(run_dir), "mode": plan.mode,
                              "manifest_digest": expect_manifest, "code_digest": expect_code},
                        preflight=preflight)
    ledger = g.Ledger.create(run_dir, {**g.protocol_header(plan, workers),
                                     "registry": str(registry),
                                     "reservation": key, "reservation_sha256": entry})
    result: Path = g._execute_phases(plan, run_dir, workers, ledger, fn, report_fn, None, None)
    return result
