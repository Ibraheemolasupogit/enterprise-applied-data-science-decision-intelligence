"""Render the A19 adoption patch for independent review without writing any files."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from textwrap import wrap

from decision_intelligence.customer_value import p8_candidate
from decision_intelligence.customer_value import predictive_protocol as protocol

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path("src/decision_intelligence/customer_value/predictive_protocol.py")
TEST_PATH = Path("tests/unit/customer_value/test_predictive_protocol.py")

CHANGE = (
    "Configure final-a19 for one governed confirmatory dispatch. Preserve the frozen "
    "estimands, scenario and seed assignments, RNG sequences, model and inference methods, "
    "simulation counts, acceptance thresholds, output semantics and fresh-last ordering. "
    "Keep isolated fits, three workers for R1/R3/REFIT and a fresh two-worker pool "
    "for each R2 pair. Retain the 4,000,000,000-byte aggregate sampled RSS ceiling and "
    "10,800-second deadline. Require owner-pinned private history, immutable source and "
    "runtime, durable consumption before dispatch, and a protected terminal receipt "
    "followed by a within-budget certification decision. Keep outcomes private. "
    "The final, final-a14 and final-a15 identities remain permanently consumed. "
    "Failure or interruption consumes final-a19 permanently."
)
REASON = (
    "The owner confirmed continued outcome blindness after prior main-seed computation. "
    "Full-count development qualification supports this execution configuration. "
    "A19 adoption alone grants no launch authority. A separate signed owner approval "
    "is required after green CI on the exact amended release and a passing immediate "
    "host preflight."
)


def _replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(f"expected exactly one occurrence of {before.splitlines()[0]!r}")
    return source.replace(before, after, 1)


def _literal_lines(text: str) -> str:
    parts = wrap(text, width=76, break_long_words=False, break_on_hyphens=False)
    return "".join(f"              {part + ' '!r}\n" for part in parts[:-1]) + (
        f"              {parts[-1]!r},\n"
    )


def render_patch(decision_date: str, *, prior: tuple[str, str], successor: tuple[str, str],
                 change: str, reason: str, old_doc: str, new_doc: str) -> str:
    """Compute a date-bound amendment and its integrity pins."""
    prior_id, prior_attempt = prior
    amendment_id, attempt = successor
    approved = date.fromisoformat(decision_date)
    previous = protocol.PROTOCOL_AMENDMENTS[-1]
    if (protocol.P8_FINAL_ATTEMPT_KEY != prior_attempt or previous.id != prior_id
            or protocol.AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION[-1] != prior_id):
        raise ValueError(f"expected the held {prior_id} protocol")
    if approved.isoformat() != decision_date or approved < date.fromisoformat(previous.date):
        raise ValueError(
            f"decision date must be YYYY-MM-DD and follow or equal the {prior_id} date")
    protocol_file = REPO / PROTOCOL_PATH
    if protocol.__file__ is None or Path(protocol.__file__).resolve() != protocol_file:
        raise ValueError("import the protocol from the checkout using PYTHONPATH=src")
    amendment = protocol.Amendment(
        amendment_id, "P8", decision_date, False, change, reason, before_outcome_inspection=True)
    source = protocol_file.read_text(encoding="utf-8")
    revised = _replace_once(
        source, f'P8_FINAL_ATTEMPT_KEY = "{prior_attempt}"\n"""{old_doc}"""',
        f'P8_FINAL_ATTEMPT_KEY = "{attempt}"\n"""{new_doc}"""')
    timing = tuple(protocol.AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION)
    timing_line = 'AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION = ("' + '", "'.join(timing)
    revised = _replace_once(revised, timing_line + '")', timing_line + f'", "{amendment_id}")')
    marker = '\n)\n"""Protocol changes since P0'
    record = (
        f'\n    Amendment("{amendment_id}", "P8", "{decision_date}", False,\n'
        + _literal_lines(change)
        + _literal_lines(reason)
        + '              before_outcome_inspection=True),'
    )
    revised = _replace_once(revised, marker, record + marker)

    manifest = protocol.protocol_manifest()
    manifest["P8_FINAL_ATTEMPT_KEY"] = attempt
    manifest["AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION"].append(amendment_id)
    manifest["PROTOCOL_AMENDMENTS"].append(asdict(amendment))
    canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    manifest_digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()

    code = hashlib.sha256()
    for name in protocol.PROTOCOL_CODE_FILES:
        code.update(name.encode("ascii") + b"\0")
        body = revised.encode("utf-8") if name == protocol_file.name else (
            protocol_file.with_name(name).read_bytes())
        code.update(body.replace(b"\r\n", b"\n"))
    tests = (REPO / TEST_PATH).read_text(encoding="utf-8")
    updated_tests = _replace_once(
        tests, f'FROZEN_DIGEST = "{protocol.protocol_digest()}"',
        f'FROZEN_DIGEST = "{manifest_digest}"')
    updated_tests = _replace_once(
        updated_tests, f'CANDIDATE_CODE_DIGEST = "{protocol.protocol_code_digest()}"',
        f'CANDIDATE_CODE_DIGEST = "{code.hexdigest()}"')
    return "".join(
        line
        for path, before, after in (
            (PROTOCOL_PATH, source, revised), (TEST_PATH, tests, updated_tests))
        for line in difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}")
    )


def prepare_patch(decision_date: str) -> str:
    return render_patch(
        decision_date, prior=("A18", "final-a15"), successor=("A19", p8_candidate.A19_ATTEMPT),
        change=CHANGE, reason=REASON,
        old_doc="The historical A16 key. The draft A19 successor requires owner adoption.",
        new_doc="The A19 successor. Execution requires separate signed owner approval.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-date", required=True, help="owner-reviewed YYYY-MM-DD")
    args = parser.parse_args()
    try:
        sys.stdout.write(prepare_patch(args.decision_date))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
