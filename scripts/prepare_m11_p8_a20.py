"""Render the A20 adoption patch without changing the working tree."""

import argparse
import sys

from scripts.prepare_m11_p8_a19 import render_patch

from decision_intelligence.customer_value import p8_candidate

CHANGE = (
    "Configure final-a20 for at most one independently authorised confirmatory dispatch. "
    "Keep final, final-a14, final-a15 and final-a19 permanently consumed. Preserve the "
    "protected A19 startup-failure record and bind approval to its journal chain tip. "
    "Require the dedicated worker to verify the immutable checkout, including the Git "
    "index, the signed release and the private history under native supervision before "
    "consumption. Preserve all frozen estimands, numerical and inference methods, seeds, "
    "RNG sequences, simulation counts and scales, acceptance thresholds, reporting "
    "semantics and fresh-last ordering. Keep the qualified hybrid schedule and the "
    "4,000,000,000-byte aggregate sampled RSS and 10,800-second limits."
)
REASON = (
    "The final-a19 attempt on release 493fd0b was consumed by a child failure with "
    "exit code 2 after 1.657 seconds. The dedicated worker could not read the root-owned "
    "Git index and stopped before an execution ledger was created. The index permissions "
    "were repaired and worker Git status and complete signed-release verification passed. "
    "The owner confirmed continued outcome blindness. A20 adoption grants no launch "
    "authority. A separate attempt-specific signed approval requires green CI on the "
    "exact amended release and a passing immediate host preflight."
)


def prepare_patch(decision_date: str) -> str:
    return render_patch(
        decision_date, prior=("A19", p8_candidate.A19_ATTEMPT),
        successor=("A20", p8_candidate.A20_ATTEMPT), change=CHANGE, reason=REASON,
        old_doc="The A19 successor. Execution requires separate signed owner approval.",
        new_doc="The A20 successor. Execution requires separate signed owner approval.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-date", required=True, help="independently approved YYYY-MM-DD")
    args = parser.parse_args()
    try:
        sys.stdout.write(prepare_patch(args.decision_date))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
