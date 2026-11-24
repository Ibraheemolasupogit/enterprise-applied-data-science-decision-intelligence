"""Render the A21 adoption patch without changing the working tree."""

import argparse
import sys

from scripts.prepare_m11_p8_a19 import render_patch

from decision_intelligence.customer_value import p8_candidate

CHANGE = (
    "Configure final-a21 for at most one separately authorised confirmatory dispatch. "
    "Keep all earlier attempt identities permanently consumed and bind approval to the "
    "protected A19 and A20 chain tips and A20 failure metadata. For finite E1/E2 draws "
    "outside the fitted log-parameter box, evaluate the same conditional survival ratio "
    "directly in the log domain. Keep the existing evaluator inside the box, the "
    "unconstrained draw distribution and RNG consumption. Preserve the estimands, seed "
    "plan, fitting and inference methods, simulation counts and scales, reporting, "
    "acceptance criteria, hybrid scheduling, fresh-last ordering and the 4,000,000,000-byte "
    "sampled RSS and 10,800-second limits."
)
REASON = (
    "A20 on release 7b07e24 stopped in R2 with an OverflowError while exponentiating "
    "an unbounded contractual log-parameter draw. A20 is consumed. A log-domain beta "
    "product for E1 and conditional cumulative hazard for E2 avoid overflow and "
    "cancellation in the tail without changing the statistical model or resampling. "
    "Fitted-range probabilities and bounded ordinary development R2 results were bitwise "
    "or exactly equal, and the bounded overflow fixture preserved both RNG states. "
    "Out-of-box results can change numerically, and the failed run has no equivalent "
    "completed output. The owner confirmed continued outcome blindness. A21 adoption "
    "requires independent review and grants no launch authority. Any launch requires "
    "separate signed approval after exact-release CI and immediate host preflight."
)


def prepare_patch(decision_date: str) -> str:
    return render_patch(
        decision_date, prior=("A20", p8_candidate.A20_ATTEMPT),
        successor=("A21", p8_candidate.A21_ATTEMPT), change=CHANGE, reason=REASON,
        old_doc="The A20 successor. Execution requires separate signed owner approval.",
        new_doc="The A21 successor. Execution requires separate signed owner approval.")


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
