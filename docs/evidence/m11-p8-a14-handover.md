# Milestone 11 P8 — post-A14 handover

Documentation only. This file changes no code, protocol value or evidence. It records the state
at which the single authorised retry of the final P8 run may be executed.

## 1. Milestone, commit and CI

| Item | Value |
|---|---|
| Milestone | M11 Predictive Customer Lifetime Value, phase P8 (final statistical evaluation) |
| Protocol stage | `P8-amended`; `PENDING_BEFORE_MAIN` empty |
| Commit | `805e8ccb551d8ef0a3f9da8595fa4a5ff9d63b08` ("M11 P8: amend numerical stability and governed final retry"), on `main` and `origin/main` |
| CI | GitHub Actions run 44700249953: success on Python 3.11 and 3.12 (ruff, mypy, pytest) |

## 2. Amendment A14

- **Numerical correction (`E4_ZERO_DROPOUT_RULE`).** In E4's outcome simulation (P7
  `e4_simulate`, P8 `e4_simulate_fast`) a dropout probability that is exactly 0 in floating point
  (a Beta(a, b + x) draw underflowing for very small a) is its limit, no dropout in the window:
  purchases until dropout = the largest int64, the value NumPy itself returns for every positive
  p below about 1e-300. The geometric draw is made with p replaced by 1, so every other draw and
  the random stream are unchanged. Non-finite or out-of-range inputs raise an error naming the
  task's study and seed.
- **Accelerated E4 range (`E4_ACCELERATED_MIN_A` = 0.4, `E4_ACCELERATED_SMALL_A`).** Parameter
  draws with a < 0.4 use the original per-count Gauss-Jacobi nodes with the telescoped integrand
  (finite at p = 0); for a ≥ 0.4 the accelerated rule is within 1.3e-11 of the per-count rule
  (the A12 bound 2e-11 holds). On development seeds the correction changed expected 12-month
  values by at most 1.2e-9 relative and no customer's coverage; P7 record sections and every P8
  dry-run table reproduce byte-identically.
- **One retry (`P8_FINAL_ATTEMPT_KEY` = `final-a14`, `P8_RETRY_RULE`)** on the same seed plan.
- **No change** to any estimand, comparison, gate, tolerance, study size, model or seed.
- **Timing.** A1–A13 were made before any main-seed computation. A14 was made after the first
  final attempt had computed the main-seed analysis but before any outcome was inspected. The
  protocol records this explicitly: A14 has `before_main_seed=False` and
  `before_outcome_inspection=True` and is the only amendment in
  `AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION`; `validate_protocol` requires every other amendment to
  precede any main-seed computation and every amendment to precede any outcome inspection.

## 3. Failed first attempt and preserved evidence

Full record: [m11-p8-final-attempt-1.md](m11-p8-final-attempt-1.md) (committed in `87db596`).

| Evidence | SHA-256 |
|---|---|
| `docs/evidence/m11-p8-final/ledger.json` (status `failed`, phase `repeated_sampling`) | `cd7c8391347df2f9a2cf802161f35a0cb6a9a889b29c8fce3825a22ebe5902a9` |
| `docs/evidence/m11-p8-attempts/final.json` (reservation `final`, consumed) | `e5dc98f0402381f5c73083278ca2cf140e6cf76fb77a3b6e5c5641f724c6ad04` |
| `docs/evidence/m11-p8-attempts/attempts.log` (hash chain: dry run, `final`) | `05fe45b2c9d4a8ff3fde2fb6109b37cdec4b44623332bd6a10c817c6ab17c8df` |
| `docs/evidence/m11-p8-final/raw/main.pkl` (Git-ignored, kept on disk) | `d514d59e4b8e813f23855c482f2f952818ebf202446bfef71a341b072cd23b11` |

The attempt completed the main phase, then failed in repeated sampling with
`ValueError: p <= 0, p > 1 or p contains NaNs` from E4's R2 outcome simulation (a seed in
810000–810499). No repeated-sampling result was stored; the M10 comparability phase and the
fresh seed were not reached. The reservation `final` is consumed and can never be reused.

## 4. Disclosure: `main.pkl`

During the post-A14 verification `main.pkl` was read as raw bytes to compute its SHA-256 and
confirm it matches the value above. It was **not** unpickled, loaded or otherwise inspected, and
no outcome value was seen. Its file times (creation and last access) remain 2026-11-20 01:04:52,
the moment the failed run wrote it. The owner has confirmed not opening, loading or inspecting it.

## 5. Frozen digests (pinned in `tests/unit/customer_value/test_predictive_protocol.py`)

| Digest | SHA-256 |
|---|---|
| Protocol manifest | `27c949b491a25ee4f571f9f0243903a7d3d3e4c7ca2097db30abdf94dfb0ba08` |
| Protocol code | `8830e11ebb46a6f712c458f9dff23a27c6ab710ff1359d2dd256ee90ea0d8879` |

## 6. The unused `final-a14` reservation and the execution command

At `805e8cc`: the registry validates (no lock, entries `dry-run-a3cef63df948e378` and `final`),
`final-a14` has never been reserved, `docs/evidence/m11-p8-final-a14` does not exist, and the only
final ledger present is the one listed in `P8_FAILED_ATTEMPTS`.

Commit this handover first: final mode refuses a working tree with uncommitted or untracked
files. Then, from the repository root, in a terminal that stays open for the run:

```
source <python-3.12-venv>/bin/activate
PYTHONPATH=src caffeinate -i python scripts/generate_m11_evidence.py --mode final \
  --expect-manifest 27c949b491a25ee4f571f9f0243903a7d3d3e4c7ca2097db30abdf94dfb0ba08 \
  --expect-code 8830e11ebb46a6f712c458f9dff23a27c6ab710ff1359d2dd256ee90ea0d8879
```

The run reserves `final-a14`, writes to `docs/evidence/m11-p8-final-a14`, and takes about
2.3 hours on 3 workers with about 3 GB peak memory (`BUDGETS`: 180 min, 4 GB). The reservation is
consumed when the run starts, whatever its outcome.

## 7. Python environment

| Requirement | Value |
|---|---|
| Interpreter | CPython 3.12.8 (the tested venv was created from python.org `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`) |
| Packages | numpy 2.5.5, scipy 1.18.1, pandas 2.3.3, statsmodels 0.15.0 |
| Import path | `PYTHONPATH=src` is required, so the project is imported from this checkout; the tested venv also contains a stale installed copy of the project |
| Threads | set by the script (one numeric thread per worker; `P8_EXECUTION`: 3 workers) |

The tested venv was in a temporary scratch directory, so it may not survive a reboot; any
equivalent environment must match the versions above. Before running, confirm
`python -c "import sys; print(sys.executable)"` names that venv. The digest check refuses
mismatched code but not a different numerical stack.

## 8. Remaining restrictions

- `final-a14` is the last authorised attempt; any failure or interruption requires a new
  amendment before any further attempt (`P8_RETRY_RULE`, `P8_SINGLE_ATTEMPT`).
- Same seeds only: the main seed, the M10 comparability seed, the coverage block
  (810000–810999) and the refit block (820000–820199); no replacement seeds.
- The fresh seed is used once, last, and only by the run itself.
- The failed attempt's `main.pkl` stays uninspected; the failed ledger, reservation and registry
  are never edited, moved or deleted.
- No change to the protocol, estimands, PC1–PC5, the Holm family, selection gates, tolerances,
  study sizes, models or the coverage reporting rule before or during the run.
- Coverage is reported, never asserted, and never used for recalibration.
- Confirmatory results are PC1–PC5 (Holm), `ORACLE_GATE` (R1) and `ACCEPTANCE_GATE_4` (R3);
  everything else is exploratory. The E3 refit-bootstrap validation is not performed
  (`E3_REFIT_DISCLOSURE`).
- Do not inspect any outcome before the run has completed and been verified.

## 9. After the run

1. If the command exits non-zero, do not rerun. The ledger in `docs/evidence/m11-p8-final-a14`
   records the failure; commit that evidence (as for the first attempt) and stop for review.
2. Verify the completed run (status, phases, checksums, registry entry and hash chain):

   ```
   PYTHONPATH=src python scripts/generate_m11_evidence.py --verify docs/evidence/m11-p8-final-a14
   ```

   It must report "run verified: complete and unaltered".
3. Commit the evidence by explicit paths only; `raw/` stays outside Git (`.gitignore`) with its
   hashes in the ledger:

   ```
   git add -- docs/evidence/m11-p8-attempts/attempts.log \
     docs/evidence/m11-p8-attempts/final-a14.json \
     docs/evidence/m11-p8-final-a14/ledger.json \
     docs/evidence/m11-p8-final-a14/record.md \
     docs/evidence/m11-p8-final-a14/summary.json \
     docs/evidence/m11-p8-final-a14/*.csv
   git diff --cached --name-only   # no raw/ path; nothing outside these
   ```

4. Only then review the record. Writing up the results (findings, methodology, capability
   status, roadmap) is a separate, later step.
