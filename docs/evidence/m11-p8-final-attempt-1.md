# Milestone 11 P8 — first final attempt (failed): evidence record

This note records the first final P8 run, which failed. It is evidence only: it changes no code,
protocol value or result. The files it describes are kept exactly as the run wrote them.

## The attempt

| Item | Value |
|---|---|
| Command | `generate_m11_evidence.py --mode final` with manifest `9f92d0eb5e2b14ae4b7b06450c0adcf44ad86e2ada44de950c586c74f18ccef4` and code `a8b23ce6fc5052f5f7fe6d29fceaece95a7faf8461f42630ccc5ebd985ac1545` (A13) |
| Commit | `981aa10`, clean working tree (the ledger's "+ 2 uncommitted paths" are the run's own registry writes, made after the clean-tree check) |
| Reservation | `final` in `docs/evidence/m11-p8-attempts`, reserved 2026-11-20T01:04:11Z; entry `final.json` sha256 `e5dc98f0402381f5c73083278ca2cf140e6cf76fb77a3b6e5c5641f724c6ad04`; its line is the second in the hash-chained `attempts.log` (chain hash `e06b4729…cc7d7`) |
| Ledger | `docs/evidence/m11-p8-final/ledger.json`, sha256 `cd7c8391347df2f9a2cf802161f35a0cb6a9a889b29c8fce3825a22ebe5902a9`; status `failed`, phase `repeated_sampling` |
| Main phase | completed 2026-11-20T01:04:52Z on the main seed; results in `raw/main.pkl` (Git-ignored), sha256 `d514d59e4b8e813f23855c482f2f952818ebf202446bfef71a341b072cd23b11` |
| Repeated-sampling phase | started 01:04:52Z (2,300 tasks); failed, recorded at 03:23:18Z: `ValueError: p <= 0, p > 1 or p contains NaNs`; no results stored |
| M10 comparability and fresh seed | not reached; the fresh seed was **not** consumed |

## Diagnosis (read-only; development seeds only)

The message is raised only by NumPy's `Generator.geometric` with p = 0 or NaN; in the
repeated-sampling studies that call occurs only in E4's outcome simulation in R2
(purchases_noncontractual, C = 30), so the failing task is one or more R2 draws with seeds in
810000-810499. The ledger did not record which (`Pool.map` reports a worker error only after every
task finishes). Mechanism: a customer's dropout probability is drawn from Beta(a, b + x); when a
weakly identified cell's Hessian-based parameter draw gives a below about 0.02, Beta draws underflow
to exactly 0 in double precision (development check: 6e-4 of draws at a = 0.01), and `geometric(0)`
raises. In development purchases scenarios the smallest cell's log-a standard error reached 3.0,
putting 1-2% of its parameter draws below a = 0.02, so over R2's 500 draws x 200 parameter draws an
exact zero was practically certain; the 10-draw dry run was too small to meet it. Classification:
an implementation defect (an unhandled floating-point limit), not an unsupported parameter value
and not a fitting instability. The correction and the new attempt are the subject of amendment A14.

## Main-seed outcomes remain uninspected

- `raw/main.pkl` has never been opened, loaded or printed. Its file times are creation, last
  modification and last access all 2026-11-20 01:04:52, the moment the run wrote it.
- The failure analysis read only `ledger.json`, the registry files and file metadata (`stat`,
  SHA-256 hashes); the A14 work used development seeds only and no reserved seed.
- No primary-comparison estimate, p-value, selection result or metric from the main seed has been
  seen; the ledger contains none.
- `raw/main.pkl` stays outside Git (`.gitignore`: `docs/evidence/m11-p8-*/raw/`) with its hash
  recorded here and in the ledger. It is not deleted, so the record can be audited.
