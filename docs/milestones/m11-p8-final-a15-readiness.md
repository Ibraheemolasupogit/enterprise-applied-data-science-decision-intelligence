# M11 P8 — final-a15 execution authorisation

## A18 launch control

A18 prepares an independently enforced launch gate. The owner must separately approve the
exact, clean release commit after successful Python 3.11 and 3.12 CI and a passing immediate
host preflight. The A15-era activation proposal below is historical.

On the designated Mac, the runner expects three files under `/private/etc/m11-p8/`.
Host administration must establish this root-controlled directory and its trust anchor
with the owner's public signing key. The operator should run as an unprivileged user.
Directories and files must be root-owned and readable by the operator, with no group or
other writes, symlinks or ACLs that grant untrusted writers access.

| File | Required contents |
|---|---|
| `allowed_signers` | OpenSSH allowed-signers entry for the owner's public key and principal `m11-p8-owner` |
| `final-a15.json` | Exact release binding and preflight attestation as described below |
| `final-a15.json.sig` | Owner's detached OpenSSH signature over the exact JSON bytes in namespace `m11-p8-final-a15` |

The JSON fields are `commit` (full SHA), `manifest_sha256`, `code_sha256`, `runner_sha256`,
`supervisor_sha256`, `host_uuid` (lowercase `IOPlatformUUID`), `attempt` (`final-a15`),
`registry` and `output` (absolute canonical paths), and `ci_run_id` (GitHub Actions run ID
as a decimal string). All are strings. The additional field `host_preflight_passed` must
be JSON `true`. The owner verifies CI and preflight before signing. The runner verifies the
signature and live binding locally and makes no remote CI or host-preflight query.

Keep the approval available through the fresh phase because seed authorisation is checked
again there. The final attempt remains single-use through the existing registry. Remove the
approval when the attempt closes or if the owner withdraws approval before dispatch.

**PROPOSED — NOT AUTHORISED**

**NO-GO for dispatch.** A15 is active at protocol stage `P8-amended`. Its numerical
amendment records the owner's outcome-blindness confirmation. The separate execution hold in
`PENDING_BEFORE_MAIN` still blocks main and fresh seed access, and the configured key
`final-a14` is consumed. The owner has approved the sampled-RSS method. Execution requires
explicit owner authorisation, an exact green activation commit with new digest pins, and a
passing preflight on the designated execution host. [Monitoring decision](m11-p8-resource-monitoring-approval.md) ·
[Active protocol](../../src/decision_intelligence/customer_value/predictive_protocol.py)

## Proposed scope and version pins

- One reservation key, `final-a15`, through `docs/evidence/m11-p8-attempts`. The runner alone
  creates `docs/evidence/m11-p8-attempts/final-a15.json` and appends its chain record.
- One output directory, `docs/evidence/m11-p8-final-a15/`, with the runner's default report
  location, `record.md` inside that directory. The key and directory were absent at review.
  Both must remain absent until the authorised dispatch.
- Frozen design and seed plan. Three workers with one numerical thread per worker. R1 has
  1,000 draws, R2 has 500, R3 has 200 at each of 0.5, 1 and 2, and REFIT has 200 draws
  with 200 refits per draw. Preserve all estimands, models, comparisons, tolerances,
  PC1–PC5 Holm family, acceptance gates and descriptive coverage rules.

| Pin | Verified review baseline or required execution version |
|---|---|
| Full checkout | Baseline `afd2769fbbf9b113ce700babf8034d2ab3211118`. [CI 44896070680](https://github.com/Ibraheemolasupogit/enterprise-applied-data-science-decision-intelligence/actions/runs/44896070680) completed successfully on this SHA, with both Python jobs green. Execution requires the full SHA and green CI run of the future owner-approved activation commit. |
| Protocol manifest | Held baseline SHA-256 `9a6317be2bad202fde5873f7ee7a688c439a71693bc8ed9f29c9284f44fdb91b`. Compute, pin and approve the replacement after the activation diff is final. |
| Protocol code | Held baseline SHA-256 `60981a52e28611269ed3b3ec4c31bc93a102eb664f45fb43a81981f8e0de3861`. Compute, pin and approve the replacement. This digest covers `predictive_protocol.py`, `model_selection.py` and `precision.py`. |
| Runner | `scripts/generate_m11_evidence.py` from `e580871c8a2200ed303e1ff370e6f858dda6f45e`, as shipped at baseline. File SHA-256 `c4e3ab0ffe9ba89f7c6c28b1d93ebce73a3d0c1b16d6935954e7d03d69464e03`. |
| Supervisor and watchdog | `src/decision_intelligence/customer_value/resource_supervision.py` from `ce42235d18cb709b19ee0ff70415452b609b9ac4`, as shipped at baseline. File SHA-256 `7e8c78ba341eafe11ac3c1587246a5e2a62ea43023d388388f0b4750afec5c9b`. |

The two held digests match the test pins. The final commit and replacement digests depend on
the approved authorisation text and date. The runner and supervisor are pinned independently
because neither is covered by the protocol code digest.

## Permanently consumed attempts

Both ledgers are final-mode failures in `repeated_sampling`. Both record main completion
and end before repeated-sampling completion, M10 comparability and fresh-seed consumption.
Both keys remain permanently consumed across output paths and are ineligible for reuse.
Paths below are relative to `docs/evidence/`.

| Key and recorded failure | Ledger and SHA-256 | Reservation and SHA-256 |
|---|---|---|
| `final` — E4 R2 geometric draw rejected its probability | `m11-p8-final/ledger.json` · `cd7c8391347df2f9a2cf802161f35a0cb6a9a889b29c8fce3825a22ebe5902a9` | `m11-p8-attempts/final.json` · `e5dc98f0402381f5c73083278ca2cf140e6cf76fb77a3b6e5c5641f724c6ad04` |
| `final-a14` — R2, seed 810085, scale 1, zero refits, `ZeroDivisionError`; cause unconfirmed | `m11-p8-final-a14/ledger.json` · `74a0cdd66175ab19f5ecf0dddef08114230713897d5f968d6295df390b678008` | `m11-p8-attempts/final-a14.json` · `806a2b126e94a903a170e7ea50e7b2ae0a8dd7039852772df950dc7dc9cc7ae3` |

Read-only registry and A15 history validation passed at baseline. The entire pre-launch
`attempts.log` (dry run, `final`, `final-a14`) has SHA-256
`52c6031120ff02a58b6b773134f69ad9d48d322a20fb10ce9418c6af58eec2fb`.
Both protected `raw/main.pkl` files were present. This review checked file presence only.
The earlier byte-only hash check is disclosed in the historical handover.
[First failure](../evidence/m11-p8-final-attempt-1.md) ·
[A14 handover](../evidence/m11-p8-a14-handover.md) · [A15 numerical review](m11-p8-a15-draft.md)

## Minimum activation changes after owner approval

1. In `predictive_protocol.py`, append a narrow execution amendment, proposed as `A16`.
   Record the actual approval date, `before_main_seed=False`, and
   `before_outcome_inspection=True` only with owner confirmation that protected outcomes
   remain uninspected through approval. Add `A16` to `AMENDMENTS_AFTER_MAIN_SEED_COMPUTATION`.
   Authorise the exact key, directory and registry above for one supervised launch on the
   frozen plan with both pinned historical failures required. Record that the A14 retry is
   exhausted. Preserve the A15 amendment as approved and its ID so the A15 gate still applies.
2. In the same change, set `P8_FINAL_ATTEMPT_KEY = "final-a15"` and
   `PENDING_BEFORE_MAIN: tuple[str, ...] = ()`. Align the hold and current-key docstrings.
   Keep the historical A14 rule scoped to `final-a14` by making its key reference literal.
   Preserve the two `P8_FAILED_ATTEMPTS` ledger pins and all historical evidence bytes.
3. Recompute `protocol_digest()` and `protocol_code_digest()` from that exact tree and
   replace `FROZEN_DIGEST` and `FROZEN_CODE_DIGEST` in
   `tests/unit/customer_value/test_predictive_protocol.py`. Adjust the amendment and timing
   assertions, live key and hold assertions, and the historical rule text assertion.
   Update the live-key assertion in `test_p8_execution.py` and the live key/hold assertions
   in `test_p8_a14.py` in the same directory. Preserve synthetic hold/refusal tests and
   permanent-consumption checks. The pinned runner and supervisor already implement the
   gate and need no changes.

## Authorisation, preflight and single dispatch

1. **Authorise configuration.** Obtain the owner's explicit recorded approval for this
   activation and one proposed attempt, including the outcome-blindness confirmation and
   acceptance of the remaining uncertainties below. Apply only the activation changes.
2. **Validate and bind the launch.** Confirm a manifest diff restricted to the authorised
   changes and `validate_protocol() == []`. Use fake tasks and temporary fixtures for
   gate tests, including history tampering, wrong path/key, duplicate dispatch and fresh-last
   sequencing. Run the Darwin supervisor tests on the designated host, including memory
   breach, monitor or watchdog loss, worker loss and timeouts during serialization and
   report writing. Require green full CI on both Python 3.11 and 3.12 for the activation SHA.
   Record the owner's launch approval against that exact commit, CI run, replacement digests,
   runner/supervisor hashes, host and canonical paths.
3. **Require every preflight check immediately before launch.**
   - **Git and imports** — `HEAD` equals the approved SHA, and
     `git status --porcelain=v1 --untracked-files=all` is empty, including the index.
     Verify green CI on that SHA, both live digests against the approved values and test
     pins, both file hashes, and imports from this checkout via `PYTHONPATH=src`.
   - **Host and capacity** — Confirm macOS with working native `libproc` and CPython
     **3.12.8**, NumPy **2.5.5**, SciPy **1.18.1**, pandas **2.3.3** and statsmodels
     **0.15.0**. Confirm three worker slots, available RAM for the 4 GB budget plus OS
     headroom, and writable space in the evidence and temporary filesystems sufficient
     for raw results, serialization and final reports. Keep the host awake, on stable
     power, with a clear 180-minute window and competing workloads out of the way.
   - **Resource enforcement** — Require a valid initial aggregate RSS sample and a ready
     watchdog before the child can dispatch work. Confirm the native sampler covers the
     runner group, including descendants, plus supervisor and watchdog. Use the approved
     sampled peak ceiling of **4,000,000,000 bytes** and **10,800 seconds** through final
     report writing and completion certification. A measured breach, timeout or monitoring
     loss fails the attempt.
   - **History and exclusivity** — In the one designated canonical checkout, use
     `AttemptRegistry.validate()` and `verify_a15_history()` read-only. Match the entire
     pre-launch chain to its pin above and require exactly the two pinned final ledgers and
     their corresponding entries. Preserve protected raw artifacts and outcome blindness.
     Require absent candidate directory, reservation file, chain entry and lock, including
     dangling symlinks, and exclude parallel operators or dispatches from other checkouts.
     Missing, changed, redirected, unreadable or unexpected evidence is a NO-GO.
4. **Dispatch once.** From the repository root use the approved absolute interpreter path
   and the two new owner-approved digest values in this command template.

   ```sh
   PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     caffeinate -i "$approved_python" scripts/generate_m11_evidence.py --mode final --workers 3 \
     --registry docs/evidence/m11-p8-attempts --run-dir docs/evidence/m11-p8-final-a15 \
     --expect-manifest "$approved_manifest_digest" --expect-code "$approved_code_digest"
   ```

   The normal entry point resolves the frozen main plan and starts the external supervisor.
   Once the supervisor opens the gate, the child verifies the protocol and digests, validates
   history under the registry lock, exclusively creates the reservation and appends the
   chain, then creates the ledger before generating any reserved-seed data.
   The runner uses main, repeated sampling, then M10 comparability. Only that active ledger,
   with those phases completed in order, can authorise the frozen fresh seed. The runner logs
   `fresh / seed consumed` before generating fresh data once. The seed value itself is
   fixed by protocol and is resolved only inside that phase.
5. **Close out.** Only the supervisor can mark a successful run complete. Before opening
   outcomes, verify the completed run with the pinned runner's
   `--verify docs/evidence/m11-p8-final-a15` and raw verification enabled, check the exact
   phase sequence and single fresh consumption, and verify that the old registry bytes
   are the prefix of the chain with exactly one new entry. On any refusal, error, timeout,
   monitoring failure, interruption or ambiguous completion, preserve all evidence and
   stop for owner review. Treat an existing or partial reservation as consumed, even with
   no completed ledger. Any further attempt requires a new amendment and authorisation.

## Remaining NO-GO conditions and uncertainties

The authorisation, activation SHA and post-change pins are pending. The execution host and its
capacity must pass preflight. The Python found on the current path has different NumPy and
SciPy versions and lacks statsmodels, so it fails the environment check.

The three-worker benchmark measured 2.282 GB sampled RSS and projected 159.25 minutes.
Full-run retention, serialization, reporting and runtime variance remain unmeasured by that
benchmark, and the approved sampling method can miss transient spikes. The A14 crash cause
remains unconfirmed. The A15 review identifies residual Gamma-Gamma mean and extreme-rate
precision risks. Owner review must accept these limits before dispatch. A failed or unverified
preflight item keeps the decision at **NO-GO**. [A15 numerical review and benchmark](m11-p8-a15-draft.md)
