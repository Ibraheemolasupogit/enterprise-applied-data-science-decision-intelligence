# M11 P8 — A15 owner approval package

**Status: ready for owner review. A15 is pending and inactive.** The implementation and
development validation are based on checkpoint `8e12c1231c4556e698e05089f7d297ff92cc1d3f`.

**Recommendation: GO for amendment approval once the owner confirms outcome blindness for
both attempts. NO-GO for execution under this package.** Amendment approval accepts the
documented correction and scope. Execution requires a separate, explicit owner decision
and the preflight below. Both historical reservations remain permanently consumed.

## Diagnosis and attribution

The checkpoint's `btyd.beta_nodes` formed `al = b - 1`, `be = a - 1`, then evaluated
`(be - al) / (al + be + 2)`. For positive shapes such as `a = 2**-60`, `b = 2**-61`, both
subtractions round to -1 and the Python scalar division is 0/0. The batched implementation had
the same cancellation. The first off-diagonal also had a removable 0/0 at `a + b = 1`.
These are implementation defects for valid positive Beta shapes, independently of any seed.

The recorded `final-a14` failure identifies R2 and its task metadata but supplies neither a
traceback nor shape values. The batched eigensolve precedes the scalar fallback, and R2 includes
other components. **The historical crash is not attributed conclusively to this defect.**
A14 expanded the path into an existing scalar helper; it did not introduce that helper's
formula. Its separate zero-dropout correction is retained without modification.

## Three numerical corrections

1. **Beta quadrature and continuous window integrand.** Scalar and batched rules share a
   shifted Jacobi recurrence in the positive shapes. The first diagonal is `a/(a+b)`.
   The first off-diagonal is `sqrt(a/(a+b))*sqrt(b/(a+b))/sqrt(a+b+1)`, with the removable
   factor cancelled analytically. Beta reflection preserves small complementary node
   coordinates. The scalar tridiagonal eigensolver also handles batched rows where the dense
   solver reaches an endpoint or shape addition loses the smaller shape. Numerical node
   excursions are projected to [0, 1], and the nonnegative eigenvector-squared weights are
   normalised. The shared conditional predictor and P8 window calculation use the same
   log1p/exprel factorisation of `(A(left)-A(right))/p`. Its continuous value at zero is
   `(r+x)*(right-left)/(alpha+T)`.
2. **Accelerated Beta weights.** Use base weights directly at `x = 0` wherever the accelerated
   rule applies. For positive counts, use the stable per-count Beta(a, b+x) rule when the
   complementary coordinate vanishes or the reweighting is numerically unusable. The
   `a < 0.4` A14 and `x > 44` A12 fallbacks still apply. The synthetic
   `a = 0.8, b = 1e-310` fixture showed `betaln` infinity and nonfinite windows at
   `x = 0, 1, 44` before this correction. It now has finite windows at all tested counts.
3. **Alive probability.** Scalar and batched paths share a log-odds calculation with denominator
   `b + (x-1)` for positive counts. Mantissa/exponent arithmetic preserves the log shape ratio.
   A direct log-odds-to-probability conversion avoids overflow and retains representable
   subnormals. The alive probability at `x = 0` is one. With `r = 446, a = 1e-310, b = 3`
   and `x = 1`, the pre-correction path returned zero. The corrected fixture returns
   `0.010784996567235729` against a high-precision reference near `0.010784996567238`.

These three defects were established on deterministic inputs. None identifies the historical
`final-a14` failure site. R2 now raises on nonfinite E4 point predictions, simulated values,
interval bounds or targets before those observations can be silently lost from the coverage
denominator. The correction preserves the statistical estimand, 64 nodes, Beta shapes, fit
procedure, Hessian draw law and draw count. A14 zero-dropout handling remains in force.

## Safe failure capture

`run_task` now raises a RuntimeError subclass carrying a serializable chain of exception types
and frame locations. Capture occurs in the worker before multiprocessing loses its traceback.
The ledger records that structured chain. Explicit causes, implicit contexts, suppression and
cycles are handled. Only file basenames, line numbers and function names are retained: no
exception payloads, source lines, local variables, remote traceback text or statistical outcomes.
The wrapper message contains task metadata and an exception type. Generic ledger/CLI failure
messages contain the type only. This applies to future failures, not to either existing ledger.

## Initial development validation

Validation uses deterministic handcrafted histories/fits and existing development-seed tests.
No failed `raw/main.pkl` was opened, hashed or unpickled. No final mode was executed, including
fake final-mode orchestration tests. New diagnostic tests use temporary development registries;
the committed registry is untouched.

Initial development stack: CPython 3.12.8, NumPy 2.4.6, SciPy 1.17.1, pandas 2.3.3. This differs from the A14
handover's NumPy 2.5.5 / SciPy 1.18.1; statsmodels is absent locally.

| Check | Result |
|---|---|
| New boundary/trace tests | 37 passed |
| Combined focused regression selection | 61 passed in 3.03 s |
| Beta moments (powers 1–5), scalar and batch | Passed at relative tolerance 1e-11, zero absolute tolerance |
| Tiny positive shapes | Includes 2**-60 / 2**-61, shapes down to 1e-300, and asymmetric tails |
| Recurrence singularity | a+b=1 and adjacent floating-point inputs; one-node sanity check |
| Histories | x=0/1/44/45; both scalar and accelerated paths |
| Independent reference | 90-digit Decimal integrated binomial series with analytic Beta moments; 32 histories, relative tolerance 2e-11 |
| p=0 and subnormal positive p | Continuous limit passed; endpoint evaluations independently checked at 90 digits |
| Ordinary frozen monthly formula | Max relative change 3.6066261088762985e-11 across 1,080 values; frozen tolerance 1e-10 |
| Ordinary fast window vs frozen formula | Max relative change 1.389555137620846e-12 across 90 histories; frozen tolerance 2e-11 |
| Handcrafted point-prediction frame | Max relative change 1.9206858326015208e-13; identical four-decimal formatting on this fixture |
| P7 and P8 outcome streams | Identical outcome arrays and RNG states against checkpoint formulas on four synthetic customers, 20 draws per path |
| A14 zero-dropout, small-a and input-validation tests | Passed in focused selection |
| Ruff | Full repository passed |
| Focused mypy | Four changed implementation files passed with imported-module diagnostics suppressed |
| Full-source mypy | 38 errors in five untouched files; identical error diagnostics on checkpoint source extracted into a temporary directory |
| Frozen protocol | Manifest/code digest pins, internal consistency and amendment timing tests passed; no A15 activated |

The reference series is independent of the recurrence and eigensolvers; its test domain has
right/(alpha+T) < 1. It is a deterministic test oracle, not a replacement production algorithm.
The fixtures do not exhaust all representable parameter draws or numerical-library versions.
Two existing predictive-cases tests initially could not be collected because their test module imports
the absent statsmodels package; this limitation is resolved by the cross-environment validation below.
Handcrafted checkpoint-comparison tests cover the relevant
scalar/fast calculation and random-stream behavior. Dependencies and unrelated type errors
were not changed as part of this numerical repair.

The combined selection comprised both new test modules, the six numerical/task-identification
test functions in `test_p8_a14.py` (including parametrizations), the existing Beta-moment,
published-closed-form, doubled-rule and estimator-isolation tests in `test_btyd_models.py`, and
the internal-consistency, two digest-pin and amendment-timing tests in
`test_predictive_protocol.py`. It excluded final-mode execution and governance tests that
construct fake final attempts. Pytest used `PYTHONPATH=src`, one BLAS/OpenMP thread, and
`-p no:cacheprovider`. Ruff used `python3 -m ruff check .`; focused mypy used
`--follow-imports=silent` on `failure_trace.py`, `btyd.py`, `predictive_cases.py` and
`scripts/generate_m11_evidence.py`. Full-source mypy used `python3 -m mypy src`, with its cache
placed in a temporary directory, on both working source and checkpoint source.

## Impact on frozen methods and evidence

This changes the shared `bgnbd_conditional` calculation, so it is **not restricted to R2**.
E4 point predictions and P7 expected simulated values can change numerically. The ordinary
fixture changes meet the existing tolerances, while formerly undefined boundary cases now
produce finite expectations. Existing P7 documents and all committed evidence remain intact;
byte-identical regeneration was not demonstrated in the initial development environment. The
cross-environment validation below reproduced all committed statistical sections before the
two targeted weight and alive-probability follow-ups;
metadata and runtime differences are listed explicitly.

PC4, non-contractual selection and R3 consume E4 point predictions. Their numerical inputs can
therefore change. Their estimands, comparison definitions, gates, tolerances, Holm family and
acceptance rules are unchanged. No claim is made that final p-values, selections, coverage or
gate outcomes are unchanged: none were inspected or recomputed. Near a decision boundary even
a small numerical input change could matter. There are no new RNG call sites or changes to call
order or the parameter-draw mechanism. Ordinary P7 and P8 fixtures retained bit-identical
sampled outcomes and post-simulation RNG states. A corrected extreme alive decision can
change a purchase count and a subsequent Gamma shape. A variable-consumption sampler can
then change the RNG state and downstream draws. Universal bitwise RNG equivalence is not claimed.

## Proposed owner-review wording

> A15 corrects E4 Beta quadrature and its continuous window integrand, accelerated Beta
> reweighting and BG/NBD alive probabilities. The frozen statistical design and A14 protections
> continue to apply. Deterministic fixtures establish the defects. The cause of the recorded
> final-a14 crash remains unconfirmed. R2 fails on nonfinite E4 coverage inputs and future
> failure ledgers retain trace locations without outcome payloads. Both attempts remain
> permanently consumed. Any third attempt requires separate explicit owner authorisation.

Record the actual approval date and `before_main_seed=False` if approved. Both attempts had
completed main before A15. Record `before_outcome_inspection=True` only if the owner confirms
that no protected main, fresh, coverage-block or refit-block outcome from either attempt was
inspected. The review verifies the recorded timeline and its own access scope. It cannot attest
to other operators' access.

## Preservation and any future execution

| Consumed reservation | Main completed (UTC, 2026-11-20) | Repeated sampling failed (UTC) | Ledger error |
|---|---|---|---|
| `final` | 01:04:52 | 03:23:18 | `ValueError: p <= 0, p > 1 or p contains NaNs` |
| `final-a14` | 09:50:17 | 12:07:20 | R2, seed 810085, scale 1, refits 0. `ZeroDivisionError: float division by zero` |

Both ledgers end in repeated sampling. Neither records a completed repeated-sampling phase,
M10 comparability or fresh-seed consumption. Preserve both failed directories, raw artifacts,
reservations and the existing registry-chain prefix permanently. The checksum table below
identifies the committed evidence.

The A14 handover discloses that the first `raw/main.pkl` was previously read as raw bytes for
SHA-256 verification. The handover records that the pickle was never deserialised and no
outcomes were inspected. A15 development and review did not read either raw artifact.

If the owner later authorises a third attempt, apply the following controls:

- Approve the amendment text, implementation commit, manifest and code digests, and approved
  environment. The protocol code digest covers selected decision files, so verify the
  implementation commit and a clean working tree as separate checks.
- Require both exact failed ledger paths, hashes and failure states, including checking that
  neither is missing. Verify both consumed reservations, the canonical registry chain and the
  absence of unknown prior attempts. Review the outcome-blindness attestation separately.
- Bind any authorisation to one new single-use key, one fresh output directory and the
  repository's canonical registry. Atomically append the new reservation under the registry
  lock before any computation, with the existing registry history preserved. Use fake
  development fixtures to check the new gate before any reserved-seed execution.
- Run main, repeated sampling and M10 comparability in the frozen order. Authorise and record
  the fresh seed only when the final phase is reached. Preserve all frozen seeds, counts,
  scales, estimands, comparisons, tolerances, Holm family and acceptance rules. Coverage stays
  descriptive, without recalibration, replacement seeds or filtering of failed draws.
- If the authorised attempt fails or is interrupted, retain the evidence and stop. Further
  execution requires another explicit amendment and authorisation. Completed results remain
  protected until full evidence verification under the approved procedure.

## Cross-environment validation — 2026-11-21 (no approval or activation)

This section records the baseline validation before the targeted weight and alive-probability
follow-ups. The completed patch's focused verification is recorded below.

Only this draft was edited during this validation. The uncommitted implementation and tests
were retained. No packages were installed, no repository evidence was overwritten, no final
mode was executed, and neither failed raw/main.pkl was accessed. No protocol value or digest
was changed. All additional scripts, checkpoint extracts and generated results are temporary:

`/private/tmp/m11-a15-cross-d954rh6c`

### Approved environment and checks

The previously used scratch environment was found at:

`/private/tmp/claude-501/-Users-privilege-Desktop-GitHub-repository-enterprise-applied-data-science-decision-intelligence/eae9bef3-5591-4283-8f18-10ccbfdc723e/scratchpad/venv312/bin/python`

Its interpreter and imported packages match the handover: CPython 3.12.8, NumPy 2.5.5,
SciPy 1.18.1, pandas 2.3.3, statsmodels 0.15.0. `PYTHONPATH=src` was verified to import this
checkout, rather than the environment's installed project. Tools are pytest 9.2.0,
Ruff 0.17.5 and mypy 2.4.1; pandas-stubs is 3.1.0.261029.

- All 61 original focused tests plus the two previously uncollected predictive-cases tests:
  **63 passed in 9.62 s**. The latter tests are
  `test_accelerated_and_fallback_rules_match_the_original_per_count_rule` and
  `test_fast_e4_simulation_is_equivalent_to_p7`.
- Full-repository Ruff: passed (`--no-cache`). Focused mypy on the four implementation files:
  passed. Full-source mypy: passed, 126 source files.
- Checkpoint `8e12c12`, extracted with git archive into temporary storage: full-source mypy
  passed, 125 source files, in the same approved environment.

The earlier 38 diagnostics were also reproduced afresh on both source trees using the initial
global environment (mypy 1.16.1, pandas-stubs 2.3.3.260113, NumPy 2.4.6, SciPy 1.17.1).
Sorted error diagnostics are exactly identical between checkpoint and working source:

| Untouched file | Errors in each source tree |
|---|---:|
| customer_value/backtest.py | 15 |
| data/interactions.py | 19 |
| data/predictive_ltv_scenarios.py | 2 |
| data/transactions.py | 1 |
| statistics/distributions.py | 1 |

Thus these diagnostics are environment dependent and pre-existing, and are not a blocker in
the approved environment. This comparison does not isolate which individual tool/stub/library
version caused each diagnostic; no packages or unrelated source files were changed.

### Complete P7 reproduction and exhaustive committed-record differences

The unmodified `report_m11_p7_evaluation.py` was imported and its `OUTPUT` redirected to
`p7-current.md` in the temporary directory. All of main(), including A11 confirmation, ran.
A wrapper asserted the development seed constants, refused reserved-seed authorisation and
guarded generator invocations. Actual scenario seeds were 5701, 5703, 5702, 5711, 5712, 5713;
bootstrap/simulation/censoring/PIT/noise seeds remained 5704–5708. Numeric threads were one.

**Sections 1–7, including 4a and 4b, are byte-identical to the committed P7 record.** The exact
26,695-byte statistical section span has SHA-256
`fdbb1b76810bc33057bbd36dd3fe76c10015c2f64c4ffc75faa5ebd62ed73ab4` in both files. This checks
every printed numerical digit, label, interval classification, selection and gate input in
those sections. The committed Markdown stores rounded values: this is not a claim that
unrecorded historical full-precision intermediate arrays are bit-identical.

Every other difference in the generated record is listed below; no statistical cells differ.

| Metadata field | Committed | Generated |
|---|---|---|
| protocol stage | P7-amended | P8-amended |
| manifest digest | `0a1ea3bdb7c60e0a82fd4b2f08ff7cde067da852ef69e5b0a3327921f4f608a8` | `27c949b491a25ee4f571f9f0243903a7d3d3e4c7ca2097db30abdf94dfb0ba08` |
| code digest | `244acc014ecd3acd5ca995e47bd3bc7866eb38946d1d220d7a8c68363d49ada0` | `8830e11ebb46a6f712c458f9dff23a27c6ab710ff1359d2dd256ee90ea0d8879` |
| Git description | 363ea12 + 15 uncommitted or untracked paths | 8e12c12 + 8 uncommitted or untracked paths |

These are historical-versus-current metadata, not digest repinning or A15 activation. Python,
package versions, platform, seed and draw-count metadata reproduce exactly.

| Runtime row (seconds) | Committed | Generated |
|---|---:|---:|
| A11 confirmation | 13.8 | 15.3 |
| B1 | 0.4 | 0.5 |
| CR1 | 1.8 | 2.5 |
| E1 | 0.4 | 0.6 |
| E1 simulation | 3.4 | 3.5 |
| E2 | 0.4 | 0.6 |
| E3 fit | 10.1 | 12.9 |
| E3 simulation | 7.1 | 9.6 |
| E4 | 1.4 | 1.9 |
| E4 count pmf | 3.8 | 5.0 |
| E4 simulation | 37.6 | 109.9 |
| E5 fit | 3.8 | 5.4 |
| S2 | 3.9 | 5.6 |
| selection | 9.5 | 10.6 |
| total | 124.2 | 216.1 |

B2 (0.0 s) and E2 simulation (2.5 s) are unchanged. The reproduction overlapped tests and
type checking, so these timings are not an isolated patch benchmark. The increase, especially
E4 simulation, must not be hidden; full P8 time/memory budget feasibility is still unverified.
No runtime target or ceiling was changed.

Temporary audit artifacts: `p7-current.md`, `p7-current.seeds.json`, `p7-comparison.json`
(every changed cell), and `p7-full.diff` (the complete, unfiltered textual diff).

### Synthetic comparisons against the actual checkpoint implementation

Separate processes imported either the working source or the extracted `8e12c12` source in
the same approved environment. No approximate reimplementation was used for this comparison.
Both used fixed four-customer histories x=0/1/44/45, known synthetic fits at three ordinary
shape pairs, identical labels/comparator/oracle arrays, and the frozen default 2,000 bootstrap
resamples. Additional fixed fixtures exercise both qualified selection candidates and R3
per-draw/aggregate calculations over three synthetic draws and all three frozen scales.
Only development RNG metadata/seeds 7004–7006 and synthetic identifiers 7100–7102 were used.
No scenario generator, reserved-seed task or actual repeated-sampling study ran here.

There are **391 scalar numerical comparisons: 144 bit-identical and 247 changed**. Every
scalar, including unchanged values, has its exact old/new float representation, hexadecimal
representation, absolute delta and relative delta in `synthetic-all-comparisons.csv`.
`synthetic-checkpoint.json` and `synthetic-current.json` retain the full fixture results;
`synthetic-summary.json` groups the deltas. The following summarizes the changes without
discarding the full comparison artifact:

| Quantity | Bit-identical / values | Max absolute delta | Max relative delta |
|---|---:|---:|---:|
| E4 expected spend | 12 / 12 | 0 | 0 |
| E4 P(alive at T) | 12 / 12 | 0 | 0 |
| E4 P(alive at window start) | 2 / 12 | 4.440892098500626e-16 | 1.852654288052919e-15 |
| E4 expected purchases | 0 / 12 | 9.39803790345195e-14 | 7.116350312216329e-13 |
| E4 point predictions / PC4 E4 inputs | 0 / 12 | 1.3153922395758855e-12 | 7.117156190200726e-13 |
| E4 discounted predictions | 0 / 12 | 1.120881165661558e-12 | 6.832433142439553e-13 |
| E4 monthly purchases | 0 / 144 | 3.406615267653734e-13 | 1.6272844805123327e-11 |
| PC4 B2 and realised inputs | 24 / 24 | 0 | 0 |
| PC4 paired squared-error differences | 6 / 12 | 2.2317703241014897e-12 | 7.962128212480622e-12 |
| PC4 test fields | 19 / 27 | 1.1530776333756876e-12 | 1.7246505590886644e-12 |
| Ordinary selection-candidate fields | 6 / 12 | 3.652189661806915e-12 | 3.381730386316279e-13 |
| Qualified selection fixture, including paired intervals | 5 / 14 | 1.965094753586527e-13 | 2.0892838441004942e-10 |
| R3 per-draw fields | 38 / 48 | 7.549516567451064e-15 | 7.880829787811406e-15 |
| R3 aggregate bias fields | 9 / 15 | 2.3252141545526683e-08 | 4.743423838897545e-12 |
| R3 aggregate slope fields | 5 / 5 | 0 | 0 |

All ordinary E4 monthly outputs meet the existing 1e-10 tolerance and total-value inputs meet
2e-11. All gate booleans and selection decisions are identical; the qualified fixture selects
E4 in both versions. R3's largest absolute difference is an amplified z value, with relative
change 4.743423838897545e-12. The qualified selection fixture's tiny MSE changes from
8.288889403610093e-06 to 8.288889401878308e-06 (absolute 1.7317842716498347e-15, relative
2.0892838441004942e-10). The prediction tolerance is not a specified tolerance for every
derived near-zero statistic: no new acceptance tolerance was imposed or relaxed. Both
bootstrap interval signs and all frozen decisions remain unchanged on these fixtures.

The extreme positive-shape fixture a=2**-60, b=2**-61 changes materially: the actual checkpoint
raises ZeroDivisionError; working source produces finite predictions
`[3.823249517566688, 0.10996993181629884, 242.78323607401245, 248.17333704099659]`.
There is no finite checkpoint value against which to assert relative equivalence. This proves
the repaired implementation boundary, not the historical final-a14 crash attribution.

### Immutable attempt evidence

During package review the registry's read-only `validate()` returned `[]`, with no lock. Its keys remain
`dry-run-a3cef63df948e378`, `final`, `final-a14`. The following files are byte-identical to
their `git show 8e12c12:<path>` contents:

| File | SHA-256 |
|---|---|
| `m11-p8-final/ledger.json` | `cd7c8391347df2f9a2cf802161f35a0cb6a9a889b29c8fce3825a22ebe5902a9` |
| `m11-p8-final-a14/ledger.json` | `74a0cdd66175ab19f5ecf0dddef08114230713897d5f968d6295df390b678008` |
| `m11-p8-attempts/final.json` | `e5dc98f0402381f5c73083278ca2cf140e6cf76fb77a3b6e5c5641f724c6ad04` |
| `m11-p8-attempts/final-a14.json` | `806a2b126e94a903a170e7ea50e7b2ae0a8dd7039852772df950dc7dc9cc7ae3` |
| `m11-p8-attempts/attempts.log` | `52c6031120ff02a58b6b773134f69ad9d48d322a20fb10ce9418c6af58eec2fb` |

The raw artifact hashes recorded in the ledgers are
`d514d59e4b8e813f23855c482f2f952818ebf202446bfef71a341b072cd23b11` for `final` and
`0f8087f3630f9689b41c94bae723cb5524f9e3edb161d688c806d38216b77a2f` for `final-a14`.
These values come from the ledger metadata. Neither raw artifact was read or rehashed for
this package.

## Completed patch validation

The approved Python 3.12.8 environment passed **76 focused tests**, Ruff and full-source
mypy after all three corrections. The focused suite adds the 13 weight, alive-probability
and coverage cases to the earlier 63. The 90-digit Decimal references cover positive
subnormal shapes and `x = 0, 1, 44, 45`. Scalar and batched alive probabilities matched
bit-for-bit on the added fixtures. The windows matched the reference within the existing
`2e-11` relative tolerance, allowing one binary64 unit at zero for subnormal rounding.
Ordinary regression tests retained their prior tolerances. All four fixture observations
remained in the coverage denominator. Injected NaN and positive or negative infinity in
interval bounds or targets raised. The R2 path also checks point predictions and raw
simulated arrays before computing coverage.

The prior P7 byte comparison and detailed PC4, selection and R3 checkpoint comparisons
describe the earlier patch revision. The completed patch passed the focused ordinary
prediction and simulation regressions. The full P7 record and the detailed statistical
comparisons were not regenerated after the two follow-ups. The committed P7 record does
not contain historical unrounded intermediate arrays.

## Resource benchmark on the completed patch

One parent and three spawned workers ran in the approved environment, with one numerical
thread per worker. R2 used development seeds 6201–6206 with 200 draws per task. R3 used
6201–6203 at each frozen scale 0.5, 1 and 2. All three workers ran concurrently. The first
trio's task intervals overlapped for 35.73 seconds in R2 and 1.60 seconds in R3. All three
CPU counters also advanced in shared sampling intervals.

| Phase | Completed tasks | Measured wall time | Measured throughput | Full P8 estimate |
|---|---:|---:|---:|---:|
| R2 | 6 | 74.49 s | 4.83 tasks/min | 103.46 min for 500 tasks |
| R3 | 9 | 11.42 s | 47.29 tasks/min | 12.69 min for 600 tasks |
| R1, REFIT, main, M10 comparability and fresh | — | — | — | 43.10 min from the committed dry run |
| Total | 15 | 85.93 s | — | 159.25 min |

Peak sampled aggregate RSS was **2,282,225,664 bytes (2.282 GB)**, including the parent,
workers and resource tracker. Samples were taken about every 0.1 second. This gives **memory
GO on the measured workload**, with 1.718 GB of observed headroom. Scaling R2 and R3
throughput and adding the previously estimated 43.10 minutes gives **runtime GO on
projection**, with 20.75 minutes below the 180-minute ceiling. The frozen memory limit is
4 GB. See the [committed dry-run projection](../evidence/m11-p8-dry-run/record.md#1-performance).
The current session's temporary audit outputs are
`/private/tmp/m11-p8-resources.68iCpz/result.json` and `samples.jsonl` in the same directory.

The benchmark does not measure full reserved-plan workload variance, parent retention and
serialization of all results, final reporting or transient RSS peaks between samples.
It used direct Darwin process sampling. The runner uses `ps` and can stop collecting if
that command fails. A later operator must verify that aggregate monitoring produces readings
in the actual execution environment and can act on the frozen limits.

## Remaining risks and decision

- **Amendment approval:** The numerical evidence supports the three corrections. Owner
  confirmation of outcome blindness across both attempts is required before recording
  approval under the existing amendment rule. Historical crash attribution remains open.
- **Execution preflight:** The current allowlist checks prior ledgers that it discovers.
  It does not require both expected files to be present. A separate authorised execution
  change must implement and exercise the positive checks and single-use gate specified
  above. The legacy test `test_the_repository_is_ready_for_exactly_the_authorised_retry`
  still expects `final-a14` to be unused. It is outside the 76 focused tests and must be
  updated for the preserved evidence and the authorised gate before execution.
- **Numerical limits:** Deterministic fixtures and sampled development draws do not exhaust
  valid shape and history combinations. In particular the existing Gamma-Gamma spend
  mean requires posterior shape `p_s*n+q_s > 1`. A draw below that bound can produce a
  finite negative value that the new nonfinite coverage guard cannot detect. Its reachability
  under the frozen reserved plan has not been established. There is also residual precision
  risk when very large rates round to nearly equal logarithms in alive odds. Neither issue
  has been tied to either historical failure. Review these limitations explicitly before
  authorising execution. This package does not change the draw law or add an exclusion rule.

This package updates only this draft. The implementation, tests, protocol, digest pins and
committed evidence retain their reviewed state. Owner approval of the amendment and any
subsequent execution authorisation are separate decisions.
