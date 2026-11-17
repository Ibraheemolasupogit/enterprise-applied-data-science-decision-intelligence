# Milestone 10 evidence — LTV foundations and customer economics

Status: **IMPLEMENTED — awaiting scientific assurance review**. Not committed or pushed.
Statuses: [capability-status.md](capability-status.md). Methodology:
[../methodology/customer-economics.md](../methodology/customer-economics.md). Generated numbers:
[m10-ltv-foundations-results.md](m10-ltv-foundations-results.md)
(`python scripts/generate_m10_evidence.py`; reproducible apart from the runtime table).

## Acceptance criteria (approved plan P0–P6)

| Requirement | Evidence |
| --- | --- |
| P0 frozen protocol, generator with truth, precision plan before main results | `customer_value/protocol.py`; generator truth vs Monte Carlo and the Gamma Laplace transform (tests); results §1–2 |
| P1 economic definitions, point-in-time panel | `economics.py`, `panel.py`; closed forms vs brute force; post-cutoff perturbation invariance; late payments and duplicates (tests) |
| P2 retention curves | `retention.py`; KM and Greenwood vs the published 6-MP example; KM = empirical survival without censoring; results §4 |
| P3 historical and heuristic LTV with uncertainty | `historical.py`, `heuristic.py`; results §5, §6, §9, §10 |
| P4 CAC, payback, LTV/CAC, segment economics | `economics.py`, `cases.py`; results §7, §8 |
| P5 backtests, calibration, sensitivity, M2, legacy | `backtest.py`; results §11, §12; legacy H0 in §6 |
| P6 evidence, documentation, validation | this file; see Validation |
| Safeguards: separate realised / projected / expected lifetime; infinite-horizon convergence and undefined/unstable handling; complete windows; truth separation | methodology §1, §2, §5, §7; AST test over estimator modules; results §6 (convergence, refusal), §11 (excluded cutoff) |

## Headline results (main seed unless stated)

| Finding | Number |
| --- | --- |
| Data | 14,932 customers; 174,562 transactions; 9,401 observed churns, 5,531 censored; 8,667 late-recorded payments |
| Kaplan–Meier vs truth | Greenwood intervals contain the population truth in 31 of 32 group-age cells (single realisation) |
| Realised 24-month value per acquired customer | organic 76.35, paid search 65.90, social 38.81, referral 93.46 (truth 78.57 / 65.68 / 40.29 / 93.85); \|z\| ≤ 2 in 7 of 8 cells (12 and 24 months) |
| Naive historical counterexamples | all-customers mean below the truth in 8/8 cells (truncation); survivors-only above in 8/8 (survivorship) |
| Heuristic bias, projected (24/36 m), mean \|relative\| | H1 0.121, H3 0.106, **H4 0.011** |
| Heuristic bias, expected lifetime, mean \|relative\| | H1 0.084, H3 0.071, H4 0.080; every heuristic under-estimates in every channel and segment — H1 and H3 by −0.03 to −0.11, H4 by −0.01 to −0.13; H4 is no better than H3 on this quantity |
| Revenue vs contribution (designed-in) | premium ranks 1st by revenue, 2nd by contribution (standard 1st), matching the contribution truth |
| Stability | 70 heuristic estimates all `ok`; 0 of 12 segment × channel cells unstable; infinite-horizon convergence: H1 finite-horizon gap −9.4e-7 at 240 months; no-churn, no-discount case refused as `undefined` |
| LTV/CAC (media) with 95% CI | organic 2.12 [2.04, 2.19], paid search **0.94** [0.90, 0.97], social 1.66 [1.59, 1.73], referral 2.13 [2.05, 2.21]; fully loaded: paid search 0.76 and social 0.99 (< 1) |
| Precision target (±15%) | met in 4/4 channels (half-widths 3.4–4.2%); met in 11/12 segment × channel cells — **not met** for premium/social (16.2%, 138 customers) |
| Cheapest CAC vs best LTV/CAC (designed-in) | social has the lowest media CAC (23.43); referral has the best LTV/CAC (2.13) |
| Payback (discounted, media CAC, realised) | organic 8, paid search 27, social 12, referral 9 months |
| H1 uncertainty | delta-method SE is 4–10% smaller than the bootstrap SE (ratio 0.90–0.96) |
| Coverage, 200 generator draws | LTV24: 0.935, **0.905** (paid search, p = 0.008), 0.945, 0.960; **LTV/CAC: 0.79–0.85** (all p < 1e-7) |
| Heterogeneity grid (expected lifetime bias) | H1: −0.015 (v = 0), −0.082, −0.109, **−0.247** (v = 1); H4: −0.007, −0.051, −0.100, −0.070; projected 36-month H4 within ±0.01 at every level |
| Customer backtests (12 m; baseline; cutoff 36 excluded) | KM by cell −14.5% / −13.1% vs realised (cutoffs 24 / 30), driven by the run-rate input during promotions, not by the survival curve (audit §A3); historical-average no-churn +3.5% / +4.1%; run rate +14.8% / +14.3%; Spearman 0.36–0.46; top-decile capture 0.117–0.128; realised within 1% of the truth's expectation |
| Cohort backtests (12 m, new customers) | H4 within 2.5 SE of realised and within 6% of the truth in every channel; H2/H3 over-project paid search (43.99 vs 38.03 realised at cutoff 24) and social |
| Sensitivity | premium expected lifetime ranges 50 (cost-to-serve +20%) to 219 (−20%) |
| M2 compatibility | at most 12 observed months; KM 12-month retention 0.156; monthly churn 0.144; H1 gross-margin lifetime 31.2 (no cost-to-serve or CAC in M2) |
| Fresh seed 20261003 (once) | LTV/CAC 2.20 / 0.93 / 1.66 / 2.10, every true LTV/CPA inside its interval; precision met 4/4; realised LTV24 \|z\| ≤ 1.7; H4 projected bias within ±3%, expected-lifetime −0.04 to −0.15; same backtest pattern (cutoffs 24 and 30 only — 36 excluded by the same rule, not restated in §13) |

## Negative findings and failures (reported, not hidden)

1. **LTV/CAC intervals under-cover** (0.79–0.85 against the true LTV/CPA): the interval treats
   acquisition spend as known, but CAC is estimated from noisy spend. On the main seed the organic
   truth (2.245) lies outside its interval [2.043, 2.186]. The audit (§A1) attributes the whole
   shortfall to the omitted denominator uncertainty: no estimator bias, no ratio instability.
2. **LTV24 coverage for paid search is significantly below 95%** (0.905, p = 0.008); the others
   are consistent with 95%. Replicated on 1,000 independent audit draws (0.927, p = 0.002) with
   no significant bias; the customer bootstrap SE is about 5% below the Monte Carlo SD there.
   The cause is not fully explained.
3. **Every heuristic under-estimates expected lifetime value** under churn heterogeneity (H1
   by about 25% at the largest heterogeneity). H4's geometric tail, fitted to the last 12 observed
   ages, does not fix this: −0.01 to −0.13 on the main seed (mean \|bias\| 0.080 vs H3's 0.071)
   and −0.04 to −0.15 on the fresh seed. H4's advantage is confined to finite horizons.
4. **Constant-churn heuristics over-project 24–36-month value** by 7–19% on the main seed (up to
   22% on the fresh seed): they cannot represent churn concentrated in the first months.
5. **Existing-customer projections are biased**: the Kaplan–Meier-by-cell projection
   under-projects by 13–15%, and no heuristic ranks customers well (Spearman ≤ 0.46) — expected
   for a baseline; predictive LTV is M11. *Corrected attribution (audit §A3)*: the survival part is
   accurate; the bias comes from the run-rate input, which takes customers still in their
   two-month 50% promotion at their promotional contribution.
6. **Precision target not met** in one small subgroup (premium/social; see audit §A4 — the
   protocol's target is per channel and is met in every channel).
7. The delta method under-states H1's standard error by 4–10%.

## Defects, protocol issues and disclosures

1. **Protocol error found before evaluation**: backtest cutoff 36 cannot have a complete 12-month
   window. Not edited out of the protocol; refused by the code and reported as excluded.
2. **Plan change**: the M3 Weibull function is not a reference for this generator; the truth is
   verified against the Gamma Laplace transform (numerical integration) and Monte Carlo.
3. **Corrections before any main-seed result** (development seeds and tests): the customer
   backtest compared projections and realised values on different customer sets; the
   survivors-only counterexample mixed survivorship and truncation (my docstring had claimed a
   direction that is not guaranteed); Kaplan–Meier evaluated 0 × ∞ when survival reached 0;
   deprecated timedelta construction.
4. **Defect found from main-seed dry-run output**: Kaplan–Meier censored active customers at `t`
   instead of `t − 1` paid months, biasing survival upwards (Greenwood intervals contained the
   truth in only 72% of cells). Fixed with a regression test; all results regenerated; no other
   setting was changed after the main-seed output was seen. The same change also corrected the
   M2-compatibility rule (§12), which had censored active M2 customers at `ceil(days/30)`; they
   are now censored at the whole months survived, `floor(days/30)`. The changed results are
   listed in audit §A2; several became *worse*, which is consistent with a correction rather
   than tuning.
5. **Pre-existing, not M10**: an intermittent NumPy "overflow encountered in multiply" warning
   raised inside pandas' timedelta conversion while the M2 pipeline builds its data (seen before
   in M2/M4/M8 tests); it appeared once in a dry run under warnings-as-errors and not on rerun.

## Independent reference comparisons (tests)

| Component | Reference | Agreement |
| --- | --- | --- |
| Geometric finite / infinite value | brute-force sums (to 20,000 terms) | rel ≤ 1e-9 |
| Monthly rate, discount factors, curve value, payback | definitions; brute-force search | exact / ≤ 1e-12 |
| CAC (media, fully loaded) | pandas recomputation with overhead allocation | exact |
| Kaplan–Meier and Greenwood | Freireich et al. (1963) 6-MP data as tabulated by Kleinbaum & Klein | all 7 survival values and SEs to 3 dp |
| KM without censoring | empirical survival | ≤ 1e-12 |
| Paid-month KM censoring | constant-hazard simulation (20,000 customers) | within 4 SE; censoring at `t` shown biased |
| Population truth | Gamma Laplace transform vs numerical integration; Monte Carlo of the generator | rel ≤ 1e-9; mean z within 4/√n |
| Constant churn | hand count of events and trials | exact |
| Cohort-matched value, H0 legacy, H1 | per-customer sums; hand computation; geometric closed form of the estimated inputs | ≤ 1e-12 |
| Backtest realised values | direct sums of billing-month contributions | ≤ 1e-12 |

## Tests

| File | Tests |
| --- | --- |
| `tests/unit/customer_value/test_economics_retention.py` | 21 |
| `tests/unit/customer_value/test_estimators.py` | 13 |
| `tests/unit/data/test_customer_economics_scenarios.py` | 9 |
| `tests/statistical/test_m10_economics_properties.py` | 6 |
| `tests/integration/test_m10_case_studies.py` | 6 |

55 new tests, including 3 added by the audit (§A5) (the plan estimated 90–120; every reference
check in the plan is covered, with fewer parametrised cases than estimated).

## Validation

See the completion report: Ruff, strict mypy on both dependency sets, the full test suite on
both, evidence reproducibility (sections 1–12 regenerated without the fresh seed on both NumPy
versions) and M1–M9 preservation.

## Scientific assurance audit

Run after the final evaluation, without re-running it or the fresh seed and without changing the
frozen protocol. Diagnostics use seeds that the protocol does not reserve. Scripts and outputs are
in the session scratchpad (`m10audit/`), not in the repository.

### A1. Why LTV/CAC intervals cover 79–85%

1,000 independent generator draws (seeds 60000–60999), with the same estimator and interval as
§7/§9. `log` errors are relative to the truth (population LTV24, true CPA):

| quantity | organic | paid search | social | referral |
| --- | --- | --- | --- | --- |
| numerator (LTV24) mean log error | −0.0007 | −0.0008 | −0.0007 | −0.0002 |
| denominator (media CAC vs CPA) mean log error | −0.0001 | −0.0004 | +0.0001 | −0.0003 |
| mean relative error of LTV/CAC | −0.0003 | −0.0000 | −0.0003 | +0.0006 |
| SD of log LTV24 / mean bootstrap SD | 0.0169 / 0.0174 | 0.0186 / 0.0178 | 0.0217 / 0.0218 | 0.0193 / 0.0192 |
| SD of log CAC (ignored by the interval) | 0.0201 | 0.0191 | 0.0206 | 0.0207 |
| correlation of the two errors | −0.000 | −0.002 | +0.002 | +0.015 |
| coverage: evidence interval (spend known) | 0.799 | 0.796 | 0.837 | 0.821 |
| coverage: same interval ÷ true CPA | 0.961 | 0.927 | 0.947 | 0.950 |
| coverage: numerator + month-bootstrap CAC variance (log scale) | 0.954 | 0.938 | 0.942 | 0.946 |
| coverage: cohort-month cluster bootstrap of the ratio | 0.940 | 0.919 | 0.927 | 0.925 |

- **Estimator bias**: none detectable in the numerator, the denominator or the ratio (all |z| < 1.4).
- **Interval construction**: the interval is the LTV24 bootstrap interval divided by the
  realised CAC. It is a valid interval for LTV24 ÷ the *realised* CAC (conditional on spend), and
  that is what it covers — the same rate as LTV24 itself. Against the true LTV/CPA it omits the
  denominator's sampling error.
- **Ratio instability**: none. CAC varies by about 2% (CV), the ratio's skewness is 0.04–0.19, and
  the mean ratio error is below 0.1%.
- **Denominator uncertainty**: the whole shortfall. The CAC error (monthly spend noise, 25
  cohort months) is as large as the numerator error. With both, the true SD of log LTV/CAC is
  0.026–0.030 against the interval's implied 0.017–0.022, so ±1.96 implied SDs span only
  ±1.3–1.4 true SDs, which predicts 81–85% coverage — matching the observed 0.79–0.85.
- Adding a month-bootstrap CAC variance restores coverage to 0.938–0.954 (no channel
  significantly below 95%). A joint cohort-month cluster bootstrap still under-covers
  (0.919–0.940; 25 clusters). These are **diagnostics, not adopted**: the frozen protocol's
  interval and its reported coverage are unchanged.
- **Paid-search LTV24**: the shortfall replicates (0.927, p = 0.002) with no significant bias
  (z = −1.3); the bootstrap SD is about 5% below the Monte Carlo SD there. The cause is not fully
  explained.

### A2. Kaplan–Meier censoring correction

- **Original defect**: `kaplan_meier(paid_months, churned)` treated an active customer with `t`
  paid months as surviving past `t`. The churn decision after month `t` has not been observed,
  so the customer is known only to have more than `t − 1` paid months. Survival was biased
  upwards.
- **Correction**: `paid_month_km` censors active customers at `t − 1`. Churned customers remain
  events at `t`.
- **Independent references** (tests):
  - a hand life table (at risk 5/3/1; S = 0.8, 0.533, 0; Greenwood by hand);
  - statsmodels `SurvfuncRight` on 400 random customers: survival and Greenwood SEs agree to
    1e-12, except that where S = 0 statsmodels returns NaN and this code returns 0 (documented);
  - a constant-hazard simulation in which the old rule's bias is detected;
  - the published 6-MP example.

  An audit check on seed 70001 found the conditional survival used by the backtest within 0.4%
  of the truth.
- **When it was found**: from the main-seed dry run, before the fresh seed was used. That is a
  post-main-seed correction. Every main-seed result was regenerated, and no other setting
  changed.
- **Unaffected results**: realised values, CAC, LTV/CAC, precision, realised payback, coverage
  study, H1/H3.
- **Changed results** (main seed, before → after):

| result | before | after |
| --- | --- | --- |
| Greenwood intervals containing the truth (32 cells) | 72% | 97% |
| H4 projected, mean \|bias\| | 0.0147 | 0.0114 |
| H4 expected lifetime, mean \|bias\| | 0.0527 | 0.0802 (worse) |
| H4 lifetime bias by channel | −0.090, +0.025, −0.071, −0.070 | −0.128, −0.032, −0.115, −0.107 |
| Heterogeneity grid, H4 lifetime (v = 0 / 0.25 / 0.5 / 1) | +0.038 / +0.003 / −0.047 / +0.007 | −0.007 / −0.051 / −0.100 / −0.070 |
| Customer backtest, KM by cell vs realised (cutoff 24 / 30) | −12.5% / −11.5% | −14.5% / −13.1% (worse) |
| Cohort backtest H4, paid search (cutoff 24 / 30) | 39.58 / 39.60 | 38.92 / 39.06 (realised 38.03 / 36.26) |
| Premium expected lifetime, sensitivity base | 141.7 | 134.8 (truth 147.7) |
| M2: max tenure, KM 6 m, KM 12 m, churn, H1 gross margin | 13, 0.452, 0.271, 0.131, 34.2 | 12, 0.415, 0.156, 0.144, 31.2 |

The M2 12-month retention now rests on 12 customers at risk (4 events), because M2 covers 365
days. It is unstable and is reported for compatibility only.

### A3. Customer-backtest attribution (correction to this document)

The earlier text attributed the −13% to −15% bias of the Kaplan–Meier-by-cell projection to
conditional survival. An audit diagnostic (seed 70001, cutoff 24) shows:

- KM conditional paid months over the window: 8.78 per customer, against a true 8.81;
- run rate × the *true* survival: still −10% against realised;
- by channel and tenure, the bias sits in paid-search and social customers within their first
  five months — the two-month 50% promotion makes their run rate near zero or negative (for
  example −110% for paid search at ages 0–2). Customers past the promotion are within about
  ±10%.

The run-rate definition is part of the frozen protocol and a baseline, so it was **not
changed**. The finding is a limitation of run-rate inputs under promotions.

### A4. Precision: channel target vs subgroup

The protocol's ±15% target is defined **per channel** (`protocol.PRECISION_TARGET`), and it is
met in all four channels on the pilot (3.5–4.3%), main (3.4–4.2%) and fresh (3.5–4.2%) seeds. The
segment × channel table is a **supplementary subgroup analysis**: 11 of 12 cells meet the
target, and premium × social (138 customers) does not (16.2%). Subgroup LTV/CAC uses the channel's
CAC, because spend is not observed by segment. All of these half-widths treat spend as known
(§A1). Including CAC uncertainty, a realistic channel half-width is about 5–6%, still within
target.

### A5. Acceptance criteria → tests

| Criterion | Tests (file :: test) |
| --- | --- |
| Generator and truth | data :: `truth_matches_monte_carlo…`, `customer_truth_is_consistent…`, `same_seed_reproduces…`, `paid_months_are_contiguous…`, `invalid_configurations…`; retention :: `population_survival_is_the_gamma_laplace_transform` |
| Truth separation, reserved seeds | retention :: `estimators_never_read_ground_truth`, `no_test_uses_the_reserved_seeds` |
| Discounting, contribution, geometric value, convergence/undefined | retention :: `monthly_rate…`, `geometric_value_matches_brute_force`, `infinite_horizon_is_undefined…`; estimators :: `h4_finite_horizon_converges…`; integration :: `convergence_uncertainty…` |
| CAC, payback | retention :: `channel_cac_matches_a_pandas_recomputation`, `curve_value_and_payback…` |
| Point-in-time, lags, duplicates, censoring status | estimators :: `observable_view_ignores…`, `late_payments…`, `status_matches_the_generator…`, `backtest_projection_ignores_post_cutoff_data` |
| Kaplan–Meier, Greenwood, discrete-time censoring | retention :: `kaplan_meier_reproduces_the_published_6mp_example`, `…without_censoring…`, `paid_month_km_censors…`, **`paid_month_km_matches_a_hand_life_table_and_statsmodels`** (audit); statistical :: `greenwood_interval_covers…` |
| Cohort retention, constant churn | estimators :: `cohort_table_reports_only_completely_reached_ages`, `constant_churn_counts_events_and_trials_by_hand` |
| Realised LTV and counterexamples | estimators :: `cohort_matched_value_equals…`, `naive_versions_select…`; statistical :: `cohort_matched_realised_value_is_unbiased`, `naive_historical_values_are_biased…` |
| Heuristics H0–H4, stability states | estimators :: `h1_is_the_geometric_closed_form…`, `heuristics_flag_unstable_and_undefined…`, `h0_legacy_matches…`; statistical :: `constant_churn_heuristics_underestimate…` |
| Unit economics, precision rule, LTV/CAC interval construction | integration :: `heuristics_unit_and_segment_economics`, **`precision_flags_and_ltv_cac_interval_construction`** (audit) |
| Uncertainty and coverage | statistical :: `bootstrap_interval_for_24_month_ltv_covers_the_truth`; integration :: `convergence_uncertainty…` (delta method) |
| Complete-window backtests, one comparison set, invalid cutoff | estimators :: `backtest_refuses_incomplete_windows…`; statistical :: `realised_backtest_windows_agree…`; integration :: **`invalid_cutoff_is_excluded_consistently_from_every_backtest`** (audit) |
| Case studies, M2 compatibility, reproducibility | integration :: `audit_retention_and_historical`, `convergence_uncertainty_backtests_sensitivity_m2`, `case_studies_are_reproducible` |

Not covered by a test, by design: the repeated-sampling coverage *levels* (an evidence result,
not a pass/fail property; asserting them would amount to forcing coverage).

### A6. Numerical claims

Every number in this document, the methodology, README, registry and roadmap was checked against
the generated results. Corrections made:

- H4 expected-lifetime bias (the text quoted pre-fix dry-run values, −0.09 to +0.04; the
  generated value is −0.13 to −0.01, and H4 is not better than H3 on average);
- the H1/H3 lifetime range (−0.12 → −0.11);
- the customer-backtest attribution (§A3);
- the M2 rule change, which had not been disclosed;
- test counts.

The cutoff-36 exclusion is stated in §1 and §11 of the results and in this document. The §13
fresh-seed table contains only cutoffs 24 and 30, as the rule requires.

## Limitations

See methodology §8. Synthetic contractual subscriptions only; assumed economics; heuristics
cannot represent tenure-dependent population hazards; LTV/CAC intervals ignore CAC estimation
noise; single-touch attribution; customer-level ranking is a baseline; nothing here is
incremental (causal) value — that, and predictive / probabilistic LTV, is Milestone 11.
