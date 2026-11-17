# LTV foundations and customer economics (Milestone 10)

Pillar 5's first milestone: historical and heuristic customer lifetime value on a contribution
basis, retention curves, acquisition cost, payback and LTV/CAC — with stated assumptions,
uncertainty, time-respecting backtests and comparison with synthetic ground truth. Code:
`src/decision_intelligence/customer_value/` and
`src/decision_intelligence/data/customer_economics_scenarios.py`. Evidence:
[../evidence/m10-ltv-foundations.md](../evidence/m10-ltv-foundations.md).

Predictive, probabilistic (BG/NBD, Gamma-Gamma), survival-model and machine-learning LTV are
**Milestone 11**; causal (incremental) LTV is deferred to a future milestone and is not part of
M11. None of them is implemented here.

## 1. Three quantities that must not be confused

| Quantity | Meaning | Where |
| --- | --- | --- |
| **Realised** contribution | What was actually recorded (historical) | `historical.py` |
| **Projected** contribution | A heuristic forecast over a stated finite horizon (24, 36 months) | `heuristic.py`, `backtest.py` |
| **Expected lifetime** contribution | A heuristic's expectation to an infinite horizon, defined only when the sum converges | `heuristic.py` |

None of them is the *incremental* value of an intervention: a customer with high LTV may have
stayed without any offer (incremental LTV is deferred to a future milestone; not part of M11).

## 2. Economic conventions (`economics.py`)

- **Periods**: monthly. Age 0 is the acquisition month; every paid month is billed at its start.
- **Discounting**: a cash flow at age `a` is multiplied by `d^a`, `d = 1/(1 + r_m)`,
  `r_m = (1 + R)^(1/12) − 1`, `R = 10%` a year; age 0 is not discounted. A horizon of `T` months
  covers ages `0..T−1`.
- **Contribution** of a paid month = revenue − cost of goods − payment fee − cost-to-serve.
  Revenue is never treated as economic value; both are reported.
- **Geometric value**: per-month value `m`, continuation probability `q`, `s = q d`:
  finite `m (1 − s^T)/(1 − s)`; infinite `m/(1 − s)` only when `s < 1` — otherwise the estimate
  is returned with status `undefined` (e.g. no churn and no discounting).
- **CAC**: acquisition cost ÷ new customers over the same cohort months — *media* (media spend)
  or *fully loaded* (media + onboarding + marketing overhead allocated by new-customer share).
- **Payback**: the first age at which cumulative contribution per acquired customer (churned
  customers included as zeros) reaches CAC; from the realised curve where observed, otherwise
  labelled *H4-projected*.

## 3. Data (`data/customer_economics_scenarios.py`)

An additive generator (M2 is unchanged): 48 months of monthly cohorts across four channels with
known cost per acquisition and segment mix; three segments with price, cost-to-serve and churn;
first-months promotions for paid channels; payment fees; noisy cost-to-serve; payments recorded
0–5 days after billing and 5% recorded 20–40 days later; cancellations recorded days after the
last paid month. Churn: a constant monthly hazard per customer, `λ_i = b_{s,ch} Z_i`, with Gamma
heterogeneity `Z ~ Gamma(1/v, v)` — so the *population* hazard falls with tenure. Ground truth
(per-customer hazards and analytic population expectations via the Gamma Laplace transform
`E[e^{−bZa}] = (1 + vba)^{−1/v}`) is returned separately and read only by evaluation code.
Designed-in mechanisms (premium: highest price, high cost-to-serve; social: cheapest acquisition,
worst customers) are disclosed as such, not presented as findings.

## 4. Point-in-time and censoring rules (`panel.py`, `retention.py`)

- At a month-end cutoff only records with `recorded_at` at or before it are visible; duplicated
  transaction rows are dropped; months whose payments may still be unrecorded (within 45 days)
  are excluded from "complete" calculations.
- A customer is churned if a cancellation is recorded by the cutoff, otherwise active and
  presumed to have paid the cutoff month.
- **Kaplan–Meier on paid months**: churned customers are events at their number of paid months
  `t`; active customers are censored at `t − 1`, because the churn decision after their latest
  paid month has not been observed. Greenwood standard errors; plain and log-log intervals;
  valid under non-informative (here administrative) censoring.
- **Cohort retention tables** report only ages a cohort has fully reached.
- **Constant churn**: events / trials over the trailing 12 months, where each paid month whose
  outcome is known is a trial.
- **M2 compatibility** (continuous tenure in days): a churn is an event at `ceil(days/30)`; an
  active customer is censored at the whole months survived, `floor(days/30)`.

## 5. Historical and heuristic LTV

- **Realised, cohort-matched** `A`-month value: discounted contribution of ages `0..A−1`, zeros
  after churn, only for customers whose whole window is completely recorded; its mean over all
  such acquired customers is unbiased for the expected `A`-month value per acquired customer.
  Counterexamples: *all customers* (truncated windows; truncation bias) and *survivors only*
  (complete-window customers who paid every month; survivorship bias).
- **H0** legacy: historical revenue + churn-adjusted 12-month revenue projection (undiscounted).
- **H1/H2**: pooled constant churn and run-rate contribution; infinite / finite horizon.
- **H3**: H1/H2 per segment × channel cell, mixed by recent acquisitions.
- **H4**: Kaplan–Meier curve × mean contribution by age while at least 50 customers are at
  risk, then a geometric tail at the hazard of the last 12 observed ages.
- **Status**: `undefined` (no data, or a divergent infinite horizon), `unstable` (fewer than 30
  churn events), otherwise `ok`.

## 6. Uncertainty

Customer-level percentile bootstrap (M3) for realised LTV and LTV/CAC (spend treated as known);
delta-method standard error for H1 (churn by the binomial variance, run rate by the ratio
linearisation) compared with a customer bootstrap; Greenwood intervals for retention;
one-factor-at-a-time sensitivity tables for discount rate, gross margin and cost-to-serve.
Repeated generator draws check interval coverage against the population truth. The LTV/CAC
interval divides the LTV interval by the realised CAC, so it is conditional on spend; against the
true LTV/CPA it omits CAC sampling error and under-covers (evidence §A1, where adding a
cohort-month bootstrap variance for CAC is shown, as a diagnostic, to restore coverage).

## 7. Backtests (`backtest.py`)

A backtest at cutoff `C` with window `W` runs only if months `C+1..C+W` exist in the data and are
read after the 45-day recording lag; otherwise it is refused (`IncompleteWindow`). *Existing
customers*: per-customer 12-month projections (no-churn baselines, constant churn, Kaplan–Meier
conditional survival) vs realised contribution and the truth's expectation, on one comparison
set (customers every method can project); customer-level ranking is reported as a **baseline**
only. *New cohorts*: per-channel projected 12-month value at the cutoff vs the realised value of
later cohorts with complete windows.

## 8. Limitations

- Synthetic contractual subscriptions only; economics (costs, promotions, CAC) are assumptions in
  fictional currency. Non-contractual purchasing is M11.
- Constant-churn heuristics cannot represent a population hazard that falls with tenure; H4's
  tail extrapolation also under-estimates expected lifetime value when heterogeneity is large.
- Run-rate inputs take customers still in a promotion at their promotional contribution, which
  biases existing-customer projections downwards (evidence §A3).
- The ±15% precision target is per channel; subgroup (segment × channel) precision is
  supplementary and can fail in small cells.
- LTV/CAC intervals treat spend as known; their coverage of the true LTV/CPA is below nominal.
- Single-touch channel attribution; no seasonality or price changes; no reactivation.
- Customer-level ranking from heuristics is weak by design — predictive LTV is M11.
- Nothing here is incremental (causal) value.
