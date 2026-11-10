"""Optional development cross-check of M5 estimators against external reference packages.

Not part of CI and not a project dependency: ``rdrobust`` pulls in matplotlib
and plotnine, and ``wildboottest`` depends on numba, which lags NumPy
releases. Install them in a scratch environment to run this script
(``rdd`` is optional and only used for the IK-bandwidth comparison):

    pip install rdrobust wildboottest rdd
    python scripts/crosscheck_m5_external.py

It prints the comparisons recorded in docs/evidence/m5-observational-causal-inference.md.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings

import numpy as np
import pandas as pd


def main() -> int:
    missing = [m for m in ("rdrobust", "wildboottest") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"skipped: optional packages not installed: {', '.join(missing)}")
        return 0
    warnings.simplefilter("ignore")
    from importlib.metadata import version

    import statsmodels.formula.api as smf
    from rdrobust import rdrobust
    from wildboottest.wildboottest import wildboottest

    from decision_intelligence.causal_inference import rdd
    from decision_intelligence.causal_inference.cluster_inference import wild_cluster_bootstrap
    from decision_intelligence.data.rollout_panel import RolloutConfig, generate_rollout_panel
    from decision_intelligence.data.threshold_assignment import generate_threshold_data

    print(f"rdrobust {version('rdrobust')}, wildboottest {version('wildboottest')}")

    data = generate_threshold_data()
    x, y, c = data.frame["score"].to_numpy(), data.frame["spend"].to_numpy(), data.config.cutoff
    h = rdd.ik_bandwidth(x, y, c)
    ours_ll = rdd.local_linear_rdd(x, y, c, h)
    ours_rbc = rdd.robust_bias_corrected_rdd(x, y, c, h)
    ref = rdrobust(y, x, c=c, h=h, b=h, kernel="triangular", p=1)
    print(f"RDD at h = {h:.4f} (IK):")
    print(f"  local linear      ours {ours_ll.estimate:.10f}  rdrobust conventional "
          f"{float(ref.coef.iloc[0, 0]):.10f}")
    print(f"  bias-corrected    ours {ours_rbc.estimate:.10f}  rdrobust (rho = 1) "
          f"{float(ref.coef.iloc[1, 0]):.10f}")
    print(f"  robust SE         ours {ours_rbc.standard_error:.4f} (HC1)  rdrobust "
          f"{float(ref.se.iloc[2, 0]):.4f} (nearest-neighbour)")
    hc1 = rdrobust(y, x, c=c, h=h, b=h, kernel="triangular", p=1, vce="hc1")
    print(f"  with vce='hc1':   conventional SE ours {ours_ll.standard_error:.6f} rdrobust "
          f"{float(hc1.se.iloc[0, 0]):.6f}; robust SE ours {ours_rbc.standard_error:.6f} "
          f"rdrobust {float(hc1.se.iloc[2, 0]):.6f} (HC1 dof factor pooled vs per side)")
    print(f"  effective n (left, right): ours ({ours_ll.n_left}, {ours_ll.n_right})  rdrobust "
          f"{tuple(int(v) for v in hc1.N_h)}")
    own = rdrobust(y, x, c=c, kernel="triangular", p=1)
    low, high = (float(v) for v in own.ci.iloc[2])
    print(f"  rdrobust with its own MSE-optimal bandwidth {float(own.bws.iloc[0, 0]):.3f}: "
          f"robust CI [{low:.3f}, {high:.3f}] (truth {data.ground_truth.tau_at_cutoff})")

    if importlib.util.find_spec("rdd") is not None:
        from rdd import rdd as rdd_package

        package_h = rdd_package.optimal_bandwidth(data.frame["spend"], data.frame["score"], cut=c)
        print(f"  IK bandwidth: ours {h:.3f}; PyPI rdd {package_h:.3f} (floors m3^2 at 0.01)")

    panel = generate_rollout_panel(
        RolloutConfig(n_units=12, n_periods=8, cohorts=((5, 0.5),), cohort_effect_scales=(1.0,),
                      base_effect=0.5)
    ).panel
    frame = panel.assign(unit=panel["unit_id"].astype(str), period_c=panel["period"].astype(str))
    model = smf.ols("outcome ~ treated + C(unit) + C(period_c)", data=frame)
    reference = wildboottest(model, param="treated", cluster=frame["unit_id"], B=9999, seed=1,
                             show=False)
    design = pd.get_dummies(frame[["unit", "period_c"]], drop_first=True, dtype=float)
    matrix = np.column_stack([np.ones(len(frame)), frame["treated"].to_numpy(float),
                              design.to_numpy()])
    ours = wild_cluster_bootstrap(matrix, frame["outcome"].to_numpy(), frame["unit_id"].to_numpy(),
                                  1, rng=np.random.default_rng(1), n_draws=9999,
                                  confidence_level=None)
    print("Wild cluster bootstrap, TWFE treated coefficient, 12 clusters (2^12 = 4096 patterns):")
    print(f"  ours p = {ours.p_value:.4f} (enumerated: {ours.enumerated})  "
          f"wildboottest p = {float(reference['p-value'].iloc[0]):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
