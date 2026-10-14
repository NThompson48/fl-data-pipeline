"""
fl_ros_bridge_prior_test.py -- tests whether the other-league bridge should be
an informative PRIOR (Marcel-style pseudo-PA shrinkage, same mechanism as
naive_K_rate/naive_BB_rate's own n0_k/n0_bb) rather than a free linear OLS
regressor. Motivated by fl_ros_coef_drift_test.py: the bridge's OLS coefficient
was the LEAST stable regressor across in-season cutoffs (CV 0.37-0.49 vs
0.20-0.31 for everything else) and decayed from high early-season weight to
near-zero late-season weight -- exactly the pattern a pseudo-PA blend would
produce automatically, without needing a fresh refit at every cutoff.

Three models compared, all using ONLY naive prior + bridge (no surprise vars,
to isolate this specific question):
  A) naive-only:     naive_K_rate                        (n0_bridge=0 case of C)
  B) current (OLS):  naive_K_rate + fl_k_rate_bridge, fit ONCE on pooled training
                      data and FROZEN (mirrors real production: a single frozen
                      coefficient, not a per-cutoff refit)
  C) informative prior: blended = (K + n0_k*lg_k + n0_bridge*bridge_rate)
                                  / (PA + n0_k + n0_bridge)
                      n0_bridge fit ONCE by nonlinear least squares against
                      actual outcomes, FROZEN, reusing the project's existing
                      fixed n0_k/n0_bb (not re-estimated here).

Validation: genuinely held out by YEAR -- fit B's coefficient and C's n0_bridge
on all 2025 cutoffs, freeze, score on all 2026 cutoffs; then the reverse. This
is the same train-freeze-roll-forward discipline used everywhere else in this
project, applied to the hyperparameter/coefficient itself, not just the model.

Usage:
    python fl_ros_bridge_prior_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

CUTOFF_WEEKS = [2, 5, 8, 11]
SEASONS = {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}


def raw_window(raw, start, end):
    kbb = fp.build_period_kbb(fp.filter_period(raw, start, end), min_pa=1)
    return kbb[["pitcher_id", "PA", "K", "BB", "K_rate", "BB_rate"]]


def build_pool(raw, lg_k, lg_bb):
    rows = []
    for year, (season_start, season_end) in SEASONS.items():
        for wk in CUTOFF_WEEKS:
            cutoff = (pd.Timestamp(season_start) + pd.Timedelta(weeks=wk)).strftime("%Y-%m-%d")
            day_after = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.Timestamp(day_after) >= pd.Timestamp(season_end) - pd.Timedelta(weeks=2):
                continue
            win = raw_window(raw, season_start, cutoff)
            br = bt.bridge_features(win["pitcher_id"], raw, current_year=int(cutoff[:4]))
            df = win.merge(br, on="pitcher_id", how="left")
            df["fl_k_rate_bridge"] = df["fl_k_rate_bridge"].fillna(lg_k)
            df["fl_bb_rate_bridge"] = df["fl_bb_rate_bridge"].fillna(lg_bb)
            actual = bt.actual_rate(raw, [(day_after, season_end)])
            df = df.merge(actual, on="pitcher_id", how="inner")
            df["year"], df["week"], df["cutoff_PA_median"] = year, wk, df["PA"].median()
            rows.append(df)
            print(f"  built {year} wk{wk} (cutoff={cutoff}): n={len(df)}")
    return pd.concat(rows, ignore_index=True)


def blended_rate(K, PA, n0_fixed, lg, n0_bridge, bridge_rate):
    return (K + n0_fixed * lg + n0_bridge * bridge_rate) / (PA + n0_fixed + n0_bridge)


def fit_n0_bridge(train, target_col, K_col, n0_fixed, lg, bridge_col, w):
    def f(_, n0_bridge):
        return blended_rate(train[K_col].values, train["PA"].values, n0_fixed, lg,
                             n0_bridge, train[bridge_col].values)
    popt, _ = curve_fit(f, np.zeros(len(train)), train[target_col].values,
                         p0=[50.0], sigma=1.0 / np.sqrt(w + 1e-9), bounds=(0, 2000))
    return float(popt[0])


def score(y_true, y_pred, w, label):
    return {"label": label, "n": len(y_true),
            "R2_w": round(mm.weighted_r2(y_true, y_pred, w), 4),
            "RMSE_w_pp": round(mm.weighted_rmse(y_true, y_pred, w) * 100, 3)}


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    _, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)],
                                 [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    full_pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025),
                           fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True)
    term = full_pool[full_pool["is_terminal"]]
    lg_k = (term["k_or_bb"] == "strikeout").mean()
    lg_bb = (term["k_or_bb"] == "walk").mean()
    print(f"n0_k={n0_k:.1f}  n0_bb={n0_bb:.1f}  lg_k={lg_k:.4f}  lg_bb={lg_bb:.4f}")

    print("\nBuilding pooled cutoff dataset (2025 + 2026, weeks 2/5/8/11)...")
    pool = build_pool(raw, lg_k, lg_bb)
    print(f"pooled n={len(pool)}")

    SPECS = {
        "K": dict(target="K_rate_actual", Kcol="K", n0_fixed=n0_k, lg=lg_k,
                  bridge_col="fl_k_rate_bridge", naive_col_formula="naive_K_rate"),
        "BB": dict(target="BB_rate_actual", Kcol="BB", n0_fixed=n0_bb, lg=lg_bb,
                   bridge_col="fl_bb_rate_bridge", naive_col_formula="naive_BB_rate"),
    }

    for metric, spec in SPECS.items():
        print("\n" + "=" * 100)
        print(f"{metric}% -- held out by YEAR")
        print("=" * 100)
        pool["_naive_rate"] = blended_rate(pool[spec["Kcol"]], pool["PA"], spec["n0_fixed"], spec["lg"],
                                            0.0, pool[spec["bridge_col"]])
        pool = pool.rename(columns={"_naive_rate": spec["naive_col_formula"]})

        for train_year, test_year in [(2025, 2026), (2026, 2025)]:
            train = pool[pool["year"] == train_year].dropna(
                subset=[spec["target"], spec["naive_col_formula"], spec["bridge_col"]])
            test = pool[pool["year"] == test_year].dropna(
                subset=[spec["target"], spec["naive_col_formula"], spec["bridge_col"]])
            w_train = mm.reliability_weight(train["PA_test"], spec["n0_fixed"]).values
            w_test = mm.reliability_weight(test["PA_test"], spec["n0_fixed"]).values

            # A) naive-only
            pred_naive_test = test[spec["naive_col_formula"]].values

            # B) current OLS approach: fit ONCE on train, freeze, apply to test
            ols = smf.wls(f"{spec['target']} ~ {spec['naive_col_formula']} + {spec['bridge_col']}",
                          data=train, weights=w_train).fit()
            pred_ols_test = ols.predict(test).values

            # C) informative-prior blend: fit n0_bridge ONCE on train, freeze, apply to test
            n0_bridge = fit_n0_bridge(train, spec["target"], spec["Kcol"], spec["n0_fixed"],
                                       spec["lg"], spec["bridge_col"], w_train)
            pred_blend_test = blended_rate(test[spec["Kcol"]].values, test["PA"].values,
                                            spec["n0_fixed"], spec["lg"], n0_bridge,
                                            test[spec["bridge_col"]].values)

            print(f"\n--- train={train_year} (frozen) -> test={test_year} (n_train={len(train)}, n_test={len(test)}) ---")
            print(f"    fitted: OLS coef on {spec['naive_col_formula']}={ols.params[spec['naive_col_formula']]:.3f}, "
                  f"on {spec['bridge_col']}={ols.params[spec['bridge_col']]:.3f}  |  n0_bridge={n0_bridge:.1f}")
            rows = [score(test[spec["target"]].values, pred_naive_test, w_test, "A) naive-only"),
                    score(test[spec["target"]].values, pred_ols_test, w_test, "B) current OLS (frozen)"),
                    score(test[spec["target"]].values, pred_blend_test, w_test, "C) informative prior (frozen n0_bridge)")]
            print(pd.DataFrame(rows).to_string(index=False))

        # does the single frozen n0_bridge reproduce the OLS coefficient's observed
        # decay pattern across cutoffs, without being refit at each one?
        print(f"\n--- implied bridge weight n0_bridge/(n0_bridge+PA) at each cutoff's median PA (n0_bridge fit on ALL data) ---")
        all_train = pool.dropna(subset=[spec["target"], spec["naive_col_formula"], spec["bridge_col"]])
        w_all = mm.reliability_weight(all_train["PA_test"], spec["n0_fixed"]).values
        n0_bridge_all = fit_n0_bridge(all_train, spec["target"], spec["Kcol"], spec["n0_fixed"],
                                       spec["lg"], spec["bridge_col"], w_all)
        print(f"    n0_bridge (fit on all data) = {n0_bridge_all:.1f}")
        med_pa = pool.groupby(["year", "week"])["PA"].median().reset_index()
        med_pa["implied_bridge_weight"] = n0_bridge_all / (n0_bridge_all + med_pa["PA"])
        print(med_pa.to_string(index=False))
