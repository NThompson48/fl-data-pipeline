"""
fl_backtest_error_analysis.py -- residual breakdown for the half-season backtest
(fl_halfseason_backtest.py), focused on Experiment 2 (the largest genuinely
held-out test, n=262: fit on full-2025 -> actual 1H2026, frozen, scored on
2025+1H2026 -> actual 2H2026). In-sample experiments (3/4) are deliberately
excluded from this analysis -- their residuals would just restate the overfit
already confirmed in DECISION_LOG.md, not new information.

Usage:
    python fl_backtest_error_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def bucket_report(df: pd.DataFrame, bucket_col: str, resid_col: str, actual_col: str,
                   pred_col: str, w_col: str, q: int = 4, label: str = "") -> pd.DataFrame:
    d = df.copy()
    try:
        d["_bucket"] = pd.qcut(d[bucket_col], q=q, duplicates="drop")
    except ValueError:
        d["_bucket"] = pd.cut(d[bucket_col], bins=q)
    out = d.groupby("_bucket").apply(lambda g: pd.Series({
        "n": len(g),
        "mean_actual": np.average(g[actual_col], weights=g[w_col]),
        "mean_pred": np.average(g[pred_col], weights=g[w_col]),
        "mean_resid": np.average(g[resid_col], weights=g[w_col]),
        "mean_abs_resid": np.average(g[resid_col].abs(), weights=g[w_col]),
    }), include_groups=False)
    out.insert(0, "metric", label)
    return out.reset_index()


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()

    print("Rebuilding Experiment 2 train/test panels...")
    df_tr2, n0_k2, n0_bb2 = bt.assemble(raw, [(bt.START_2025, bt.END_2025)],
                                        [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=True)
    df_te2, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                                [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=True)

    spec_naive = bt.MODEL_SPECS["naive"]
    spec_full = bt.MODEL_SPECS["naive+other_league+changepoint"]

    results = {}
    for metric, spec, target in [("K%", spec_naive, "K_rate_actual"), ("BB%", spec_naive, "BB_rate_actual")]:
        regr = spec["K"] if metric == "K%" else spec["BB"]
        d_tr = df_tr2.dropna(subset=[target] + regr)
        n0 = n0_k2 if metric == "K%" else n0_bb2
        w_tr = mm.reliability_weight(d_tr["PA_test"], n0).values
        m = bt.fit_weighted(d_tr, target, regr, w_tr)
        d_te = df_te2.dropna(subset=[target] + regr).copy()
        d_te["pred_naive"] = m.predict(d_te)
        results[metric] = (d_te, n0, target)

    # full model (other_league + changepoint) for comparison
    full_preds = {}
    for metric, spec, target in [("K%", spec_full, "K_rate_actual"), ("BB%", spec_full, "BB_rate_actual")]:
        regr = spec["K"] if metric == "K%" else spec["BB"]
        d_tr = df_tr2.dropna(subset=[target] + regr)
        n0 = n0_k2 if metric == "K%" else n0_bb2
        w_tr = mm.reliability_weight(d_tr["PA_test"], n0).values
        m = bt.fit_weighted(d_tr, target, regr, w_tr)
        d_te = df_te2.dropna(subset=[target] + regr).copy()
        d_te["pred_full"] = m.predict(d_te)
        full_preds[metric] = d_te[["pitcher_id", "pred_full"]]

    print("\n" + "=" * 70)
    print("1) CALIBRATION BY TRAINING NAIVE-RATE TERCILE (regression-to-mean check)")
    print("=" * 70)
    for metric, (d_te, n0, target) in results.items():
        d_te["resid"] = d_te[target] - d_te["pred_naive"]
        d_te["w"] = mm.reliability_weight(d_te["PA_test"], n0).values
        naive_col = "naive_K_rate" if metric == "K%" else "naive_BB_rate"
        rep = bucket_report(d_te, naive_col, "resid", target, "pred_naive", "w", q=3, label=metric)
        print(f"\n--- {metric}, bucketed by training naive rate ({naive_col}) ---")
        print(rep.to_string(index=False))

    print("\n" + "=" * 70)
    print("2) CALIBRATION BY TRAINING SAMPLE SIZE (reliability check)")
    print("=" * 70)
    for metric, (d_te, n0, target) in results.items():
        rep = bucket_report(d_te, "PA", "resid", target, "pred_naive", "w", q=4, label=metric)
        print(f"\n--- {metric}, bucketed by training-window PA ---")
        print(rep.to_string(index=False))

    print("\n" + "=" * 70)
    print("3) RESIDUAL BY COLLEGE-BRIDGE MATCH STATUS")
    print("=" * 70)
    for metric, (d_te, n0, target) in results.items():
        g = d_te.groupby("has_college_match").apply(lambda x: pd.Series({
            "n": len(x),
            "mean_resid": np.average(x["resid"], weights=x["w"]),
            "mean_abs_resid": np.average(x["resid"].abs(), weights=x["w"]),
        }), include_groups=False)
        print(f"\n--- {metric} ---")
        print(g.to_string())

    print("\n" + "=" * 70)
    print("4) DOES THE RAW CHANGEPOINT DIRECTION TRACK THE RESIDUAL SIGN?")
    print("   (checks for a nonlinear/threshold signal the linear WLS term missed)")
    print("=" * 70)
    cp_col = {"K%": "wh_sva_cp", "BB%": None}
    for metric, (d_te, n0, target) in results.items():
        if metric == "K%":
            d = d_te.dropna(subset=["wh_sva_cp"])
            nonzero = d[d["wh_sva_cp"] != 0]
            print(f"\n--- K%: n with nonzero wh_sva_cp = {len(nonzero)} / {len(d)} ---")
            if len(nonzero) > 5:
                corr = nonzero["wh_sva_cp"].corr(nonzero["resid"])
                print(f"corr(wh_sva_cp, K%_residual) among nonzero-changepoint pitchers: {corr:.3f}")
                same_sign = (np.sign(nonzero["wh_sva_cp"]) == np.sign(nonzero["resid"])).mean()
                print(f"share with SAME sign (cp predicts direction of surprise): {same_sign:.3f}")
        else:
            d = d_te.dropna(subset=["horiz_dist_cp", "vert_dist_cp"])
            for col in ["horiz_dist_cp", "vert_dist_cp"]:
                nonzero = d[d[col] != 0]
                if len(nonzero) > 5:
                    corr = nonzero[col].corr(nonzero["resid"])
                    same_sign = (np.sign(nonzero[col]) == np.sign(nonzero["resid"])).mean()
                    print(f"BB%: n with nonzero {col} = {len(nonzero)} / {len(d)}, "
                          f"corr={corr:.3f}, same_sign_share={same_sign:.3f}")

    print("\n" + "=" * 70)
    print("5) NAIVE-MODEL vs FULL-MODEL (other_league+changepoint): per-pitcher residual comparison")
    print("=" * 70)
    for metric, (d_te, n0, target) in results.items():
        merged = d_te.merge(full_preds[metric], on="pitcher_id", how="inner")
        merged["resid_full"] = merged[target] - merged["pred_full"]
        merged["improved"] = merged["resid_full"].abs() < merged["resid"].abs()
        print(f"\n--- {metric}: share of pitchers where the full model's abs error improved "
              f"vs naive: {merged['improved'].mean():.3f} (n={len(merged)}) ---")

    print("\n" + "=" * 70)
    print("6) WORST 10 RESIDUALS (naive model) -- any visible pattern?")
    print("=" * 70)
    for metric, (d_te, n0, target) in results.items():
        cols = ["pitcher_id", "PA", "PA_test", target, "pred_naive", "resid", "has_college_match"]
        if metric == "K%":
            cols.append("wh_sva_cp")
        else:
            cols += ["horiz_dist_cp", "vert_dist_cp"]
        worst = d_te.reindex(d_te["resid"].abs().sort_values(ascending=False).index)[cols].head(10)
        print(f"\n--- {metric} worst 10 ---")
        print(worst.to_string(index=False))
