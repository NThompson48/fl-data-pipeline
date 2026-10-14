"""
fl_subgroup_pa_test.py -- re-runs the held-out Experiment-2 check split by
TRAINING PA (how much real Frontier League history a pitcher had going into
the prediction), comparing the two most recent BB% production changes
(DECISION_LOG.md "Location variance...replaces horiz_dist/vert_dist" and
"Recency-weighted multi-year BB% bridge input") against the model as it
stood before both -- specifically checking whether the aggregate held-out
win is spread evenly or concentrated among low-history players, where the
naive prior is weak and the model has to lean most heavily on the bridge/
surprise-variable regressors.

K% is unchanged by either of today's edits (same naive_K_rate + fl_k_rate_bridge
+ wh_sva_level spec throughout) -- reported for context, not as an OLD-vs-NEW
comparison.

Usage:
    python fl_subgroup_pa_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_college_bridge as bridge
import mlb_marcel as mm

K_SPEC = ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level"]
OLD_BB_SPEC = ["naive_BB_rate", "fl_bb_rate_bridge_old", "horiz_dist", "vert_dist"]
NEW_BB_SPEC = ["naive_BB_rate", "fl_bb_rate_bridge", "horiz_var", "vert_var"]

PA_BINS = [(0, 30, "limited (PA<30)"), (30, 100, "moderate (30-100)"), (100, 10_000, "substantial (100+)")]


def fit_old_bridge_bb(path: str = bridge.BRIDGE_PATH):
    """Reconstructs the PRE-today BB% bridge: flat TBF-only multi-year average
    (bb_decay=1.0), the design before the recency-decay change."""
    d = pd.read_parquet(path).drop_duplicates("pitcher_id").copy()
    college_years = bridge.load_all_college_years()
    rows = []
    for _, r in d.iterrows():
        feats = bridge._latest_and_multi(college_years, r["name_clean"],
                                          as_of_season=r["college_year"], bb_decay=1.0)
        if feats is None:
            continue
        rows.append({"pitcher_id": r["pitcher_id"], "bb_pct_multi": feats["bb_pct_multi"]})
    feat_df = pd.DataFrame(rows)
    d = d.merge(feat_df, on="pitcher_id", how="inner")
    bb_model = smf.ols("bb_rate ~ bb_pct_multi", data=d).fit()
    college_table = d[["pitcher_id", "bb_pct_multi"]].drop_duplicates("pitcher_id")
    return {"bb_model": bb_model, "college_table": college_table}


def attach_old_bridge(df: pd.DataFrame, old_model: dict, lg_bb_rate: float) -> pd.DataFrame:
    tbl = old_model["college_table"].copy()
    tbl["fl_bb_rate_bridge_old"] = old_model["bb_model"].predict(tbl).values
    out = df.merge(tbl[["pitcher_id", "fl_bb_rate_bridge_old"]], on="pitcher_id", how="left")
    out["fl_bb_rate_bridge_old"] = out["fl_bb_rate_bridge_old"].fillna(lg_bb_rate)
    return out


def score_by_pa(df_tr, df_te, target_col, spec, n0, label):
    d_tr = df_tr.dropna(subset=[target_col] + spec)
    w_tr = mm.reliability_weight(d_tr["PA_test"], n0).values
    m = smf.wls(f"{target_col} ~ " + " + ".join(spec), data=d_tr, weights=w_tr).fit()

    rows = []
    for lo, hi, bin_label in PA_BINS:
        sub = df_te[(df_te["PA"] >= lo) & (df_te["PA"] < hi)].dropna(subset=[target_col] + spec)
        if len(sub) < 10:
            rows.append({"model": label, "PA_bin": bin_label, "n": len(sub), "R2_w": np.nan, "RMSE_w_pp": np.nan})
            continue
        pred = m.predict(sub)
        w = mm.reliability_weight(sub["PA_test"], n0).values
        rows.append({"model": label, "PA_bin": bin_label, "n": len(sub),
                     "R2_w": round(mm.weighted_r2(sub[target_col].values, pred.values, w), 4),
                     "RMSE_w_pp": round(mm.weighted_rmse(sub[target_col].values, pred.values, w) * 100, 3)})
    # overall (all bins combined), for reference
    d_te = df_te.dropna(subset=[target_col] + spec)
    w_te = mm.reliability_weight(d_te["PA_test"], n0).values
    pred_all = m.predict(d_te)
    rows.append({"model": label, "PA_bin": "ALL", "n": len(d_te),
                 "R2_w": round(mm.weighted_r2(d_te[target_col].values, pred_all.values, w_te), 4),
                 "RMSE_w_pp": round(mm.weighted_rmse(d_te[target_col].values, pred_all.values, w_te) * 100, 3)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print("\nAssembling Experiment-2 train/test panels (current production bridge/features)...")
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)],
                                      [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    print("Fitting the OLD (pre-today) flat-TBF BB% bridge for comparison...")
    old_bridge = fit_old_bridge_bb()
    lg_bb_rate = df_tr["fl_bb_rate_bridge"].mean()  # same fallback convention as assemble_features
    df_tr = attach_old_bridge(df_tr, old_bridge, lg_bb_rate)
    df_te = attach_old_bridge(df_te, old_bridge, lg_bb_rate)

    print(f"\nTest-set PA distribution: {df_te['PA'].describe()[['min','25%','50%','75%','max']].to_dict()}")
    print(f"Test-set PA bin sizes: " +
          ", ".join(f"{lbl}: {((df_te['PA']>=lo)&(df_te['PA']<hi)).sum()}" for lo, hi, lbl in PA_BINS))

    print("\n" + "=" * 80)
    print("BB%: OLD (horiz_dist/vert_dist + flat-TBF bridge) vs NEW (horiz_var/vert_var + recency-decay bridge)")
    print("=" * 80)
    old_res = score_by_pa(df_tr, df_te, "BB_rate_actual", OLD_BB_SPEC, n0_bb, "OLD (pre-today)")
    new_res = score_by_pa(df_tr, df_te, "BB_rate_actual", NEW_BB_SPEC, n0_bb, "NEW (current production)")
    print(pd.concat([old_res, new_res], ignore_index=True).sort_values(["PA_bin", "model"]).to_string(index=False))

    print("\n" + "=" * 80)
    print("K% (unchanged spec, shown for context only)")
    print("=" * 80)
    k_res = score_by_pa(df_tr, df_te, "K_rate_actual", K_SPEC, n0_k, "K% (unchanged)")
    print(k_res.to_string(index=False))
