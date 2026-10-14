"""
fl_limited_pa_diagnosis.py -- follow-up to fl_subgroup_pa_test.py's finding
that the limited-PA (<30) test subgroup (n=26) favored the OLD BB% model
(horiz_dist/vert_dist + flat-TBF bridge) over the NEW one (horiz_var/vert_var
+ recency-decay bridge), opposite the aggregate result. Two questions:
  1. Which of today's two changes (location-variance swap, bridge recency-
     decay) is actually driving the reversal -- isolate each one.
  2. Is n=26 robust, or is one/two influential observations driving the
     whole comparison? Leave-one-out R2_w sensitivity + per-row contribution
     to the weighted sum of squares.

Usage:
    python fl_limited_pa_diagnosis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_college_bridge as bridge
import mlb_marcel as mm
from fl_subgroup_pa_test import fit_old_bridge_bb, attach_old_bridge, PA_BINS

VARIANTS = {
    "V1 OLD (flat bridge + dist)":      ["naive_BB_rate", "fl_bb_rate_bridge_old", "horiz_dist", "vert_dist"],
    "V2 loc-var swap only":             ["naive_BB_rate", "fl_bb_rate_bridge_old", "horiz_var",  "vert_var"],
    "V3 bridge-recency only":           ["naive_BB_rate", "fl_bb_rate_bridge",     "horiz_dist", "vert_dist"],
    "V4 NEW (both changes, current)":   ["naive_BB_rate", "fl_bb_rate_bridge",     "horiz_var",  "vert_var"],
}


def fit_and_score(df_tr, df_te, spec, n0, label, pa_lo=0, pa_hi=30):
    d_tr = df_tr.dropna(subset=["BB_rate_actual"] + spec)
    w_tr = mm.reliability_weight(d_tr["PA_test"], n0).values
    m = smf.wls("BB_rate_actual ~ " + " + ".join(spec), data=d_tr, weights=w_tr).fit()

    sub = df_te[(df_te["PA"] >= pa_lo) & (df_te["PA"] < pa_hi)].dropna(subset=["BB_rate_actual"] + spec).copy()
    sub["pred"] = m.predict(sub)
    sub["w"] = mm.reliability_weight(sub["PA_test"], n0).values
    sub["resid"] = sub["BB_rate_actual"] - sub["pred"]
    sub["ss_contrib"] = sub["w"] * sub["resid"] ** 2
    r2 = mm.weighted_r2(sub["BB_rate_actual"].values, sub["pred"].values, sub["w"].values)
    return m, sub, r2


def loo_r2(sub: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out: recompute R2_w on the subgroup excluding each row, see
    how much a single pitcher's removal moves it."""
    rows = []
    full_r2 = mm.weighted_r2(sub["BB_rate_actual"].values, sub["pred"].values, sub["w"].values)
    for idx in sub.index:
        rest = sub.drop(idx)
        r2_wo = mm.weighted_r2(rest["BB_rate_actual"].values, rest["pred"].values, rest["w"].values)
        rows.append({"pitcher_id": sub.loc[idx, "pitcher_id"], "pitcher": sub.loc[idx, "pitcher"],
                      "PA": sub.loc[idx, "PA"], "w": round(sub.loc[idx, "w"], 3),
                      "resid_pp": round(sub.loc[idx, "resid"] * 100, 2),
                      "ss_contrib": round(sub.loc[idx, "ss_contrib"], 5),
                      "R2_w_full": round(full_r2, 4), "R2_w_without": round(r2_wo, 4),
                      "delta": round(r2_wo - full_r2, 4)})
    return pd.DataFrame(rows).sort_values("ss_contrib", ascending=False)


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)],
                                      [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    old_bridge = fit_old_bridge_bb()
    lg_bb_rate = df_tr["fl_bb_rate_bridge"].mean()
    df_tr = attach_old_bridge(df_tr, old_bridge, lg_bb_rate)
    df_te = attach_old_bridge(df_te, old_bridge, lg_bb_rate)

    names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    df_te = df_te.merge(names, on="pitcher_id", how="left")

    print("\n" + "=" * 80)
    print("PART 1: isolating which change drives the limited-PA (<30) reversal")
    print("=" * 80)
    results = {}
    for label, spec in VARIANTS.items():
        m, sub, r2 = fit_and_score(df_tr, df_te, spec, n0_bb, label)
        results[label] = (m, sub, r2)
        print(f"{label:35s} n={len(sub):3d}  R2_w={r2:.4f}")

    print("\n" + "=" * 80)
    print("PART 2: leave-one-out sensitivity on the limited-PA subgroup, V4 (NEW/current production)")
    print("=" * 80)
    _, sub_new, _ = results["V4 NEW (both changes, current)"]
    loo_new = loo_r2(sub_new)
    print(loo_new.to_string(index=False))

    print("\n" + "=" * 80)
    print("PART 2b: same leave-one-out, V1 (OLD) for comparison")
    print("=" * 80)
    _, sub_old, _ = results["V1 OLD (flat bridge + dist)"]
    loo_old = loo_r2(sub_old)
    print(loo_old.to_string(index=False))
