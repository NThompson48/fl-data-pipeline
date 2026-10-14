"""
fl_ros_early_test.py -- tests the theoretical use case the surprise variables
(wh_sva_level, horiz_var, vert_var) were actually built for: a pitcher with
only a THIN in-season sample so far (early in the season, rest-of-season
projection needed), where the naive prior alone is unreliable and a prior-
knowledge regressor should matter most. Every other held-out check in this
project has used FULL-season(s) -> next-full/half-season transitions; this one
instead trains on just the first ~3 weeks of a season and scores against the
REST of that same season -- a genuinely different, more realistic "early ROS"
scenario, split by how much PRIOR-season history a pitcher has (true newcomer
vs returning veteran facing a thin current sample).

Two folds:
  - early-2025 -> rest-2025: no prior Trackman year exists (FL data starts
    2025), so this tests the raw "does the full model beat naive-only when
    the CURRENT sample is thin" question with the surprise vars themselves
    also only drawing on that same thin window.
  - early-2026 -> rest-2026: prior full-season 2025 data exists, closer to a
    real production scenario -- naive prior uses ONLY the thin early-2026
    slice (consistent with the already-validated "most-recent-period-only"
    finding), surprise vars pool [full-2025 + early-2026] (richer, stabler).
    Split by whether the pitcher has ANY 2025 PA (returning) or not (new
    arrival) -- the two populations the "early ROS" use case actually spans.

Usage:
    python fl_ros_early_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

EARLY_DAYS = 21
EARLY_CUTOFF_2025 = (pd.Timestamp(bt.START_2025) + pd.Timedelta(days=EARLY_DAYS)).strftime("%Y-%m-%d")
EARLY_CUTOFF_2026 = (pd.Timestamp(bt.START_2026) + pd.Timedelta(days=EARLY_DAYS)).strftime("%Y-%m-%d")
_DAY_AFTER_EARLY_2025 = (pd.Timestamp(EARLY_CUTOFF_2025) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
_DAY_AFTER_EARLY_2026 = (pd.Timestamp(EARLY_CUTOFF_2026) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

K_FULL_SPEC = ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level"]
BB_FULL_SPEC = ["naive_BB_rate", "fl_bb_rate_bridge", "horiz_var", "vert_var"]
K_NAIVE_SPEC = ["naive_K_rate"]
BB_NAIVE_SPEC = ["naive_BB_rate"]


def naive_from_window(raw, start, end, n0_k, n0_bb, lg_k, lg_bb):
    kbb = fp.build_period_kbb(fp.filter_period(raw, start, end), min_pa=1)
    kbb["naive_K_rate"] = (kbb["K"] + n0_k * lg_k) / (kbb["PA"] + n0_k)
    kbb["naive_BB_rate"] = (kbb["BB"] + n0_bb * lg_bb) / (kbb["PA"] + n0_bb)
    return kbb[["pitcher_id", "PA", "naive_K_rate", "naive_BB_rate"]]


def build_fold(raw, early_start, early_end, surprise_windows, label_start, label_end,
               n0_k, n0_bb, lg_k, lg_bb):
    naive = naive_from_window(raw, early_start, early_end, n0_k, n0_bb, lg_k, lg_bb)
    br = bt.bridge_features(naive["pitcher_id"], raw, current_year=int(early_end[:4]))
    df = naive.merge(br, on="pitcher_id", how="left")
    df["fl_k_rate_bridge"] = df["fl_k_rate_bridge"].fillna(lg_k)
    df["fl_bb_rate_bridge"] = df["fl_bb_rate_bridge"].fillna(lg_bb)

    surprise_pool = pd.concat([fp.filter_period(raw, s, e) for s, e in surprise_windows], ignore_index=True)
    wh_level, _ = bt.wh_sva_level_and_changepoint(surprise_pool)
    loc_var = bt.loc_var_levels(surprise_pool)
    df = df.merge(wh_level, on="pitcher_id", how="left").merge(loc_var, on="pitcher_id", how="left")
    for c in ["wh_sva_level", "horiz_var", "vert_var"]:
        df[c] = df[c].fillna(df[c].mean())

    actual = bt.actual_rate(raw, [(label_start, label_end)])
    df = df.merge(actual, on="pitcher_id", how="inner")
    return df


def score(df, target_col, spec, n0, label):
    d = df.dropna(subset=[target_col] + spec)
    if len(d) < 10:
        return {"label": label, "n": len(d), "R2_w": np.nan, "RMSE_w_pp": np.nan}
    w = mm.reliability_weight(d["PA_test"], n0).values
    m = smf.wls(f"{target_col} ~ " + " + ".join(spec), data=d, weights=w).fit()
    pred = m.fittedvalues
    return {"label": label, "n": len(d),
            "R2_w": round(mm.weighted_r2(d[target_col].values, pred.values, w), 4),
            "RMSE_w_pp": round(mm.weighted_rmse(d[target_col].values, pred.values, w) * 100, 3)}


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print(f"Early windows: 2025 [{bt.START_2025} - {EARLY_CUTOFF_2025}] ({EARLY_DAYS}d), "
          f"2026 [{bt.START_2026} - {EARLY_CUTOFF_2026}] ({EARLY_DAYS}d)")

    # reuse the SAME global n0/lg_rate the rest of this project uses -- fixed
    # hyperparameters, not re-derived from the thin early window
    df_ref, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)],
                                       [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    full_pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025),
                           fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True)
    term = full_pool[full_pool["is_terminal"]]
    lg_k = (term["k_or_bb"] == "strikeout").mean()
    lg_bb = (term["k_or_bb"] == "walk").mean()
    print(f"n0_k={n0_k:.1f}  n0_bb={n0_bb:.1f}  lg_k={lg_k:.4f}  lg_bb={lg_bb:.4f}")

    print("\nBuilding fold: early-2025 -> rest-of-2025...")
    fold25 = build_fold(raw, bt.START_2025, EARLY_CUTOFF_2025, [(bt.START_2025, EARLY_CUTOFF_2025)],
                         _DAY_AFTER_EARLY_2025, bt.END_2025, n0_k, n0_bb, lg_k, lg_bb)
    print(f"  early-2025 window PA distribution: "
          f"{fold25['PA'].describe()[['min', '25%', '50%', '75%', 'max']].to_dict()}")

    print("Building fold: early-2026 -> rest-of-2026...")
    fold26 = build_fold(raw, bt.START_2026, EARLY_CUTOFF_2026,
                         [(bt.START_2025, bt.END_2025), (bt.START_2026, EARLY_CUTOFF_2026)],
                         _DAY_AFTER_EARLY_2026, bt.END_2026, n0_k, n0_bb, lg_k, lg_bb)
    prior_pa = fp.build_period_kbb(fp.filter_period(raw, bt.START_2025, bt.END_2025), min_pa=1)[["pitcher_id", "PA"]]
    prior_pa = prior_pa.rename(columns={"PA": "prior_season_PA"})
    fold26 = fold26.merge(prior_pa, on="pitcher_id", how="left")
    fold26["prior_season_PA"] = fold26["prior_season_PA"].fillna(0)
    print(f"  early-2026 window PA distribution: "
          f"{fold26['PA'].describe()[['min', '25%', '50%', '75%', 'max']].to_dict()}")
    print(f"  returning (prior 2025 PA>0): {(fold26['prior_season_PA'] > 0).sum()}, "
          f"new arrivals (no 2025 PA): {(fold26['prior_season_PA'] == 0).sum()}")

    print("\n" + "=" * 80)
    print("EARLY-2025 -> REST-OF-2025 (no prior Trackman year available)")
    print("=" * 80)
    rows = []
    rows.append(score(fold25, "K_rate_actual", K_NAIVE_SPEC, n0_k, "K% naive-only"))
    rows.append(score(fold25, "K_rate_actual", K_FULL_SPEC, n0_k, "K% full model"))
    rows.append(score(fold25, "BB_rate_actual", BB_NAIVE_SPEC, n0_bb, "BB% naive-only"))
    rows.append(score(fold25, "BB_rate_actual", BB_FULL_SPEC, n0_bb, "BB% full model"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 80)
    print("EARLY-2026 -> REST-OF-2026, split by prior-season (2025) history")
    print("=" * 80)
    for pop_label, sub in [("ALL", fold26), ("returning (2025 PA>0)", fold26[fold26["prior_season_PA"] > 0]),
                           ("new arrivals (no 2025 PA)", fold26[fold26["prior_season_PA"] == 0])]:
        print(f"\n--- {pop_label} ---")
        rows = []
        rows.append(score(sub, "K_rate_actual", K_NAIVE_SPEC, n0_k, "K% naive-only"))
        rows.append(score(sub, "K_rate_actual", K_FULL_SPEC, n0_k, "K% full model"))
        rows.append(score(sub, "BB_rate_actual", BB_NAIVE_SPEC, n0_bb, "BB% naive-only"))
        rows.append(score(sub, "BB_rate_actual", BB_FULL_SPEC, n0_bb, "BB% full model"))
        print(pd.DataFrame(rows).to_string(index=False))
