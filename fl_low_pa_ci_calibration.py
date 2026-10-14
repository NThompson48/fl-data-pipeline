"""
fl_low_pa_ci_calibration.py -- the single Experiment-2 held-out fold doesn't
have enough low-training-PA cases to characterize CI calibration there (only
n=2 in the 1-5 PA bucket). This sweeps several EARLY-season cutoffs (1-12
weeks into a season, both 2025 and 2026) the same way fl_ros_early_test.py
does, pooling many more real (thin-sample snapshot -> actual rest-of-season
outcome) pairs, to directly check: does the FROZEN production model's logit
CI actually cover the real rest-of-season outcome at the stated 90% rate,
broken down finely by how much training PA the snapshot had?

Uses the SAME frozen model production deploys (fit once on full-2025 ->
actual 1H2026, applied forward) -- this characterizes what's actually
shipped, not a refit-per-cutoff model.

Usage:
    python fl_low_pa_ci_calibration.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_production_projection as fprod
import mlb_marcel as mm

CUTOFF_WEEKS = [1, 2, 3, 5, 8, 12]
Z_90 = 1.6449
SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]


def logit_ci(p, se, z):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    se_logit = se / (p * (1 - p))
    lp = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-(lp - z * se_logit))), 1 / (1 + np.exp(-(lp + z * se_logit)))


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
    stuff = bt.stuff_levels(surprise_pool)
    df = df.merge(wh_level, on="pitcher_id", how="left").merge(loc_var, on="pitcher_id", how="left")
    df = df.merge(stuff, on="pitcher_id", how="left")
    for c in ["wh_sva_level", "horiz_var", "vert_var", "spin_z_norm", "extension"]:
        df[c] = df[c].fillna(df[c].mean())

    actual = bt.actual_rate(raw, [(label_start, label_end)])
    df = df.merge(actual, on="pitcher_id", how="inner")
    return df


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    # the SAME frozen model production deploys
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    d_tr_k = df_tr.dropna(subset=["K_rate_actual"] + SPEC["K"])
    w_tr_k = mm.reliability_weight(d_tr_k["PA_test"], n0_k).values
    m_k = bt.fit_weighted(d_tr_k, "K_rate_actual", SPEC["K"], w_tr_k)

    full_pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025),
                           fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True)
    term = full_pool[full_pool["is_terminal"]]
    lg_k = (term["k_or_bb"] == "strikeout").mean()
    lg_bb = (term["k_or_bb"] == "walk").mean()

    SEASONS = {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}

    print("Building early-cutoff folds (pooling many thin-sample -> actual-rest-of-season pairs)...")
    frames = []
    for year, (season_start, season_end) in SEASONS.items():
        prior_windows = [(bt.START_2025, bt.END_2025)] if year == 2026 else []
        for wk in CUTOFF_WEEKS:
            cutoff = (pd.Timestamp(season_start) + pd.Timedelta(weeks=wk)).strftime("%Y-%m-%d")
            day_after = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.Timestamp(day_after) >= pd.Timestamp(season_end) - pd.Timedelta(weeks=2):
                continue
            surprise_windows = prior_windows + [(season_start, cutoff)]
            fold = build_fold(raw, season_start, cutoff, surprise_windows, day_after, season_end,
                               n0_k, n0_bb, lg_k, lg_bb)
            fold["cutoff_week"] = wk
            fold["year"] = year
            frames.append(fold)
            print(f"  {year} wk{wk}: n={len(fold)}")

    pooled = pd.concat(frames, ignore_index=True)
    d = pooled.dropna(subset=["K_rate_actual"] + SPEC["K"]).copy()
    w = mm.reliability_weight(d["PA"], n0_k).values
    d["pred_K"] = m_k.predict(d)
    d["se_K"] = m_k.get_prediction(d, weights=w).se_obs * fprod.K_SE_CALIBRATION
    lo, hi = logit_ci(d["pred_K"].values, d["se_K"].values, Z_90)
    d["ci_lo"], d["ci_hi"] = lo, hi
    d["covered"] = (d["K_rate_actual"] >= d["ci_lo"]) & (d["K_rate_actual"] <= d["ci_hi"])
    d["ci_width"] = d["ci_hi"] - d["ci_lo"]

    print(f"\nn={len(d)} pooled (snapshot PA, actual rest-of-season outcome) pairs across "
          f"{len(CUTOFF_WEEKS)} cutoffs x 2 seasons")
    print(f"\noverall K% CI coverage: {d['covered'].mean():.1%} (nominal 90%)")

    d["pa_bucket"] = pd.cut(d["PA"], [0, 3, 6, 10, 15, 25, 40, 60, 100, 99999],
                            labels=["1-3", "4-6", "7-10", "11-15", "16-25", "26-40", "41-60", "61-100", "100+"])
    print("\ncoverage + CI width by training-PA bucket:")
    print(d.groupby("pa_bucket", observed=True).agg(
        n=("covered", "size"), coverage=("covered", "mean"),
        mean_width=("ci_width", "mean"), mean_se=("se_K", "mean"),
        mean_pred=("pred_K", "mean"), mean_actual=("K_rate_actual", "mean"),
    ).round(3))
