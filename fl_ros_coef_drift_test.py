"""
fl_ros_coef_drift_test.py -- the open question flagged when scoping a real
ROS production tool: does the full model's fitted relationship (naive prior
vs bridge vs surprise variables) stay stable across the season, or does it
drift as current-season PA accumulates? fl_ros_early_test.py only ever
checked ONE cutoff (~3 weeks in); this sweeps several cutoffs across the
season, for BOTH 2025 and 2026 independently (FL's only two real seasons --
two replications, not a lot, but it's what's available), fitting the full
model FRESH at each cutoff (same as fl_ros_early_test.py, not reusing the
annual frozen coefficients) and comparing the fitted coefficients, not just
R2/RMSE.

Usage:
    python fl_ros_coef_drift_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
from fl_ros_early_test import build_fold, K_FULL_SPEC, BB_FULL_SPEC, score

CUTOFF_WEEKS = [2, 5, 8, 11]

SEASONS = {
    2025: (bt.START_2025, bt.END_2025),
    2026: (bt.START_2026, bt.END_2026),
}


def fit_fold_coefs(fold, target_col, spec, n0):
    d = fold.dropna(subset=[target_col] + spec)
    w = mm.reliability_weight(d["PA_test"], n0).values
    m = bt.fit_weighted(d, target_col, spec, w)
    return m, len(d)


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    df_ref, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)],
                                       [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    full_pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025),
                           fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True)
    term = full_pool[full_pool["is_terminal"]]
    lg_k = (term["k_or_bb"] == "strikeout").mean()
    lg_bb = (term["k_or_bb"] == "walk").mean()

    k_rows, bb_rows = [], []
    for year, (season_start, season_end) in SEASONS.items():
        prior_windows = [(bt.START_2025, bt.END_2025)] if year == 2026 else []
        for wk in CUTOFF_WEEKS:
            cutoff = (pd.Timestamp(season_start) + pd.Timedelta(weeks=wk)).strftime("%Y-%m-%d")
            day_after = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.Timestamp(day_after) >= pd.Timestamp(season_end) - pd.Timedelta(weeks=2):
                continue  # need at least ~2 weeks of "rest of season" left to score against

            surprise_windows = prior_windows + [(season_start, cutoff)]
            try:
                fold = build_fold(raw, season_start, cutoff, surprise_windows, day_after, season_end,
                                   n0_k, n0_bb, lg_k, lg_bb)
            except Exception as e:
                print(f"  {year} wk{wk}: FAILED ({e})")
                continue

            m_k, n_k = fit_fold_coefs(fold, "K_rate_actual", K_FULL_SPEC, n0_k)
            m_bb, n_bb = fit_fold_coefs(fold, "BB_rate_actual", BB_FULL_SPEC, n0_bb)
            k_res = score(fold, "K_rate_actual", K_FULL_SPEC, n0_k, "K")
            bb_res = score(fold, "BB_rate_actual", BB_FULL_SPEC, n0_bb, "BB")

            k_rows.append({"year": year, "week": wk, "n": n_k, "R2_w": k_res["R2_w"],
                           "RMSE_w_pp": k_res["RMSE_w_pp"], **m_k.params.to_dict()})
            bb_rows.append({"year": year, "week": wk, "n": n_bb, "R2_w": bb_res["R2_w"],
                            "RMSE_w_pp": bb_res["RMSE_w_pp"], **m_bb.params.to_dict()})
            print(f"  {year} wk{wk} (cutoff={cutoff}): done, n_k={n_k} n_bb={n_bb}")

    k_df = pd.DataFrame(k_rows)
    bb_df = pd.DataFrame(bb_rows)

    print("\n" + "=" * 100)
    print("K% model coefficients by season-week-cutoff (fit fresh at each point)")
    print("=" * 100)
    print(k_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("BB% model coefficients by season-week-cutoff")
    print("=" * 100)
    print(bb_df.to_string(index=False))

    print("\n--- K% coefficient variability across cutoffs (coefficient of variation = SD/|mean|) ---")
    for col in ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level"]:
        vals = k_df[col]
        print(f"  {col}: mean={vals.mean():.3f}  SD={vals.std():.3f}  CV={vals.std()/abs(vals.mean()):.2f}")

    print("\n--- BB% coefficient variability across cutoffs ---")
    for col in ["naive_BB_rate", "fl_bb_rate_bridge", "horiz_var", "vert_var"]:
        vals = bb_df[col]
        print(f"  {col}: mean={vals.mean():.3f}  SD={vals.std():.3f}  CV={vals.std()/abs(vals.mean()):.2f}")
