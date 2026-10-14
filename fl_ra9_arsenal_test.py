"""fl_ra9_arsenal_test.py -- same five arsenal-interaction constructions
already tested against K%/BB% (fl_arsenal_interaction_test.py,
fl_repertoire_diversity_test.py, fl_tunneling_test.py), now tested against
the RA9 model instead, on its own real training population (n=96, the one
genuine 2025-actual -> 2026-actual transition) via the SAME LOOCV discipline
RA9_SPEC itself was validated with (leave-one-pitcher-out, refit per fold).

Each candidate is computed from the pitcher's PRIOR season (2025) pitches,
same convention as ip_per_g_prior/pitches_per_pa_prior already in RA9_SPEC,
matched to war_pitchers by player NAME (that table has no pitcher_id).

Usage:
    python fl_ra9_arsenal_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
from fl_supabase import read_sql_df
import fl_ra9_trustmark_projection as ra9mod

from fl_arsenal_interaction_test import compute_arsenal_features
from fl_repertoire_diversity_test import compute_diversity
from fl_tunneling_test import load_trajectory_train_window, compute_tunneling


def loocv_eval(m: pd.DataFrame, spec: str, label: str) -> dict:
    preds = np.full(len(m), np.nan)
    for i in range(len(m)):
        train = m.drop(index=m.index[i])
        w = mm.reliability_weight(train["ip_2026"], ra9mod.RA9_N0_EVAL).values
        model = smf.wls(spec, data=train, weights=w).fit()
        preds[i] = model.predict(m.iloc[[i]]).values[0]
    w_all = mm.reliability_weight(m["ip_2026"], ra9mod.RA9_N0_EVAL).values
    r2 = mm.weighted_r2(m["target"].values, preds, w_all)
    rmse = mm.weighted_rmse(m["target"].values, preds, w_all)
    print(f"  {label}: LOOCV R2_w={r2:.4f}  RMSE_w={rmse:.4f}")
    return {"label": label, "R2_w": r2, "RMSE_w": rmse}


if __name__ == "__main__":
    print("Loading war_pitchers + cached FL pitches...")
    war = read_sql_df("select * from war_pitchers where game_type='reg_season'")
    raw = fp.load_fl_pitches_cached()

    print("Building RA9 training frame (n=96, same as fit_ra9_baseline_model)...")
    w25 = war[war["year"] == 2025].copy()
    w26 = war[war["year"] == 2026].copy()
    m = w25.merge(w26, on="player", suffixes=("_2025", "_2026"))
    m = m[(m["bf_2025"] >= 100) & (m["bf_2026"] >= 100)].copy()
    m["K_pred"] = m["so_2025"] / m["bf_2025"]
    m["BB_pred"] = m["bb_2025"] / m["bf_2025"]
    m["ppf"] = m["ppf_2026"]
    m["ip_per_g_prior"] = (m["ip_2025"] + ra9mod.N0_IPG * ra9mod.LG_IPG) / (m["g_2025"] + ra9mod.N0_IPG)
    m["target"] = m["ra9_2026"]
    ppa_2025 = ra9mod.pitches_per_pa(raw, bt.START_2025, bt.END_2025).rename(
        columns={"pitches_per_pa": "pitches_per_pa_prior"})
    m = m.merge(ppa_2025, left_on="player", right_on="pitcher", how="left").drop(columns="pitcher")
    ppa_aux_model = smf.ols(ra9mod.PPA_RESID_SPEC, data=m).fit()
    m["ppa_resid"] = ppa_aux_model.resid
    print(f"  n={len(m)}")

    names = raw[["pitcher", "pitcher_id"]].drop_duplicates("pitcher")
    m = m.merge(names, left_on="player", right_on="pitcher", how="left")
    m["pitcher_id"] = m["pitcher_id"].astype(str)

    print("\nComputing arsenal features over the prior (2025) season...")
    train_period = fp.filter_period(raw, bt.START_2025, bt.END_2025)

    arsenal = compute_arsenal_features(train_period)
    arsenal["pitcher_id"] = arsenal["pitcher_id"].astype(str)
    m = m.merge(arsenal, on="pitcher_id", how="left")
    m["velo_diff"] = m["velo_diff"].fillna(m["velo_diff"].median())
    m["arsenal_depth"] = m["arsenal_depth"].fillna(m["arsenal_depth"].median())

    div = compute_diversity(train_period)
    div["pitcher_id"] = div["pitcher_id"].astype(str)
    m = m.merge(div[["pitcher_id", "velo_sd", "move_sd"]], on="pitcher_id", how="left")
    m["velo_sd"] = m["velo_sd"].fillna(m["velo_sd"].median())
    m["move_sd"] = m["move_sd"].fillna(m["move_sd"].median())

    traj = load_trajectory_train_window()
    tun = compute_tunneling(traj)
    tun["pitcher_id"] = tun["pitcher_id"].astype(str)
    m = m.merge(tun[["pitcher_id", "spatial_sd"]], on="pitcher_id", how="left")
    m["spatial_sd"] = m["spatial_sd"].fillna(m["spatial_sd"].median())
    m["tunneling_score"] = -m["spatial_sd"]

    print(f"  matched: velo_diff/arsenal_depth real for {arsenal['pitcher_id'].isin(m['pitcher_id']).sum()}, "
          f"diversity real for {div['pitcher_id'].isin(m['pitcher_id']).sum()}, "
          f"tunneling real for {tun['pitcher_id'].isin(m['pitcher_id']).sum()} of n={len(m)}")

    print("\n=== RA9 LOOCV: baseline vs. + each arsenal candidate (forward direction, 2025->2026) ===")
    base = ra9mod.RA9_SPEC
    results = [loocv_eval(m, base, "baseline (RA9_SPEC)")]
    for feat in ["velo_diff", "arsenal_depth", "velo_sd", "move_sd", "tunneling_score"]:
        results.append(loocv_eval(m, base + f" + {feat}", f"+ {feat}"))

    print()
    print(pd.DataFrame(results).to_string(index=False))

    print("\nIn-sample coefficient check (full-96-fit, not LOOCV) for each candidate:")
    w_full = mm.reliability_weight(m["ip_2026"], ra9mod.RA9_N0_EVAL).values
    for feat in ["velo_diff", "arsenal_depth", "velo_sd", "move_sd", "tunneling_score"]:
        mod = smf.wls(base + f" + {feat}", data=m, weights=w_full).fit()
        print(f"  {feat}: coef={mod.params[feat]:.5f}  p={mod.pvalues[feat]:.4f}")
