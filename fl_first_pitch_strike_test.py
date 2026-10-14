"""fl_first_pitch_strike_test.py -- does a pitcher's first-pitch strike rate
(getting ahead 0-1 rather than behind 1-0) explain K%/BB%/RA9, beyond the
existing model?

`pitch_no` in the cached pull is a per-GAME cumulative counter, NOT a
within-PA pitch count (confirmed directly: one (game_id, pa_id) spans a
contiguous pitch_no range, not 1..n) -- the first pitch of a PA is the row
with MIN(pitch_no) within each (game_id, pa_id) group (pa_id alone recurs
across games, e.g. pa_id=28 appears in 1,725 different games -- game_id is
required to disambiguate).

Usage:
    python fl_first_pitch_strike_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from fl_supabase import read_sql_df
import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
import fl_ra9_trustmark_projection as ra9mod

MIN_PA = 20
STRIKE_CALLS = {"called_strike", "strike_called", "strike_swinging", "foul_not_fieldable",
                 "foul_fieldable", "foul_ball", "in_play", "automaticstrike"}


def compute_first_pitch_strike(period_df: pd.DataFrame) -> pd.DataFrame:
    d = period_df.dropna(subset=["pitch_no", "pitch_call"]).copy()
    first = d.loc[d.groupby(["pitcher_id", "game_id", "pa_id"])["pitch_no"].idxmin()]
    first["is_strike"] = first["pitch_call"].isin(STRIKE_CALLS)
    out = first.groupby("pitcher_id").agg(
        n_pa=("is_strike", "size"), fps_rate=("is_strike", "mean")).reset_index()
    return out[out["n_pa"] >= MIN_PA]


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()
    train_period = fp.filter_period(raw, bt.START_2025, bt.END_2025)

    print("Computing first-pitch strike rate over the TRAIN window...")
    fps = compute_first_pitch_strike(train_period)
    fps["pitcher_id"] = fps["pitcher_id"].astype(str)
    print(f"  n={len(fps)}")
    print(fps["fps_rate"].describe())

    print("\n=== K%/BB%: Experiment 2 held-out ===")
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)
    lg_fps = fps["fps_rate"].median()
    for df in (df_tr, df_te):
        df["pitcher_id"] = df["pitcher_id"].astype(str)
        df["fps_rate"] = df["pitcher_id"].map(fps.set_index("pitcher_id")["fps_rate"]).fillna(lg_fps)

    print(f"corr(fps_rate, naive_K_rate) = {df_tr['fps_rate'].corr(df_tr['naive_K_rate']):.4f}")
    print(f"corr(fps_rate, naive_BB_rate) = {df_tr['fps_rate'].corr(df_tr['naive_BB_rate']):.4f}")

    SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]
    k_base = bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"], df_te, "K_rate_actual", n0_k, "K% baseline")
    k_fps = bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["fps_rate"], df_te, "K_rate_actual", n0_k,
                            "K% + fps_rate")
    bb_base = bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"], df_te, "BB_rate_actual", n0_bb, "BB% baseline")
    bb_fps = bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"] + ["fps_rate"], df_te, "BB_rate_actual", n0_bb,
                             "BB% + fps_rate")
    print(pd.DataFrame([k_base, k_fps, bb_base, bb_fps]).to_string(index=False))

    w_k = mm.reliability_weight(df_tr["PA"], n0_k)
    m_k = smf.wls("K_rate_actual ~ " + " + ".join(SPEC["K"] + ["fps_rate"]), data=df_tr, weights=w_k).fit()
    print(f"\nK% fps_rate: coef={m_k.params['fps_rate']:.4f} p={m_k.pvalues['fps_rate']:.4f}")
    w_bb = mm.reliability_weight(df_tr["PA"], n0_bb)
    m_bb = smf.wls("BB_rate_actual ~ " + " + ".join(SPEC["BB"] + ["fps_rate"]), data=df_tr, weights=w_bb).fit()
    print(f"BB% fps_rate: coef={m_bb.params['fps_rate']:.4f} p={m_bb.pvalues['fps_rate']:.4f}")

    print("\n=== RA9: LOOCV on real n=96 training population ===")
    war = read_sql_df("select * from war_pitchers where game_type='reg_season'")
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

    names = raw[["pitcher", "pitcher_id"]].drop_duplicates("pitcher")
    m = m.merge(names, left_on="player", right_on="pitcher", how="left")
    m["pitcher_id"] = m["pitcher_id"].astype(str)
    m = m.merge(fps[["pitcher_id", "fps_rate"]], on="pitcher_id", how="left")
    print(f"  fps_rate real for {m['fps_rate'].notna().sum()} of n={len(m)}")
    m["fps_rate"] = m["fps_rate"].fillna(m["fps_rate"].median())

    def loocv_eval(m, spec, label):
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

    loocv_eval(m, ra9mod.RA9_SPEC, "baseline")
    loocv_eval(m, ra9mod.RA9_SPEC + " + fps_rate", "+ fps_rate")
    w_full = mm.reliability_weight(m["ip_2026"], ra9mod.RA9_N0_EVAL).values
    mod = smf.wls(ra9mod.RA9_SPEC + " + fps_rate", data=m, weights=w_full).fit()
    print(f"  RA9 fps_rate: coef={mod.params['fps_rate']:.4f}  p={mod.pvalues['fps_rate']:.4f}")
