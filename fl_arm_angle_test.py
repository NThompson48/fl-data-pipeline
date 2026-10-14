"""fl_arm_angle_test.py -- does a pitcher's average release arm angle
(over-the-top vs. sidearm/submarine) explain K%/BB%/RA9, beyond the
existing model? `arm_angle` exists on the real Supabase `pitches` table but
is NOT in fl_pitcher_features.load_fl_pitches_cached()'s column set, and
has real missingness (~49% null in the 2025 window, roughly uniform across
pitchers -- a per-pitch tracking/computation gap, not a per-pitcher one:
median per-pitcher coverage 53%, only 16/465 pitchers at 0%) -- a per-
pitcher MEAN over whatever non-null pitches exist is still a reasonable
trait estimate for nearly everyone.

Tests both models, same held-out disciplines already used for every other
arsenal-adjacent feature this session:
  - K%/BB%: Experiment 2 (fit full-2025->actual-1H2026, freeze, score
    2025+1H2026->actual-2H2026)
  - RA9: LOOCV on its own real n=96 training population

Usage:
    python fl_arm_angle_test.py
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

MIN_PITCHES = 20  # need a few real readings to trust a per-pitcher mean


def load_arm_angle_train_window() -> pd.DataFrame:
    q = f"""
        SELECT pitcher_id, arm_angle
        FROM pitches
        WHERE game_type = 'reg_season' AND pitcher_id IS NOT NULL
          AND date >= '{bt.START_2025}' AND date <= '{bt.END_2025}'
          AND arm_angle IS NOT NULL
    """
    return read_sql_df(q)


def compute_arm_angle(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("pitcher_id").agg(n=("arm_angle", "size"), arm_angle=("arm_angle", "mean")).reset_index()
    return out[out["n"] >= MIN_PITCHES][["pitcher_id", "arm_angle"]]


if __name__ == "__main__":
    print("Pulling arm_angle for the TRAIN window (full 2025)...")
    aa_raw = load_arm_angle_train_window()
    aa = compute_arm_angle(aa_raw)
    aa["pitcher_id"] = aa["pitcher_id"].astype(str)
    print(f"  n={len(aa)} pitchers with >= {MIN_PITCHES} real readings")
    print(aa["arm_angle"].describe())

    # ---------------- K%/BB%, Experiment 2 ---------------- #
    print("\n=== K%/BB%: Experiment 2 held-out ===")
    raw = fp.load_fl_pitches_cached()
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)
    lg_arm = aa["arm_angle"].median()
    for df in (df_tr, df_te):
        df["pitcher_id"] = df["pitcher_id"].astype(str)
        df["arm_angle"] = df["pitcher_id"].map(aa.set_index("pitcher_id")["arm_angle"]).fillna(lg_arm)

    print(f"corr(arm_angle, wh_sva_level) = {df_tr['arm_angle'].corr(df_tr['wh_sva_level']):.4f}")

    SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]
    k_base = bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"], df_te, "K_rate_actual", n0_k, "K% baseline")
    k_arm = bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["arm_angle"], df_te, "K_rate_actual", n0_k,
                            "K% + arm_angle")
    bb_base = bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"], df_te, "BB_rate_actual", n0_bb, "BB% baseline")
    bb_arm = bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"] + ["arm_angle"], df_te, "BB_rate_actual", n0_bb,
                             "BB% + arm_angle")
    print(pd.DataFrame([k_base, k_arm, bb_base, bb_arm]).to_string(index=False))

    w_k = mm.reliability_weight(df_tr["PA"], n0_k)
    m_k = smf.wls("K_rate_actual ~ " + " + ".join(SPEC["K"] + ["arm_angle"]), data=df_tr, weights=w_k).fit()
    print(f"\nK% model: arm_angle coef={m_k.params['arm_angle']:.5f} p={m_k.pvalues['arm_angle']:.4f}")
    w_bb = mm.reliability_weight(df_tr["PA"], n0_bb)
    m_bb = smf.wls("BB_rate_actual ~ " + " + ".join(SPEC["BB"] + ["arm_angle"]), data=df_tr, weights=w_bb).fit()
    print(f"BB% model: arm_angle coef={m_bb.params['arm_angle']:.5f} p={m_bb.pvalues['arm_angle']:.4f}")

    # ---------------- RA9, LOOCV on real n=96 ---------------- #
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
    m = m.merge(aa, on="pitcher_id", how="left")
    print(f"  arm_angle real for {m['arm_angle'].notna().sum()} of n={len(m)}")
    m["arm_angle"] = m["arm_angle"].fillna(m["arm_angle"].median())

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
        return r2, rmse

    loocv_eval(m, ra9mod.RA9_SPEC, "baseline")
    loocv_eval(m, ra9mod.RA9_SPEC + " + arm_angle", "+ arm_angle")

    w_full = mm.reliability_weight(m["ip_2026"], ra9mod.RA9_N0_EVAL).values
    mod = smf.wls(ra9mod.RA9_SPEC + " + arm_angle", data=m, weights=w_full).fit()
    print(f"  arm_angle coef={mod.params['arm_angle']:.5f}  p={mod.pvalues['arm_angle']:.4f}")
