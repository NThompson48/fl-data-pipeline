"""fl_offspeed_zone_mech_consistency_test.py -- two more candidates:

1. Non-fastball Zone% -- the complement of the already-tested (and
   rejected) `fl_pitcher_features.compute_fb_zone_pct` ("Fastball Zone% as
   a direct BB% regressor" in DECISION_LOG.md: real standalone signal,
   redundant once added to the full model). Same construction, applied to
   OFFSPEED/breaking pitches instead -- being able to throw secondary
   stuff for strikes (not just bury it as a chase pitch) is a different,
   untested mechanism: more about getting ahead/finishing with secondary
   stuff (K%-relevant) than fastball command (BB%-relevant).
2. Mechanical consistency -- per-pitcher standard deviation of arm_angle
   (release-angle repeatability), the natural "how consistent is the
   delivery" analog to the already-tested within-pitch-type release-point
   variance (found weak for BB%, not tested for K%) and to Velocity/
   Movement Diversity+ (which rewarded SPREAD; this rewards the opposite,
   LOW spread = more repeatable).

Both tested on both models, same held-out disciplines as every other
candidate this session: K%/BB% via Experiment 2, RA9 via LOOCV on its own
real n=96 population.

Usage:
    python fl_offspeed_zone_mech_consistency_test.py
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

MIN_OFF_PITCHES = 20
MIN_ARM_ANGLE_READINGS = 20


def compute_offspeed_zone_pct(df: pd.DataFrame, min_off: int = MIN_OFF_PITCHES) -> pd.DataFrame:
    off = df[~df["pitch_type"].isin(fp.FASTBALL_TYPES + ["", "not_him", "other"])].dropna(
        subset=["plate_x", "plate_z", "sz_top", "sz_bot"]).copy()
    off["in_zone"] = ((off["plate_x"].abs() <= fp.ZONE_HALF_WIDTH) & (off["plate_z"] >= off["sz_bot"])
                       & (off["plate_z"] <= off["sz_top"]))
    agg = off.groupby("pitcher_id").agg(off_n=("in_zone", "size"), off_zone_pct=("in_zone", "mean")).reset_index()
    return agg[agg["off_n"] >= min_off][["pitcher_id", "off_zone_pct"]]


def load_arm_angle_train_window() -> pd.DataFrame:
    q = f"""
        SELECT pitcher_id, arm_angle
        FROM pitches
        WHERE game_type = 'reg_season' AND pitcher_id IS NOT NULL
          AND date >= '{bt.START_2025}' AND date <= '{bt.END_2025}'
          AND arm_angle IS NOT NULL
    """
    return read_sql_df(q)


def compute_arm_angle_sd(df: pd.DataFrame, min_n: int = MIN_ARM_ANGLE_READINGS) -> pd.DataFrame:
    out = df.groupby("pitcher_id").agg(n=("arm_angle", "size"), arm_angle_sd=("arm_angle", "std")).reset_index()
    return out[out["n"] >= min_n][["pitcher_id", "arm_angle_sd"]]


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()
    train_period = fp.filter_period(raw, bt.START_2025, bt.END_2025)

    print("Computing non-fastball Zone% over the TRAIN window...")
    off_zone = compute_offspeed_zone_pct(train_period)
    off_zone["pitcher_id"] = off_zone["pitcher_id"].astype(str)
    print(f"  n={len(off_zone)}  describe:\n{off_zone['off_zone_pct'].describe()}")

    print("\nPulling arm_angle and computing mechanical consistency (SD)...")
    aa_raw = load_arm_angle_train_window()
    mech = compute_arm_angle_sd(aa_raw)
    mech["pitcher_id"] = mech["pitcher_id"].astype(str)
    print(f"  n={len(mech)}  describe:\n{mech['arm_angle_sd'].describe()}")

    print("\n=== K%/BB%: Experiment 2 held-out ===")
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)
    lg_off = off_zone["off_zone_pct"].median()
    lg_mech = mech["arm_angle_sd"].median()
    for df in (df_tr, df_te):
        df["pitcher_id"] = df["pitcher_id"].astype(str)
        df["off_zone_pct"] = df["pitcher_id"].map(off_zone.set_index("pitcher_id")["off_zone_pct"]).fillna(lg_off)
        df["arm_angle_sd"] = df["pitcher_id"].map(mech.set_index("pitcher_id")["arm_angle_sd"]).fillna(lg_mech)

    SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]
    results = []
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"], df_te, "K_rate_actual", n0_k, "K% baseline"))
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["off_zone_pct"], df_te, "K_rate_actual", n0_k,
                                   "K% + off_zone_pct"))
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["arm_angle_sd"], df_te, "K_rate_actual", n0_k,
                                   "K% + arm_angle_sd"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"], df_te, "BB_rate_actual", n0_bb, "BB% baseline"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"] + ["off_zone_pct"], df_te, "BB_rate_actual",
                                   n0_bb, "BB% + off_zone_pct"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"] + ["arm_angle_sd"], df_te, "BB_rate_actual",
                                   n0_bb, "BB% + arm_angle_sd"))
    print(pd.DataFrame(results).to_string(index=False))

    w_k = mm.reliability_weight(df_tr["PA"], n0_k)
    for feat in ["off_zone_pct", "arm_angle_sd"]:
        mod = smf.wls("K_rate_actual ~ " + " + ".join(SPEC["K"] + [feat]), data=df_tr, weights=w_k).fit()
        print(f"K% {feat}: coef={mod.params[feat]:.5f} p={mod.pvalues[feat]:.4f}")
    w_bb = mm.reliability_weight(df_tr["PA"], n0_bb)
    for feat in ["off_zone_pct", "arm_angle_sd"]:
        mod = smf.wls("BB_rate_actual ~ " + " + ".join(SPEC["BB"] + [feat]), data=df_tr, weights=w_bb).fit()
        print(f"BB% {feat}: coef={mod.params[feat]:.5f} p={mod.pvalues[feat]:.4f}")

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
    m = m.merge(off_zone, on="pitcher_id", how="left")
    m = m.merge(mech, on="pitcher_id", how="left")
    print(f"  off_zone_pct real for {m['off_zone_pct'].notna().sum()}, "
          f"arm_angle_sd real for {m['arm_angle_sd'].notna().sum()} of n={len(m)}")
    m["off_zone_pct"] = m["off_zone_pct"].fillna(m["off_zone_pct"].median())
    m["arm_angle_sd"] = m["arm_angle_sd"].fillna(m["arm_angle_sd"].median())

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
    loocv_eval(m, ra9mod.RA9_SPEC + " + off_zone_pct", "+ off_zone_pct")
    loocv_eval(m, ra9mod.RA9_SPEC + " + arm_angle_sd", "+ arm_angle_sd")

    w_full = mm.reliability_weight(m["ip_2026"], ra9mod.RA9_N0_EVAL).values
    for feat in ["off_zone_pct", "arm_angle_sd"]:
        mod = smf.wls(ra9mod.RA9_SPEC + f" + {feat}", data=m, weights=w_full).fit()
        print(f"  RA9 {feat}: coef={mod.params[feat]:.5f}  p={mod.pvalues[feat]:.4f}")
