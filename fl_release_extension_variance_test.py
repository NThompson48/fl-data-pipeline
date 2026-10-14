"""fl_release_extension_variance_test.py -- release point variance and
extension variance, tested fresh/comprehensively on all three models
(K%, BB%, RA9) with TODAY's production specs.

Release point variance was previously tested in pieces, on older specs:
"Release point as a K% regressor" (DECISION_LOG.md) covered raw LEVEL and
BETWEEN-pitch-type ("deception") variance for K% only, both null; a
WITHIN-pitch-type release-point variance was separately noted as "found
weak for BB%" on the OLD horiz_dist/vert_dist-era BB% spec (since replaced
by horiz_var/vert_var). Neither was tested on RA9 (a newer model). This
re-tests release_pos_z/rel_side_as variance (pooled per-pitcher SD, same
simple construction as the arm_angle_sd/velo_sd "consistency" tests
earlier this session) on CURRENT specs, all three models.

Extension variance (release-distance consistency) has never been tested at
all -- only extension's MEAN is a shipped K% regressor.

Usage:
    python fl_release_extension_variance_test.py
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

MIN_PITCHES = 50


def compute_variances(period_df: pd.DataFrame) -> pd.DataFrame:
    d = period_df.dropna(subset=["release_pos_z", "rel_side_as", "extension"]).copy()
    out = d.groupby("pitcher_id").agg(
        n=("release_pos_z", "size"),
        release_z_sd=("release_pos_z", "std"),
        rel_side_sd=("rel_side_as", "std"),
        extension_sd=("extension", "std"),
    ).reset_index()
    return out[out["n"] >= MIN_PITCHES]


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()
    train_period = fp.filter_period(raw, bt.START_2025, bt.END_2025)

    print("Computing release point / extension variance over the TRAIN window...")
    var = compute_variances(train_period)
    var["pitcher_id"] = var["pitcher_id"].astype(str)
    print(f"  n={len(var)}")
    print(var[["release_z_sd", "rel_side_sd", "extension_sd"]].describe())

    print("\n=== K%/BB%: Experiment 2 held-out ===")
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)
    feats = ["release_z_sd", "rel_side_sd", "extension_sd"]
    for df in (df_tr, df_te):
        df["pitcher_id"] = df["pitcher_id"].astype(str)
        for f in feats:
            df[f] = df["pitcher_id"].map(var.set_index("pitcher_id")[f]).fillna(var[f].median())

    SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]
    results = [bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"], df_te, "K_rate_actual", n0_k, "K% baseline")]
    for f in feats:
        results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + [f], df_te, "K_rate_actual", n0_k,
                                       f"K% + {f}"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"], df_te, "BB_rate_actual", n0_bb, "BB% baseline"))
    for f in feats:
        results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"] + [f], df_te, "BB_rate_actual", n0_bb,
                                       f"BB% + {f}"))
    print(pd.DataFrame(results).to_string(index=False))

    w_k = mm.reliability_weight(df_tr["PA"], n0_k)
    for f in feats:
        mod = smf.wls("K_rate_actual ~ " + " + ".join(SPEC["K"] + [f]), data=df_tr, weights=w_k).fit()
        print(f"K% {f}: coef={mod.params[f]:.5f} p={mod.pvalues[f]:.4f}")
    w_bb = mm.reliability_weight(df_tr["PA"], n0_bb)
    for f in feats:
        mod = smf.wls("BB_rate_actual ~ " + " + ".join(SPEC["BB"] + [f]), data=df_tr, weights=w_bb).fit()
        print(f"BB% {f}: coef={mod.params[f]:.5f} p={mod.pvalues[f]:.4f}")

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
    m = m.merge(var[["pitcher_id"] + feats], on="pitcher_id", how="left")
    print(f"  real coverage: " + ", ".join(f"{f}={m[f].notna().sum()}" for f in feats) + f" of n={len(m)}")
    for f in feats:
        m[f] = m[f].fillna(m[f].median())

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
    for f in feats:
        loocv_eval(m, ra9mod.RA9_SPEC + f" + {f}", f"+ {f}")

    w_full = mm.reliability_weight(m["ip_2026"], ra9mod.RA9_N0_EVAL).values
    for f in feats:
        mod = smf.wls(ra9mod.RA9_SPEC + f" + {f}", data=m, weights=w_full).fit()
        print(f"  RA9 {f}: coef={mod.params[f]:.5f}  p={mod.pvalues[f]:.4f}")
