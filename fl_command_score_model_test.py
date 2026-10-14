"""fl_command_score_model_test.py -- does the MLB-calibrated Command Score
(mlb_opencommand_locvar_test.py's frozen `inferred_in ~ horiz_var + vert_var`
model, applied to FL's own horiz_var/vert_var with training-range clipping)
do BETTER than horiz_var/vert_var used directly?

- BB%: command_score REPLACES horiz_var+vert_var (already in the production
  spec) -- a fixed, externally-calibrated linear combination vs. two
  freely-fit coefficients. Mechanically a CONSTRAINT on the current model
  (same two inputs, one fewer free parameter), so it can only match or lose
  in-sample; the real question is whether it generalizes better held out.
- RA9: horiz_var/vert_var aren't in RA9_SPEC at all -- tests both ADDING
  command_score and ADDING raw horiz_var+vert_var as new candidates, same
  LOOCV discipline as every other RA9 candidate this session, for a fair
  side-by-side (does the MLB-calibrated combination help where the raw
  ingredients wouldn't, or vice versa).

Usage:
    python fl_command_score_model_test.py
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

# Frozen MLB model (mlb_opencommand_locvar_test.py, n=769 workload-qualified pitchers)
MLB_INTERCEPT = 4.424373963818792
MLB_HORIZ_COEF = 3.9415731433625307
MLB_VERT_COEF = 4.281096060834336
MLB_HV_RANGE = (0.353837, 1.152937)
MLB_VV_RANGE = (0.262952, 1.694887)


def add_command_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hv_c = df["horiz_var"].clip(*MLB_HV_RANGE)
    vv_c = df["vert_var"].clip(*MLB_VV_RANGE)
    df["command_score"] = MLB_INTERCEPT + MLB_HORIZ_COEF * hv_c + MLB_VERT_COEF * vv_c
    return df


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print("\n=== BB%: Experiment 2 held-out -- command_score REPLACES horiz_var+vert_var ===")
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)
    df_tr = add_command_score(df_tr)
    df_te = add_command_score(df_te)

    SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]
    bb_base = bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"], df_te, "BB_rate_actual", n0_bb,
                              "BB% baseline (horiz_var+vert_var)")
    bb_cs_spec = [c for c in SPEC["BB"] if c not in ("horiz_var", "vert_var")] + ["command_score"]
    bb_cs = bt.run_heldout(df_tr, "BB_rate_actual", bb_cs_spec, df_te, "BB_rate_actual", n0_bb,
                            "BB% command_score REPLACES horiz_var+vert_var")
    bb_both_spec = SPEC["BB"] + ["command_score"]
    bb_both = bt.run_heldout(df_tr, "BB_rate_actual", bb_both_spec, df_te, "BB_rate_actual", n0_bb,
                              "BB% command_score ADDED alongside horiz_var+vert_var")
    print(pd.DataFrame([bb_base, bb_cs, bb_both]).to_string(index=False))

    w_bb = mm.reliability_weight(df_tr["PA"], n0_bb)
    mod_cs = smf.wls("BB_rate_actual ~ " + " + ".join(bb_cs_spec), data=df_tr, weights=w_bb).fit()
    print(f"\nBB% command_score (replacement) coef={mod_cs.params['command_score']:.5f} "
          f"p={mod_cs.pvalues['command_score']:.4f}")

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

    train_period = fp.filter_period(raw, bt.START_2025, bt.END_2025)
    lv = fp.compute_loc_var(train_period)
    lv["pitcher_id"] = lv["pitcher_id"].astype(str)
    m = m.merge(lv[["pitcher_id", "horiz_var", "vert_var"]], on="pitcher_id", how="left")
    print(f"  horiz_var/vert_var real for {m['horiz_var'].notna().sum()} of n={len(m)}")
    m["horiz_var"] = m["horiz_var"].fillna(m["horiz_var"].median())
    m["vert_var"] = m["vert_var"].fillna(m["vert_var"].median())
    m = add_command_score(m)

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

    loocv_eval(m, ra9mod.RA9_SPEC, "baseline (no location-var feature)")
    loocv_eval(m, ra9mod.RA9_SPEC + " + command_score", "+ command_score")
    loocv_eval(m, ra9mod.RA9_SPEC + " + horiz_var + vert_var", "+ raw horiz_var + vert_var")

    w_full = mm.reliability_weight(m["ip_2026"], ra9mod.RA9_N0_EVAL).values
    mod1 = smf.wls(ra9mod.RA9_SPEC + " + command_score", data=m, weights=w_full).fit()
    print(f"  RA9 command_score: coef={mod1.params['command_score']:.5f}  p={mod1.pvalues['command_score']:.4f}")
    mod2 = smf.wls(ra9mod.RA9_SPEC + " + horiz_var + vert_var", data=m, weights=w_full).fit()
    print(f"  RA9 horiz_var: coef={mod2.params['horiz_var']:.5f}  p={mod2.pvalues['horiz_var']:.4f}")
    print(f"  RA9 vert_var: coef={mod2.params['vert_var']:.5f}  p={mod2.pvalues['vert_var']:.4f}")
