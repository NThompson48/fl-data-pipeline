"""
fl_error_analysis_v2.py -- error analysis on the CURRENT production K%/BB%/RA9
models (K%: naive+other_league+surprise_prior OLS; BB%: blended_BB_rate
informative-prior spec; RA9: K%+BB%-only). Extends/updates
fl_backtest_error_analysis.py's K%/BB% analysis (which ran on an older
changepoint-era model spec and never covered RA9) with team, pitcher-trait,
and "descriptive occurrence" breakdowns, per the user's explicit ask to look
broadly rather than just repeat the earlier narrower check.

K%/BB%: held out on Experiment 2's real test period (2025+1H2026 -> actual
2H2026, n=260). RA9: leave-one-pitcher-out on the only real transition
available (2025 actual -> 2026 actual, war_pitchers, n=96) -- same
populations already reviewed earlier this session, now broken down by
residual direction instead of just aggregate R2/RMSE/MAE.

Usage:
    python fl_error_analysis_v2.py
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import mlb_marcel as mm
from fl_supabase import read_sql_df
from fl_ra9_trustmark_projection import RA9_SPEC, RA9_N0_EVAL
from fl_replacement_prob_calibration import build_training_frame, loocv_predict

pd.set_option("display.width", 140)


def clean_name(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def flip_lastfirst(s):
    if "," not in s:
        return s
    last, first = s.split(",", 1)
    return first.strip() + " " + last.strip()


def team_breakdown(df, resid_col, weight_col, label, min_n=5):
    g = df.groupby("team").apply(
        lambda d: pd.Series({
            "n": len(d),
            "mean_resid": np.average(d[resid_col], weights=d[weight_col]) if d[weight_col].sum() > 0 else d[resid_col].mean(),
        }), include_groups=False
    ).reset_index()
    g = g[g["n"] >= min_n].sort_values("mean_resid")
    print(f"\n--- {label}: team mean residual (teams with n>={min_n}) ---")
    print(g.to_string(index=False))


def trait_corr(df, resid_col, weight_col, traits, label):
    print(f"\n--- {label}: residual correlation with continuous traits ---")
    for t in traits:
        d = df.dropna(subset=[t, resid_col])
        if len(d) < 20:
            continue
        r = np.corrcoef(d[t], d[resid_col])[0, 1]
        print(f"  {t:20s} r={r:+.3f}  (n={len(d)})")


def tercile_breakdown(df, col, resid_col, weight_col, label):
    d = df.dropna(subset=[col, resid_col]).copy()
    if len(d) < 30:
        return
    d["tercile"] = pd.qcut(d[col], 3, labels=["low", "mid", "high"], duplicates="drop")
    g = d.groupby("tercile", observed=True).apply(
        lambda s: pd.Series({"n": len(s), "mean_resid": np.average(s[resid_col], weights=s[weight_col])}),
        include_groups=False
    )
    print(f"\n--- {label}: by {col} tercile ---")
    print(g.to_string())


def categorical_breakdown(df, col, resid_col, weight_col, label):
    d = df.dropna(subset=[col, resid_col])
    g = d.groupby(col).apply(
        lambda s: pd.Series({"n": len(s), "mean_resid": np.average(s[resid_col], weights=s[weight_col])}),
        include_groups=False
    )
    print(f"\n--- {label}: by {col} ---")
    print(g.to_string())


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    # =================================================================== #
    # K% and BB%
    # =================================================================== #
    K_SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]["K"]
    K_NAIVE = ["naive_K_rate"]
    df_tr2, n0_k2, n0_bb2 = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=True)
    df_te2, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)], [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=True)

    d_tr = df_tr2.dropna(subset=["K_rate_actual"] + K_SPEC).copy()
    d_te = df_te2.dropna(subset=["K_rate_actual"] + K_SPEC).copy()
    w_tr = mm.reliability_weight(d_tr["PA_test"], n0_k2).values
    w_te = mm.reliability_weight(d_te["PA_test"], n0_k2).values
    m_k = bt.fit_weighted(d_tr, "K_rate_actual", K_SPEC, w_tr)
    d_te["w"] = w_te
    d_te["k_resid"] = d_te["K_rate_actual"].values - m_k.predict(d_te).values  # + = overperformed (actual K% higher)

    d_tr2 = df_tr2.dropna(subset=["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "horiz_var", "vert_var"]).copy()
    d_te2 = df_te2.dropna(subset=["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "horiz_var", "vert_var"]).copy()
    w_tr2 = mm.reliability_weight(d_tr2["PA_test"], n0_bb2).values
    w_te2 = mm.reliability_weight(d_te2["PA_test"], n0_bb2).values
    term_tr = fp.filter_period(raw, bt.START_2025, bt.END_2025)
    term_tr = term_tr[term_tr["is_terminal"]]
    lg_bb_tr = (term_tr["k_or_bb"] == "walk").mean()
    d_tr2["BB_raw"] = d_tr2["naive_BB_rate"] * (d_tr2["PA"] + n0_bb2) - n0_bb2 * lg_bb_tr
    n0_bridge_bb = mm.fit_n0_bridge(d_tr2["BB_raw"], d_tr2["PA"], n0_bb2, lg_bb_tr,
                                     d_tr2["fl_bb_rate_bridge"], d_tr2["BB_rate_actual"], w_tr2)
    d_tr2["blended_BB_rate"] = mm.blended_rate(d_tr2["BB_raw"], d_tr2["PA"], n0_bb2, lg_bb_tr,
                                                n0_bridge_bb, d_tr2["fl_bb_rate_bridge"])
    term_te_pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025),
                              fp.filter_period(raw, bt.START_2026, bt.CUTOFF_2026)], ignore_index=True)
    term_te_pool = term_te_pool[term_te_pool["is_terminal"]]
    lg_bb_te = (term_te_pool["k_or_bb"] == "walk").mean()
    d_te2["BB_raw"] = d_te2["naive_BB_rate"] * (d_te2["PA"] + n0_bb2) - n0_bb2 * lg_bb_te
    d_te2["blended_BB_rate"] = mm.blended_rate(d_te2["BB_raw"], d_te2["PA"], n0_bb2, lg_bb_te,
                                                n0_bridge_bb, d_te2["fl_bb_rate_bridge"])
    BB_SPEC = ["blended_BB_rate", "horiz_var", "vert_var"]
    m_bb = bt.fit_weighted(d_tr2, "BB_rate_actual", BB_SPEC, w_tr2)
    d_te2["w"] = w_te2
    # + = actual BB% higher than predicted = pitcher did WORSE than predicted (BB% is bad)
    d_te2["bb_resid"] = d_te2["BB_rate_actual"].values - m_bb.predict(d_te2).values

    # stuff_plus for the test pool (same 2025+1H2026 window used for the test features)
    feat_pool_te = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025),
                              fp.filter_period(raw, bt.START_2026, bt.CUTOFF_2026)], ignore_index=True)
    wh_sva_pitch = fp.compute_pooled_wh_sva(feat_pool_te)
    stuff_plus_df, _, _ = fp.compute_stuff_plus(wh_sva_pitch)

    # team / throws
    recent_team = (raw.sort_values("date").groupby("pitcher_id")["pitcher_team"].last().rename("team").reset_index())
    throws = raw.dropna(subset=["p_throws_clean"]).groupby("pitcher_id")["p_throws_clean"].agg(lambda s: s.mode().iloc[0]).rename("throws").reset_index()

    d_te = d_te.merge(recent_team, on="pitcher_id", how="left").merge(throws, on="pitcher_id", how="left").merge(stuff_plus_df, on="pitcher_id", how="left")
    d_te2 = d_te2.merge(recent_team, on="pitcher_id", how="left").merge(throws, on="pitcher_id", how="left").merge(stuff_plus_df, on="pitcher_id", how="left")

    print("\n" + "=" * 90)
    print(f"K-pct ERROR ANALYSIS (held out, 2025+1H2026 -> actual 2H2026, n={len(d_te)})")
    print("=" * 90)
    team_breakdown(d_te, "k_resid", "w", "K%")
    trait_corr(d_te, "k_resid", "w", ["wh_sva_level", "spin_z_norm", "extension", "stuff_plus", "fl_k_rate_bridge"], "K%")
    categorical_breakdown(d_te, "throws", "k_resid", "w", "K%")
    categorical_breakdown(d_te, "has_college_match", "k_resid", "w", "K%")
    tercile_breakdown(d_te, "PA_test", "k_resid", "w", "K%")
    tercile_breakdown(d_te, "naive_K_rate", "k_resid", "w", "K%")

    print("\n" + "=" * 90)
    print(f"BB-pct ERROR ANALYSIS (held out, 2025+1H2026 -> actual 2H2026, n={len(d_te2)})")
    print("=" * 90)
    team_breakdown(d_te2, "bb_resid", "w", "BB%")
    trait_corr(d_te2, "bb_resid", "w", ["horiz_var", "vert_var", "stuff_plus", "fl_bb_rate_bridge"], "BB%")
    categorical_breakdown(d_te2, "throws", "bb_resid", "w", "BB%")
    categorical_breakdown(d_te2, "has_college_match", "bb_resid", "w", "BB%")
    tercile_breakdown(d_te2, "PA_test", "bb_resid", "w", "BB%")
    tercile_breakdown(d_te2, "naive_BB_rate", "bb_resid", "w", "BB%")

    # =================================================================== #
    # RA9
    # =================================================================== #
    print("\nLoading war_pitchers for RA9 error analysis...")
    war = read_sql_df("select player, year, bf, ip, so, bb, ra, ra9, rar, team, throws, gs, g, ip_sp, ip_rp from war_pitchers where game_type='reg_season'")
    # build manually (not via build_training_frame) to keep team/throws/role columns
    w25 = war[war["year"] == 2025].copy()
    w26 = war[war["year"] == 2026].copy()
    m = w25.merge(w26, on="player", suffixes=("_2025", "_2026"))
    m = m[(m["bf_2025"] >= 100) & (m["bf_2026"] >= 100)].copy().reset_index(drop=True)
    m["K_pred"] = m["so_2025"] / m["bf_2025"]
    m["BB_pred"] = m["bb_2025"] / m["bf_2025"]
    m["target"] = m["ra9_2026"]
    m["role"] = np.where(m["ip_sp_2026"] >= m["ip_rp_2026"], "SP", "RP")

    m["ra9_pred_loocv"] = loocv_predict(m)
    w_rel = mm.reliability_weight(m["ip_2026"], RA9_N0_EVAL).values
    m["w"] = w_rel
    # + = actual RA9 higher than predicted = pitcher did WORSE (allowed more runs) than predicted
    m["ra9_resid"] = m["target"] - m["ra9_pred_loocv"]

    # merge FL pitch-level traits (stuff_plus, wh_sva_level) for this same n=96
    # population via name match -- known collision risk, flagged not fixed
    # (same caveat as the earlier error-analysis entry in DECISION_LOG.md)
    m["name_clean"] = m["player"].apply(lambda s: clean_name(flip_lastfirst(s)))
    pool_26 = fp.filter_period(raw, bt.START_2026, bt.END_2026)
    name_of_id = pool_26[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    name_of_id["name_clean"] = name_of_id["pitcher"].apply(clean_name)
    wh_level_26, _ = bt.wh_sva_level_and_changepoint(pool_26)
    wh_level_26 = wh_level_26.merge(name_of_id, on="pitcher_id", how="left")
    # dedup: keep the id with the most 2026 pitches per name (per the known
    # name-collision caveat already documented in DECISION_LOG.md)
    counts = pool_26.groupby("pitcher_id").size().rename("n_pitches").reset_index()
    wh_level_26 = wh_level_26.merge(counts, on="pitcher_id", how="left").sort_values("n_pitches", ascending=False)
    wh_level_26 = wh_level_26.drop_duplicates("name_clean", keep="first")
    m = m.merge(wh_level_26[["name_clean", "wh_sva_level"]], on="name_clean", how="left")

    print("\n" + "=" * 90)
    print(f"RA9 ERROR ANALYSIS (LOOCV, real 2025 actual -> 2026 actual, n={len(m)})")
    print("=" * 90)
    m = m.rename(columns={"team_2026": "team"})
    team_breakdown(m, "ra9_resid", "w", "RA9")
    trait_corr(m, "ra9_resid", "w", ["wh_sva_level", "K_pred", "BB_pred"], "RA9")
    categorical_breakdown(m, "throws_2026", "ra9_resid", "w", "RA9")
    categorical_breakdown(m, "role", "ra9_resid", "w", "RA9")
    tercile_breakdown(m, "ip_2026", "ra9_resid", "w", "RA9")
    tercile_breakdown(m, "ra9_2025", "ra9_resid", "w", "RA9")

    m.to_parquet("output/_scratch_ra9_error_analysis.parquet")
    d_te.to_parquet("output/_scratch_k_error_analysis.parquet")
    d_te2.to_parquet("output/_scratch_bb_error_analysis.parquet")
