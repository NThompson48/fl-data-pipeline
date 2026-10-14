"""
mlb_milb_bridge_analysis.py -- does minor-league K%/BB% actually predict the
FOLLOWING year's MLB K%/BB%, and does the relationship (translation factor)
differ by level (AAA vs. AA vs. A+ vs. A vs. Rookie)?

Bridge pairs: pitcher's MiLB season in year Y (any level) -> that SAME
pitcher's MLB season in year Y+1. Checked against a minimum sample on both
sides (MiLB BF>=100, MLB PA>=30 -- lower than the project's usual MIN_PA=50
since these are, by construction, players with limited MLB opportunity, the
exact population this would matter for).

The real test of value: for these THIN-MLB-HISTORY pitchers (which is exactly
who Marcel can't currently say anything useful about -- no/minimal trailing
MLB history means Marcel just falls back near the league average), does the
prior year's MiLB rate beat that naive league-average fallback?

Usage:
    python mlb_milb_bridge_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlb_marcel as mm

pd.set_option("display.width", 160, "display.max_columns", 40)

MIN_MILB_BF = 100
MIN_MLB_PA = 30
LEVEL_ORDER = ["AAA", "AA", "A+", "A", "ROK"]

if __name__ == "__main__":
    milb = pd.read_parquet("output/milb_pitching_2018_2025.parquet")
    mlb = pd.read_parquet("output/mlb_pitcher_season_features.parquet")[
        ["pitcher", "game_year", "PA", "K_rate", "BB_rate", "Marcel_K_rate", "Marcel_BB_rate"]]

    # for pitchers who appear at MULTIPLE levels in the same year, keep the level
    # with the most batters faced (their primary level that season)
    milb = milb.sort_values("battersFaced", ascending=False).drop_duplicates(
        subset=["pitcher", "season"], keep="first")
    milb = milb[milb["battersFaced"] >= MIN_MILB_BF]

    bridge = milb.merge(mlb.rename(columns={"game_year": "season_mlb"}),
                         left_on=["pitcher"], right_on=["pitcher"], how="inner")
    bridge = bridge[bridge["season_mlb"] == bridge["season"] + 1]
    bridge = bridge[bridge["PA"] >= MIN_MLB_PA]
    print(f"n bridge pairs (MiLB year Y -> MLB year Y+1) = {len(bridge)}")
    print(bridge["level"].value_counts())

    print("\n=== Overall: does MiLB rate predict next-year MLB rate? ===")
    print(f"corr(MiLB K_rate, MLB K_rate) = {bridge['K_rate_x'].corr(bridge['K_rate_y']):.4f}")
    print(f"corr(MiLB BB_rate, MLB BB_rate) = {bridge['BB_rate_x'].corr(bridge['BB_rate_y']):.4f}")
    print(f"(for comparison: MLB-to-MLB Marcel already gets corr(Marcel_K_rate, K_rate)~0.57 on the full population)")

    print("\n=== By level: correlation and translation slope ===")
    import statsmodels.formula.api as smf
    for lev in LEVEL_ORDER:
        sub = bridge[bridge["level"] == lev]
        if len(sub) < 20:
            print(f"  {lev}: n={len(sub)} -- too few, skipping")
            continue
        ck = sub["K_rate_x"].corr(sub["K_rate_y"])
        cb = sub["BB_rate_x"].corr(sub["BB_rate_y"])
        mk = smf.ols("K_rate_y ~ K_rate_x", data=sub).fit()
        mb = smf.ols("BB_rate_y ~ BB_rate_x", data=sub).fit()
        print(f"  {lev:4s} n={len(sub):4d}  corr(K)={ck:+.3f} slope={mk.params['K_rate_x']:.3f} "
              f"intercept={mk.params['Intercept']*100:.2f}pp  |  "
              f"corr(BB)={cb:+.3f} slope={mb.params['BB_rate_x']:.3f} intercept={mb.params['Intercept']*100:.2f}pp")
        print(f"       mean MiLB K%={sub['K_rate_x'].mean()*100:.2f}  mean MLB K%={sub['K_rate_y'].mean()*100:.2f}  "
              f"mean MiLB BB%={sub['BB_rate_x'].mean()*100:.2f}  mean MLB BB%={sub['BB_rate_y'].mean()*100:.2f}")

    print("\n=== The real test: does MiLB rate beat the current Marcel fallback for THIN-MLB-HISTORY pitchers? ===")
    # this population's Marcel_K_rate is almost always league-average-dominated since
    # they have little/no trailing MLB history -- check both against actual MLB K_rate
    thin = bridge.dropna(subset=["Marcel_K_rate", "Marcel_BB_rate"])
    print(f"n = {len(thin)}")
    n0_k, _ = mm.estimate_n0(thin["K_rate_y"], thin["PA"])
    n0_bb, _ = mm.estimate_n0(thin["BB_rate_y"], thin["PA"])
    w = mm.reliability_weight(thin["PA"], n0_k).values
    print(f"Marcel_K_rate alone:  R2_w={mm.weighted_r2(thin['K_rate_y'].values, thin['Marcel_K_rate'].values, w):.4f}  "
          f"RMSE_w={mm.weighted_rmse(thin['K_rate_y'].values, thin['Marcel_K_rate'].values, w)*100:.3f}pp")
    print(f"MiLB K_rate alone:    R2_w={mm.weighted_r2(thin['K_rate_y'].values, thin['K_rate_x'].values, w):.4f}  "
          f"RMSE_w={mm.weighted_rmse(thin['K_rate_y'].values, thin['K_rate_x'].values, w)*100:.3f}pp")
    m_combo = smf.wls("K_rate_y ~ Marcel_K_rate + K_rate_x", data=thin, weights=w).fit()
    pred_combo = m_combo.fittedvalues.values
    print(f"Marcel + MiLB combined: R2_w={mm.weighted_r2(thin['K_rate_y'].values, pred_combo, w):.4f}  "
          f"RMSE_w={mm.weighted_rmse(thin['K_rate_y'].values, pred_combo, w)*100:.3f}pp")
    print(f"  combo coefs: Marcel_K_rate={m_combo.params['Marcel_K_rate']:.3f} (p={m_combo.pvalues['Marcel_K_rate']:.4f}), "
          f"K_rate_x (MiLB)={m_combo.params['K_rate_x']:.3f} (p={m_combo.pvalues['K_rate_x']:.4f})")

    w_bb = mm.reliability_weight(thin["PA"], n0_bb).values
    print(f"\nMarcel_BB_rate alone: R2_w={mm.weighted_r2(thin['BB_rate_y'].values, thin['Marcel_BB_rate'].values, w_bb):.4f}  "
          f"RMSE_w={mm.weighted_rmse(thin['BB_rate_y'].values, thin['Marcel_BB_rate'].values, w_bb)*100:.3f}pp")
    print(f"MiLB BB_rate alone:   R2_w={mm.weighted_r2(thin['BB_rate_y'].values, thin['BB_rate_x'].values, w_bb):.4f}  "
          f"RMSE_w={mm.weighted_rmse(thin['BB_rate_y'].values, thin['BB_rate_x'].values, w_bb)*100:.3f}pp")
    m_combo_bb = smf.wls("BB_rate_y ~ Marcel_BB_rate + BB_rate_x", data=thin, weights=w_bb).fit()
    pred_combo_bb = m_combo_bb.fittedvalues.values
    print(f"Marcel + MiLB combined: R2_w={mm.weighted_r2(thin['BB_rate_y'].values, pred_combo_bb, w_bb):.4f}  "
          f"RMSE_w={mm.weighted_rmse(thin['BB_rate_y'].values, pred_combo_bb, w_bb)*100:.3f}pp")
    print(f"  combo coefs: Marcel_BB_rate={m_combo_bb.params['Marcel_BB_rate']:.3f} (p={m_combo_bb.pvalues['Marcel_BB_rate']:.4f}), "
          f"BB_rate_x (MiLB)={m_combo_bb.params['BB_rate_x']:.3f} (p={m_combo_bb.pvalues['BB_rate_x']:.4f})")
