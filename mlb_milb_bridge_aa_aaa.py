"""
mlb_milb_bridge_aa_aaa.py -- narrows the bridge to AA+AAA only (the levels
with enough same-pitcher bridge pairs to trust, per mlb_milb_bridge_analysis.py),
and adds a SAME-YEAR bridge: for pitchers promoted mid-season, their AA/AAA
stats for year Y are, by construction, the PRE-promotion portion of that year
(the MiLB stats endpoint aggregates only time spent at that level; the MLB
Statcast panel separately captures only their time in the majors) -- so
milb.season == mlb.game_year is already a clean "before call-up -> after
call-up" (rest-of-season) split, with no date-cutting needed.

Two bridges:
  1. Year-prior: AA/AAA year Y -> MLB year Y+1 (projects the FOLLOWING full season)
  2. Same-year:  AA/AAA year Y -> MLB year Y   (projects REST-OF-SEASON after callup)

For each: does AA+AAA rate (pooled) beat/complement the current Marcel
fallback for these thin-MLB-history pitchers?

Usage:
    python mlb_milb_bridge_aa_aaa.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mlb_marcel as mm

pd.set_option("display.width", 160, "display.max_columns", 40)

MIN_MILB_BF = 100
MIN_MLB_PA = 30
LEVELS = ["AAA", "AA"]


def run_bridge(milb: pd.DataFrame, mlb: pd.DataFrame, year_offset: int, label: str):
    print(f"\n{'='*70}\n{label}  (MiLB year Y -> MLB year Y+{year_offset})\n{'='*70}")
    bridge = milb.merge(mlb.rename(columns={"game_year": "season_mlb"}), on="pitcher", how="inner")
    bridge = bridge[bridge["season_mlb"] == bridge["season"] + year_offset]
    bridge = bridge[bridge["PA"] >= MIN_MLB_PA]
    print(f"n bridge pairs = {len(bridge)}  (AAA={sum(bridge.level=='AAA')}, AA={sum(bridge.level=='AA')})")
    if len(bridge) < 30:
        print("  too few pairs -- skipping")
        return None

    ck = bridge["K_rate_x"].corr(bridge["K_rate_y"])
    cb = bridge["BB_rate_x"].corr(bridge["BB_rate_y"])
    print(f"corr(MiLB K_rate, MLB K_rate) = {ck:.4f}    corr(MiLB BB_rate, MLB BB_rate) = {cb:.4f}")

    mk = smf.ols("K_rate_y ~ K_rate_x", data=bridge).fit()
    mb = smf.ols("BB_rate_y ~ BB_rate_x", data=bridge).fit()
    print(f"K%  translation: MLB = {mk.params['Intercept']*100:.2f}pp + {mk.params['K_rate_x']:.3f} x MiLB_K%")
    print(f"BB% translation: MLB = {mb.params['Intercept']*100:.2f}pp + {mb.params['BB_rate_x']:.3f} x MiLB_BB%")
    print(f"mean MiLB K%={bridge['K_rate_x'].mean()*100:.2f}  mean MLB K%={bridge['K_rate_y'].mean()*100:.2f}  "
          f"mean MiLB BB%={bridge['BB_rate_x'].mean()*100:.2f}  mean MLB BB%={bridge['BB_rate_y'].mean()*100:.2f}")

    thin = bridge.dropna(subset=["Marcel_K_rate", "Marcel_BB_rate"])
    if len(thin) < 30:
        print("  too few with a Marcel value -- skipping the head-to-head test")
        return bridge
    n0_k, _ = mm.estimate_n0(thin["K_rate_y"], thin["PA"])
    n0_bb, _ = mm.estimate_n0(thin["BB_rate_y"], thin["PA"])
    w_k = mm.reliability_weight(thin["PA"], n0_k).values
    w_bb = mm.reliability_weight(thin["PA"], n0_bb).values

    print(f"\nhead-to-head (n={len(thin)}):")
    r2_marcel_k = mm.weighted_r2(thin["K_rate_y"].values, thin["Marcel_K_rate"].values, w_k)
    r2_milb_k = mm.weighted_r2(thin["K_rate_y"].values, thin["K_rate_x"].values, w_k)
    m_combo_k = smf.wls("K_rate_y ~ Marcel_K_rate + K_rate_x", data=thin, weights=w_k).fit()
    r2_combo_k = mm.weighted_r2(thin["K_rate_y"].values, m_combo_k.fittedvalues.values, w_k)
    print(f"  K%:  Marcel alone R2_w={r2_marcel_k:.4f}  |  MiLB alone R2_w={r2_milb_k:.4f}  |  "
          f"combined R2_w={r2_combo_k:.4f}  (Marcel coef={m_combo_k.params['Marcel_K_rate']:.3f} "
          f"p={m_combo_k.pvalues['Marcel_K_rate']:.4f}, MiLB coef={m_combo_k.params['K_rate_x']:.3f} "
          f"p={m_combo_k.pvalues['K_rate_x']:.4f})")

    r2_marcel_bb = mm.weighted_r2(thin["BB_rate_y"].values, thin["Marcel_BB_rate"].values, w_bb)
    r2_milb_bb = mm.weighted_r2(thin["BB_rate_y"].values, thin["BB_rate_x"].values, w_bb)
    m_combo_bb = smf.wls("BB_rate_y ~ Marcel_BB_rate + BB_rate_x", data=thin, weights=w_bb).fit()
    r2_combo_bb = mm.weighted_r2(thin["BB_rate_y"].values, m_combo_bb.fittedvalues.values, w_bb)
    print(f"  BB%: Marcel alone R2_w={r2_marcel_bb:.4f}  |  MiLB alone R2_w={r2_milb_bb:.4f}  |  "
          f"combined R2_w={r2_combo_bb:.4f}  (Marcel coef={m_combo_bb.params['Marcel_BB_rate']:.3f} "
          f"p={m_combo_bb.pvalues['Marcel_BB_rate']:.4f}, MiLB coef={m_combo_bb.params['BB_rate_x']:.3f} "
          f"p={m_combo_bb.pvalues['BB_rate_x']:.4f})")
    return bridge


if __name__ == "__main__":
    milb = pd.read_parquet("output/milb_pitching_2018_2025.parquet")
    milb = milb.sort_values("battersFaced", ascending=False).drop_duplicates(
        subset=["pitcher", "season"], keep="first")
    milb = milb[(milb["battersFaced"] >= MIN_MILB_BF) & milb["level"].isin(LEVELS)]
    print(f"AA+AAA pitcher-seasons available: {len(milb)}")

    mlb = pd.read_parquet("output/mlb_pitcher_season_features.parquet")[
        ["pitcher", "game_year", "PA", "K_rate", "BB_rate", "Marcel_K_rate", "Marcel_BB_rate"]]

    bridge_prior = run_bridge(milb, mlb, year_offset=1, label="YEAR-PRIOR bridge")
    bridge_same = run_bridge(milb, mlb, year_offset=0, label="SAME-YEAR (rest-of-season) bridge")

    if bridge_prior is not None and bridge_same is not None:
        print(f"\n{'='*70}\nSame pitcher-season appearing in BOTH bridges?\n{'='*70}")
        overlap = set(zip(bridge_prior.pitcher, bridge_prior.season)) & set(zip(bridge_same.pitcher, bridge_same.season))
        print(f"n overlapping (pitcher, MiLB-season) pairs = {len(overlap)} "
              f"(a pitcher who got a partial-season callup AND is also someone's 'year prior' entry -- "
              f"shouldn't happen often, since year-prior requires NO MLB action in year Y)")
