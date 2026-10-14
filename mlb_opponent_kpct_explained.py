"""
mlb_opponent_kpct_explained.py -- how much of a pitcher-season's K_rate is just
who they faced, not their own skill?

For each pitcher-season, computes an "expected K_rate" purely from opponent
quality: the PA-weighted average of each batter's OWN K% that season, using a
LEAVE-ONE-OUT batter K% (that batter's K% EXCLUDING their PA against this
specific pitcher) to avoid the mechanical leakage of a batter's own rate
already containing the very outcome we're trying to explain. Then checks how
much of the pitcher's actual K_rate that opponent-quality index explains.

Usage:
    python mlb_opponent_kpct_explained.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_kbb_model import MIN_PA

pd.set_option("display.width", 160, "display.max_columns", 40)

if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)
    term = raw[raw["is_terminal_pitch"]].copy()
    term["is_k"] = term["events"].astype(str).isin(mf.K_EVENTS)

    print("Batter-season totals...")
    batter_season = term.groupby(["batter", "game_year"]).agg(
        PA_bat=("is_k", "size"), K_bat=("is_k", "sum")).reset_index()

    print("Pitcher-batter-season matchup totals...")
    matchup = term.groupby(["pitcher", "batter", "game_year"]).agg(
        PA_mu=("is_k", "size"), K_mu=("is_k", "sum")).reset_index()

    matchup = matchup.merge(batter_season, on=["batter", "game_year"], how="left")
    # leave-one-out: this batter's K rate EXCLUDING PA against this pitcher
    matchup["PA_loo"] = matchup["PA_bat"] - matchup["PA_mu"]
    matchup["K_loo"] = matchup["K_bat"] - matchup["K_mu"]
    matchup = matchup[matchup["PA_loo"] >= 10]  # need enough OTHER PA to trust the batter's own rate
    matchup["K_rate_loo"] = matchup["K_loo"] / matchup["PA_loo"]

    print("Aggregating to pitcher-season opponent-quality index...")
    pitcher_opp = matchup.groupby(["pitcher", "game_year"]).apply(
        lambda g: pd.Series({
            "expected_K_rate": np.average(g["K_rate_loo"], weights=g["PA_mu"]),
            "PA_covered": g["PA_mu"].sum(),
        })
    ).reset_index()

    season = mf.season_rates(raw)
    d = season.merge(pitcher_opp, on=["pitcher", "game_year"], how="inner")
    d["coverage"] = d["PA_covered"] / d["PA"]
    print(f"\nmedian PA coverage (batters with enough OTHER PA to get a LOO rate): {d['coverage'].median():.3f}")
    d = d[d["PA"] >= MIN_PA]
    print(f"n pitcher-seasons = {len(d)}")

    n0_k, _ = mm.estimate_n0(d["K_rate"], d["PA"])
    w = mm.reliability_weight(d["PA"], n0_k)

    corr = d["expected_K_rate"].corr(d["K_rate"])
    r2_unweighted = corr ** 2
    r2_w = mm.weighted_r2(d["K_rate"].values, d["expected_K_rate"].values, w.values)
    rmse_w = mm.weighted_rmse(d["K_rate"].values, d["expected_K_rate"].values, w.values)

    import statsmodels.formula.api as smf
    m = smf.wls("I(K_rate*100) ~ I(expected_K_rate*100)", data=d, weights=w).fit()

    print(f"\ncorr(pitcher K_rate, opponent-quality-implied K_rate) = {corr:.4f}")
    print(f"R2 (corr^2, unweighted) = {r2_unweighted:.4f}")
    print(f"R2 (using opponent-implied K_rate AS THE PREDICTION, reliability-weighted) = {r2_w:.4f}")
    print(f"RMSE (opponent-implied K_rate as prediction, reliability-weighted) = {rmse_w*100:.3f}pp")
    print(f"\nWLS: K_rate ~ expected_K_rate\n{m.summary2().tables[1].round(4)}")
    print(f"\nspread comparison: SD(actual K_rate)={d['K_rate'].std()*100:.2f}pp, "
          f"SD(expected/opponent-implied K_rate)={d['expected_K_rate'].std()*100:.2f}pp "
          f"(ratio={d['K_rate'].std()/d['expected_K_rate'].std():.2f}x)")

    # does adding it to the ALREADY-VALIDATED model add anything beyond Marcel+shape?
    panel = pd.read_parquet("output/mlb_pitcher_season_features.parquet")
    dd = panel.merge(pitcher_opp, on=["pitcher", "game_year"], how="left")
    elig = dd[dd["PA"] >= MIN_PA].dropna(subset=["Marcel_K_rate", "expected_K_rate"])
    print(f"\ncorr(Marcel_K_rate, expected_K_rate/opponent quality) = "
          f"{elig['Marcel_K_rate'].corr(elig['expected_K_rate']):.4f}  "
          f"(does opponent quality track a pitcher's own established talent, e.g. via role/rotation slot?)")
    resid_marcel = elig["K_rate"] - elig["Marcel_K_rate"]
    print(f"corr(residual after Marcel alone, expected_K_rate) = "
          f"{resid_marcel.corr(elig['expected_K_rate']):.4f}  "
          f"(does opponent quality explain anything Marcel doesn't already know?)")
