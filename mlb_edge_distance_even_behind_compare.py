"""
mlb_edge_distance_even_behind_compare.py -- follow-up to
mlb_edge_distance_by_leverage.py: how do the "even" and "behind" edge-distance
metrics relate to EACH OTHER (same pitcher, same season), and does combining
them add anything a single one doesn't already capture for BB_rate?

Usage:
    python mlb_edge_distance_even_behind_compare.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_kbb_model import MIN_PA, PANEL_PATH
from mlb_count_leverage_command import leverage_bucket
from mlb_signed_edge_distance import signed_edge_distance

pd.set_option("display.width", 160, "display.max_columns", 40)

FAMILIES = ["FB", "BRK", "OFF"]


def season_metric(raw: pd.DataFrame, leverage: str, min_n: int = 50) -> pd.DataFrame:
    pop = raw[raw["leverage"] == leverage]
    out = pop.groupby(["pitcher", "game_year"])["edge_dist"].agg(
        [(f"edge_dist_{leverage}", "mean"), (f"n_{leverage}", "size")]).reset_index()
    return out[out[f"n_{leverage}"] >= min_n]


if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)
    raw = raw[raw["pitch_family"].isin(FAMILIES) & raw["plate_x"].notna() & raw["plate_z"].notna()].copy()
    raw["edge_dist"] = signed_edge_distance(raw["plate_x"].values, raw["plate_z"].values)
    raw["leverage"] = leverage_bucket(raw["balls"].values, raw["strikes"].values)

    even = season_metric(raw, "even")
    behind = season_metric(raw, "behind")
    ahead = season_metric(raw, "ahead")

    panel = pd.read_parquet(PANEL_PATH)
    d = panel.merge(even, on=["pitcher", "game_year"], how="inner").merge(
        behind, on=["pitcher", "game_year"], how="inner").merge(
        ahead, on=["pitcher", "game_year"], how="left")
    elig = d[d["PA"] >= MIN_PA].copy()
    print(f"\nn eligible (both even AND behind qualify) = {len(elig)}")

    print("\n=== Concurrent cross-correlation between leverage-specific metrics ===")
    print(f"corr(edge_dist_even, edge_dist_behind) = {elig['edge_dist_even'].corr(elig['edge_dist_behind']):.4f}")
    if "edge_dist_ahead" in elig.columns:
        print(f"corr(edge_dist_even, edge_dist_ahead)  = {elig['edge_dist_even'].corr(elig['edge_dist_ahead']):.4f}")
        print(f"corr(edge_dist_behind, edge_dist_ahead) = {elig['edge_dist_behind'].corr(elig['edge_dist_ahead']):.4f}")

    print("\n=== On this common sample: standalone BB_rate correlations ===")
    print(f"corr(edge_dist_even, BB_rate)   = {elig['edge_dist_even'].corr(elig['BB_rate']):.4f}")
    print(f"corr(edge_dist_behind, BB_rate) = {elig['edge_dist_behind'].corr(elig['BB_rate']):.4f}")

    print("\n=== Joint regression: does behind subsume even, or do both contribute independently? ===")
    n0_bb, _ = mm.estimate_n0(elig["BB_rate"], elig["PA"])
    elig["wt"] = mm.reliability_weight(elig["PA"], n0_bb)
    m_even = smf.wls("I(BB_rate*100) ~ edge_dist_even", data=elig, weights=elig["wt"]).fit()
    m_behind = smf.wls("I(BB_rate*100) ~ edge_dist_behind", data=elig, weights=elig["wt"]).fit()
    m_both = smf.wls("I(BB_rate*100) ~ edge_dist_even + edge_dist_behind", data=elig, weights=elig["wt"]).fit()
    print(f"even alone:    R2={m_even.rsquared:.4f}  coef={m_even.params['edge_dist_even']:.3f} "
          f"(p={m_even.pvalues['edge_dist_even']:.4f})")
    print(f"behind alone:  R2={m_behind.rsquared:.4f}  coef={m_behind.params['edge_dist_behind']:.3f} "
          f"(p={m_behind.pvalues['edge_dist_behind']:.4f})")
    print(f"both together: R2={m_both.rsquared:.4f}")
    print(f"  even coef={m_both.params['edge_dist_even']:.3f} (p={m_both.pvalues['edge_dist_even']:.4f})")
    print(f"  behind coef={m_both.params['edge_dist_behind']:.3f} (p={m_both.pvalues['edge_dist_behind']:.4f})")

    print("\n=== Stability recap: even vs. behind (from mlb_edge_distance_by_leverage.py) ===")
    print("  even:   n0(YoY curve)=281.4  n0(method-of-moments)=58.4  divergence=4.8x  YoY raw corr=0.4611")
    print("  behind: n0(YoY curve)=349.5  n0(method-of-moments)=55.2  divergence=6.3x  YoY raw corr=0.3505")
    print("  -> behind is LESS stable than even on every measure (higher n0, lower YoY persistence),")
    print("     despite being the stronger BB_rate signal -- a real precision/reliability trade-off.")
