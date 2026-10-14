"""
fl_bridge_recency_test.py -- the other-league bridges' BB% side uses a
TBF-weighted AVERAGE across every available season (fl_college_bridge.py's
`_latest_and_multi`, mirrored in fl_pioneer_bridge.py/fl_milb_bridge.py) --
weighted by sample size, but NOT by how long ago that season was. A 3-year-old
season currently counts exactly as much as last year's, TBF for TBF. This
tests whether adding a recency decay on top of the existing TBF-weight
improves each bridge's own real, validated fit against actual FL BB_rate
outcomes (the same real transition-pair samples fl_college_bridge.py,
fl_pioneer_bridge.py, fl_milb_bridge.py were each validated on -- not a
separate held-out split, same in-sample WLS-R2 comparison methodology
DECISION_LOG.md's "Multi-year college input" entry already used to validate
TBF-weighting itself over single-year BB%).

decay=1.0 reproduces each bridge's current (shipped) TBF-only average
exactly. Lower decay downweights older seasons more aggressively.

Usage:
    python fl_bridge_recency_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_college_bridge as bridge
import fl_pioneer_bridge as pbridge
import fl_milb_bridge as mbridge

DECAYS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]


def decayed_multi(years_df: pd.DataFrame, name_clean: str, season_col: str,
                   as_of_season: int, decay: float):
    """Same population as _latest_and_multi (every season up to and including
    as_of_season), but each season's TBF weight is ALSO multiplied by
    decay**(lag), lag = as_of_season_of_latest - that_season. decay=1.0 is
    the existing TBF-only average; decay=0.0 collapses to the single most
    recent season (same value k_pct already uses)."""
    hist = years_df[(years_df["name_clean"] == name_clean) & (years_df[season_col] <= as_of_season)]
    if hist.empty:
        return None
    latest_season = hist[season_col].max()
    lag = latest_season - hist[season_col]
    w = hist["tbf"] * (decay ** lag)
    if w.sum() <= 0:
        return None
    return float(np.average(hist["bb_pct"], weights=w))


def sweep_bridge(label: str, pair_df: pd.DataFrame, years_df: pd.DataFrame, season_col: str,
                  as_of_col: str, bb_rate_col: str) -> pd.DataFrame:
    """pair_df: real transition pairs (pitcher_id/name_clean, as_of season, real
    bb_rate outcome). Refits bb_pct_multi at each decay and reports the plain
    (unweighted) OLS R2/corr against the real outcome -- same methodology
    fl_college_bridge.fit_bridge already uses for its own reported numbers."""
    rows = []
    for decay in DECAYS:
        vals = []
        for _, r in pair_df.iterrows():
            v = decayed_multi(years_df, r["name_clean"], season_col, r[as_of_col], decay)
            vals.append(v)
        d = pair_df.copy()
        d["bb_pct_multi_decay"] = vals
        d = d.dropna(subset=["bb_pct_multi_decay", bb_rate_col])
        if len(d) < 15:
            rows.append({"bridge": label, "decay": decay, "n": len(d), "R2": np.nan, "corr": np.nan})
            continue
        m = smf.ols(f"{bb_rate_col} ~ bb_pct_multi_decay", data=d).fit()
        corr = d["bb_pct_multi_decay"].corr(d[bb_rate_col])
        rows.append({"bridge": label, "decay": decay, "n": len(d),
                      "R2": round(max(m.rsquared, 0.0), 4), "corr": round(corr, 4)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    # ---- COLLEGE ----
    print("\nLoading college bridge fitting sample (n=152 real transition pairs)...")
    d_college = pd.read_parquet(bridge.BRIDGE_PATH).drop_duplicates("pitcher_id").copy()
    college_years = bridge.load_all_college_years()
    college_pairs = d_college.rename(columns={"college_year": "as_of_season"})[
        ["pitcher_id", "name_clean", "as_of_season", "bb_rate"]]
    college_res = sweep_bridge("College", college_pairs, college_years, "season", "as_of_season", "bb_rate")

    # ---- PIONEER ----
    print("Fitting Pioneer transition pairs (n~68)...")
    pioneer_years = pbridge.load_all_pioneer_years()
    pioneer_d = pbridge.build_fl_transition_pairs(raw, pioneer_years)
    pioneer_d = pioneer_d.rename(columns={"latest_season": "as_of_season"})
    pioneer_d["as_of_season"] = pioneer_d["fl_year"] - 1  # as_of = the year BEFORE the FL debut, not latest matched season
    pioneer_pairs = pioneer_d[["pitcher_id", "fl_year", "as_of_season", "bb_rate"]].copy()
    pioneer_pairs["name_clean"] = pioneer_d["pitcher"].apply(pbridge._fl_name_clean)
    pioneer_res = sweep_bridge("Pioneer", pioneer_pairs, pioneer_years, "year", "as_of_season", "bb_rate")

    # ---- MILB ----
    print("Fitting MiLB transition pairs (n~194)...")
    milb_years = mbridge.load_all_milb_years()
    milb_d = mbridge.build_fl_transition_pairs(raw, milb_years)
    milb_d["as_of_season"] = milb_d["fl_year"] - 1
    milb_pairs = milb_d[["pitcher_id", "fl_year", "as_of_season", "bb_rate"]].copy()
    milb_pairs["name_clean"] = milb_d["pitcher"].apply(mbridge._fl_name_clean)
    milb_res = sweep_bridge("MiLB", milb_pairs, milb_years, "year", "as_of_season", "bb_rate")

    print("\n" + "=" * 70)
    print("Recency-decay sweep on each bridge's BB% multi-year average")
    print("(decay=1.0 = current shipped TBF-only average)")
    print("=" * 70)
    all_res = pd.concat([college_res, pioneer_res, milb_res], ignore_index=True)
    print(all_res.to_string(index=False))

    print("\n--- n_years>1 share (decay only matters for players with multiple seasons) ---")
    for label, pairs, years_df, season_col, as_of_col in [
        ("College", college_pairs, college_years, "season", "as_of_season"),
        ("Pioneer", pioneer_pairs, pioneer_years, "year", "as_of_season"),
        ("MiLB", milb_pairs, milb_years, "year", "as_of_season"),
    ]:
        n_multi = 0
        n_total = 0
        for _, r in pairs.iterrows():
            hist = years_df[(years_df["name_clean"] == r["name_clean"]) & (years_df[season_col] <= r[as_of_col])]
            if len(hist) == 0:
                continue
            n_total += 1
            if len(hist) > 1:
                n_multi += 1
        print(f"{label}: {n_multi}/{n_total} matched players have >1 trailing season")
