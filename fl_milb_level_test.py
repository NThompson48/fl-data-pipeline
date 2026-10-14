"""
fl_milb_level_test.py -- the MiLB bridge currently pools every level a pitcher
appeared at within a season into ONE blended rate (sum SO/BB/TBF across
levels, no distinction) -- level never enters the bridge as a feature, filter,
or weight, despite a real level effect in the raw data (K_rate 0.2515 at A ->
0.2236 at AAA, BB_rate 0.1046 at A+ -> 0.1290 at ROK) and 32% of matched
pitcher-seasons spanning multiple levels.

Per the user's direction: keep pooling all levels from the same season
together (not separate per-level bridges), but weight each level's
contribution by its difficulty rather than TBF alone. Operationalized as
level_base**level_rank applied multiplicatively on top of each level-stint's
real SO/BB/TBF counts before summing -- same mechanism (and same sweep
structure) as the recency-decay weighting already validated for the college
bridge: level_base=1.0 reproduces the current flat (TBF-only) pooling
exactly; level_base>1 up-weights advanced levels (AAA/AA count more than
A/ROK); level_base<1 down-weights them.

LEVEL_RANK: ROK=0, A=1, A+=2, AA=3, AAA=4 (standard MiLB hierarchy).

Usage:
    python fl_milb_level_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_milb_bridge as mbridge

LEVEL_RANK = {"ROK": 0, "A": 1, "A+": 2, "AA": 3, "AAA": 4}
LEVEL_BASES = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0]


def load_milb_years_level_weighted(level_base: float, path: str = mbridge.MILB_PATH) -> pd.DataFrame:
    """Same shape as fl_milb_bridge.load_all_milb_years (one row per
    name_clean/season), but SO/BB/TBF are each multiplied by
    level_base**level_rank before summing across levels within a season --
    level_base=1.0 reproduces the original exactly."""
    df = pd.read_parquet(path).copy()
    df["name_clean"] = df["player_name"].apply(mbridge._clean_name)
    df["level_rank"] = df["level"].map(LEVEL_RANK)
    w = level_base ** df["level_rank"]
    df["w_tbf"] = df["battersFaced"] * w
    df["w_so"] = df["strikeOuts"] * w
    df["w_bb"] = df["baseOnBalls"] * w
    agg = df.groupby(["name_clean", "season"]).agg(
        tbf=("w_tbf", "sum"), so=("w_so", "sum"), bb=("w_bb", "sum"),
        player=("player_name", "first"),
    ).reset_index().rename(columns={"season": "year"})
    agg["k_pct"] = agg["so"] / agg["tbf"]
    agg["bb_pct"] = agg["bb"] / agg["tbf"]
    return agg[agg["tbf"] > 0]


def sweep(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level_base in LEVEL_BASES:
        milb_years = load_milb_years_level_weighted(level_base)
        d = mbridge.build_fl_transition_pairs(raw, milb_years)
        if len(d) < 15:
            rows.append({"level_base": level_base, "n": len(d), "K_R2": np.nan, "K_corr": np.nan,
                         "BB_R2": np.nan, "BB_corr": np.nan})
            continue
        k_model = smf.wls("k_rate ~ k_pct", data=d, weights=d["pa"]).fit()
        bb_model = smf.wls("bb_rate ~ bb_pct_multi", data=d, weights=d["pa"]).fit()
        rows.append({
            "level_base": level_base, "n": len(d),
            "K_R2": round(max(k_model.rsquared, 0), 4), "K_corr": round(d["k_pct"].corr(d["k_rate"]), 4),
            "BB_R2": round(max(bb_model.rsquared, 0), 4),
            "BB_corr": round(d["bb_pct_multi"].corr(d["bb_rate"]), 4),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print("\n" + "=" * 70)
    print(f"Level-difficulty-weight sweep: level_base in {LEVEL_BASES}")
    print("(1.0 = current flat/no-level-distinction pooling; >1 up-weights AAA/AA; <1 down-weights them)")
    print("=" * 70)
    res = sweep(raw)
    print(res.to_string(index=False))
