"""
fl_milb_level_translate_test.py -- follow-up to fl_milb_level_test.py, which
found simple difficulty-WEIGHTING of pooled counts doesn't meaningfully beat
flat pooling. This tests a different mechanism: TRANSLATE each level-stint's
rate to an AAA-equivalent (MLE-style) before pooling, using within-player
fixed-effects offsets (fit on 22,384 multi-level stints across 10,385
player-seasons, netting out selection bias -- a pitcher who gets promoted is
better than one who doesn't, so a naive cross-sectional level average
conflates level difficulty with who plays where; the within-player estimate
doesn't).

K% offset vs AAA: AA +0.0169, A+ +0.0178, A +0.0231, ROK +0.0418
BB% offset vs AAA: ROK -0.0225, A -0.0091, A+ -0.0108, AA -0.0114

translated_rate = raw_rate - offset[level], then pool (TBF-weighted) across
levels within a season as before -- same "pool together, not separate
bridges" structure the user asked for, just translating each level's
contribution onto a common scale first instead of only reweighting it.

Usage:
    python fl_milb_level_translate_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_milb_bridge as mbridge

K_OFFSET = {"AAA": 0.0, "AA": 0.016903, "A+": 0.017770, "A": 0.023148, "ROK": 0.041838}
BB_OFFSET = {"AAA": 0.0, "AA": -0.011424, "A+": -0.010819, "A": -0.009114, "ROK": -0.022484}


def load_milb_years_translated(path: str = mbridge.MILB_PATH) -> pd.DataFrame:
    """Same shape as load_all_milb_years, but each level-stint's SO/BB are
    rescaled to its AAA-equivalent rate (raw_rate - offset[level]) before
    being turned back into counts and pooled (TBF-weighted) across levels
    within a season."""
    df = pd.read_parquet(path).copy()
    df["name_clean"] = df["player_name"].apply(mbridge._clean_name)
    k_raw = df["strikeOuts"] / df["battersFaced"]
    bb_raw = df["baseOnBalls"] / df["battersFaced"]
    k_adj = (k_raw - df["level"].map(K_OFFSET)).clip(0, 1)
    bb_adj = (bb_raw - df["level"].map(BB_OFFSET)).clip(0, 1)
    df["so_adj"] = k_adj * df["battersFaced"]
    df["bb_adj"] = bb_adj * df["battersFaced"]
    agg = df.groupby(["name_clean", "season"]).agg(
        tbf=("battersFaced", "sum"), so=("so_adj", "sum"), bb=("bb_adj", "sum"),
        player=("player_name", "first"),
    ).reset_index().rename(columns={"season": "year"})
    agg["k_pct"] = agg["so"] / agg["tbf"]
    agg["bb_pct"] = agg["bb"] / agg["tbf"]
    return agg[agg["tbf"] > 0]


def fit_and_report(raw: pd.DataFrame, milb_years: pd.DataFrame, label: str) -> dict:
    d = mbridge.build_fl_transition_pairs(raw, milb_years)
    k_model = smf.wls("k_rate ~ k_pct", data=d, weights=d["pa"]).fit()
    bb_model = smf.wls("bb_rate ~ bb_pct_multi", data=d, weights=d["pa"]).fit()
    return {"label": label, "n": len(d),
            "K_R2": round(max(k_model.rsquared, 0), 4), "K_corr": round(d["k_pct"].corr(d["k_rate"]), 4),
            "BB_R2": round(max(bb_model.rsquared, 0), 4),
            "BB_corr": round(d["bb_pct_multi"].corr(d["bb_rate"]), 4)}


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print("\nBaseline (current production, no level adjustment)...")
    baseline_years = mbridge.load_all_milb_years()
    baseline = fit_and_report(raw, baseline_years, "Baseline (flat, no level adj)")

    print("Level-translated (AAA-equivalent, within-player offsets)...")
    translated_years = load_milb_years_translated()
    translated = fit_and_report(raw, translated_years, "Level-translated (AAA-equivalent)")

    print("\n" + "=" * 70)
    print("MiLB bridge: flat pooling vs. level-translated pooling")
    print("=" * 70)
    print(pd.DataFrame([baseline, translated]).to_string(index=False))
