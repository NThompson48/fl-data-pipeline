"""
mlb_edge_distance_by_leverage.py -- split the signed edge-distance metric by
count-leverage bucket (ahead/even/behind) instead of pooling all counts into
one pitcher-season mean, per the diagnosis in mlb_edge_distance_robust.py: the
pooled metric's instability (7x divergence between its two n0 estimators) came
from mixing genuinely different situational targets, not from outlier
contamination -- robustifying the AGGREGATION didn't help, so this tests
whether separating the SITUATIONS does.

Builds mean_edge_dist per (pitcher, game_year, family, leverage), blends
across families within each leverage bucket, and reruns the full battery
(K_rate/BB_rate correlation, stickiness, n0 via both estimators) per leverage
bucket, compared against the pooled version's numbers:
    pooled: corr_K=+0.025, corr_BB=+0.307, YoY=0.520, n0_yoy=587, n0_mom=85

Usage:
    python mlb_edge_distance_by_leverage.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_kbb_model import MIN_PA, PANEL_PATH
from mlb_count_leverage_command import leverage_bucket
from mlb_signed_edge_distance import signed_edge_distance

pd.set_option("display.width", 160, "display.max_columns", 40)

FAMILIES = ["FB", "BRK", "OFF"]
LEVERAGES = ["ahead", "even", "behind"]
MIN_N = 20  # lower than MIN_N_FAMILY=30 since this is a finer (family x leverage) split


def evaluate_leverage(raw, panel, leverage: str):
    print(f"\n{'='*70}\nLEVERAGE = {leverage}\n{'='*70}")
    pop = raw[raw["leverage"] == leverage]
    season = pop.groupby(["pitcher", "game_year", "pitch_family"])["edge_dist"].agg(
        [("value", "mean"), ("n", "size")]).reset_index()
    season = season[season["n"] >= MIN_N]
    wide = season.pivot(index=["pitcher", "game_year"], columns="pitch_family", values=["value", "n"])
    wide.columns = [f"{a}_{b.lower()}" for a, b in wide.columns]
    wide = wide.reset_index()
    val_cols = [f"value_{f.lower()}" for f in FAMILIES if f"value_{f.lower()}" in wide.columns]
    wide["overall"] = wide[val_cols].mean(axis=1, skipna=True)
    n_cols = [f"n_{f.lower()}" for f in FAMILIES if f"n_{f.lower()}" in wide.columns]
    wide["n_overall"] = wide[n_cols].fillna(0).sum(axis=1)

    d = panel.merge(wide, on=["pitcher", "game_year"], how="inner")
    elig = d[(d["PA"] >= MIN_PA) & d["overall"].notna()].copy()

    ck = elig["overall"].corr(elig["K_rate"])
    cb = elig["overall"].corr(elig["BB_rate"])

    b = elig[["pitcher", "game_year", "overall", "n_overall"]].copy()
    b["game_year"] -= 1
    cp = elig.merge(b, on=["pitcher", "game_year"], suffixes=("1", "2"))
    yoy_raw = cp["overall1"].corr(cp["overall2"])
    curve = mm.empirical_reliability_curve(cp, "overall1", "overall2", "n_overall1", "n_overall2")
    n0_yoy = mm.fit_n0_from_curve(curve) if len(curve) >= 4 else np.nan

    sw2, sb2, gm = mm.fit_shrinkage_params(pop, "edge_dist", group_cols=("pitcher", "game_year"), min_n=MIN_N)
    n0_mom = sw2 / sb2

    print(f"  n eligible = {len(elig)}, n couplets = {len(cp)}")
    print(f"  mean edge_dist (pooled families) = {elig['overall'].mean():.4f}")
    print(f"  corr(K_rate) = {ck:+.4f}   corr(BB_rate) = {cb:+.4f}")
    print(f"  YoY raw corr = {yoy_raw:.4f}   n0 (YoY curve) = {n0_yoy:.1f}   n0 (method-of-moments) = {n0_mom:.1f}")
    if len(curve):
        print(f"  curve:\n{curve.round(4).to_string(index=False)}")
    return {"corr_K": ck, "corr_BB": cb, "yoy_raw": yoy_raw, "n0_yoy": n0_yoy, "n0_mom": n0_mom, "n_elig": len(elig)}


if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)
    raw = raw[raw["pitch_family"].isin(FAMILIES) & raw["plate_x"].notna() & raw["plate_z"].notna()].copy()
    raw["edge_dist"] = signed_edge_distance(raw["plate_x"].values, raw["plate_z"].values)
    raw["leverage"] = leverage_bucket(raw["balls"].values, raw["strikes"].values)

    panel = pd.read_parquet(PANEL_PATH)
    results = {}
    for lev in LEVERAGES:
        results[lev] = evaluate_leverage(raw, panel, lev)

    print(f"\n\n{'='*70}\nSUMMARY (pooled/no-leverage-split version: corr_K=+0.0252, corr_BB=+0.3067, "
          f"yoy_raw=0.5195, n0_yoy=587.4, n0_mom=84.6)\n{'='*70}")
    print(pd.DataFrame(results).T.round(4).to_string())

    # ---- does the BEHIND-specific metric add anything to the actual BB% model,
    # beyond horiz_dist/vert_dist and the pooled edge_dist? ---- #
    print("\n\n=== Incremental check: does edge_dist_behind help BB_rate beyond what's already known? ===")
    behind = raw[raw["leverage"] == "behind"]
    season_b = behind.groupby(["pitcher", "game_year"])["edge_dist"].agg(
        [("edge_dist_behind", "mean"), ("n_behind", "size")]).reset_index()
    season_b = season_b[season_b["n_behind"] >= 50]
    d = panel.merge(season_b, on=["pitcher", "game_year"], how="inner")
    elig = d[d["PA"] >= MIN_PA]
    print(f"n = {len(elig)}")
    print(f"corr(edge_dist_behind, BB_rate) = {elig['edge_dist_behind'].corr(elig['BB_rate']):.4f}")
    print(f"corr(edge_dist_behind, horiz_dist) = {elig['edge_dist_behind'].corr(elig['horiz_dist']):.4f}")
    print(f"corr(edge_dist_behind, vert_dist) = {elig['edge_dist_behind'].corr(elig['vert_dist']):.4f}")
    resid = elig["BB_rate"] - 0  # placeholder for a simple partial check against horiz+vert_dist
    import statsmodels.formula.api as smf
    base_m = smf.ols("BB_rate ~ horiz_dist + vert_dist", data=elig).fit()
    resid = elig["BB_rate"] - base_m.fittedvalues
    print(f"corr(edge_dist_behind, BB_rate residual after horiz_dist+vert_dist) = {elig['edge_dist_behind'].corr(resid):.4f}")
