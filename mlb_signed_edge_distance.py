"""
mlb_signed_edge_distance.py -- a continuous alternative to the heart/shadow/
chase/waste TIERS: signed distance from the TRUE strike-zone edge, in
zone-width units.

The Attack Zone boundaries (mlb_attack_zone_command.py) scale out perfectly at
ring = 1.0 (heart edge), 2.0 (shadow outer / chase inner), 3.0 (chase outer /
waste inner), using the Chebyshev ("ring") distance:
    ring = max(|plate_x| / 0.558, |plate_z - 2.5| / 0.6665)
The actual physical rulebook strike-zone edge (x halfwidth ~0.83ft, z half-
height ~1.0ft off the fixed 1.5-3.5 reference) sits exactly at ring = 1.5 --
the midpoint of the Shadow band, which is WHY it's called "Shadow": it
straddles the true edge symmetrically. So:
    signed_edge_dist = ring - 1.5
is a continuous, zone-calibrated command metric: 0 = right on the true edge,
negative = inside the zone (toward the heart), positive = outside (toward
chase/waste) -- in the same normalized units as the tier system, not raw feet.

By pitch type (family): distribution of signed_edge_dist, and how whiff
probability (swing-conditioned) varies continuously with it -- does the
relationship actually peak somewhere, or keep climbing to the edge of the
data the way the discrete-zone/interaction analyses suggested?

Usage:
    python mlb_signed_edge_distance.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_kbb_model import MIN_PA, PANEL_PATH
from mlb_count_leverage_command import leverage_bucket

pd.set_option("display.width", 160, "display.max_columns", 40)

FAMILIES = ["FB", "BRK", "OFF"]
HALF_WIDTH = 0.558   # heart's horizontal half-width (ring=1.0 reference)
HALF_HEIGHT = 0.6665  # heart's vertical half-height (ring=1.0 reference)
Z_CENTER = 2.5


def signed_edge_distance(plate_x, plate_z) -> np.ndarray:
    plate_x = np.asarray(plate_x, dtype=float)
    plate_z = np.asarray(plate_z, dtype=float)
    ring = np.maximum(np.abs(plate_x) / HALF_WIDTH, np.abs(plate_z - Z_CENTER) / HALF_HEIGHT)
    return ring - 1.5


if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)
    raw = raw[raw["pitch_family"].isin(FAMILIES) & raw["plate_x"].notna() & raw["plate_z"].notna()].copy()
    raw["edge_dist"] = signed_edge_distance(raw["plate_x"].values, raw["plate_z"].values)
    raw["leverage"] = leverage_bucket(raw["balls"].values, raw["strikes"].values)

    print("\n=== Distribution of signed edge distance, by pitch family ===")
    print(raw.groupby("pitch_family")["edge_dist"].describe().round(3).to_string())

    print("\n=== Whiff probability (swing-conditioned) by edge-distance decile, by family ===")
    sw = raw[raw["is_swing"]].copy()
    for fam in FAMILIES:
        sub = sw[sw["pitch_family"] == fam].copy()
        sub["decile"] = pd.qcut(sub["edge_dist"], 10, duplicates="drop")
        tab = sub.groupby("decile", observed=True).agg(
            n=("is_whiff", "size"), mean_edge_dist=("edge_dist", "mean"), whiff_rate=("is_whiff", "mean"))
        print(f"\n  {fam}:")
        print(tab.round(4).to_string())

    print("\n=== Mean edge distance by family x count-leverage (does aggression track leverage?) ===")
    print(raw.groupby(["pitch_family", "leverage"])["edge_dist"].mean().round(4).to_string())

    # ---- pitcher-season aggregation + the standard battery ---- #
    season_edge = raw.groupby(["pitcher", "game_year", "pitch_family"]).agg(
        n=("edge_dist", "size"), mean_edge_dist=("edge_dist", "mean")).reset_index()
    season_edge = season_edge[season_edge["n"] >= mf.MIN_N_FAMILY]
    wide = season_edge.pivot(index=["pitcher", "game_year"], columns="pitch_family",
                              values=["mean_edge_dist", "n"])
    wide.columns = [f"{a}_{b.lower()}" for a, b in wide.columns]
    wide = wide.reset_index()
    edge_cols = [f"mean_edge_dist_{f.lower()}" for f in FAMILIES if f"mean_edge_dist_{f.lower()}" in wide.columns]
    wide["edge_dist_overall"] = wide[edge_cols].mean(axis=1, skipna=True)
    n_cols_e = [f"n_{f.lower()}" for f in FAMILIES if f"n_{f.lower()}" in wide.columns]
    wide["n_overall"] = wide[n_cols_e].fillna(0).sum(axis=1)

    panel = pd.read_parquet(PANEL_PATH)
    d = panel.merge(wide, on=["pitcher", "game_year"], how="inner")
    elig = d[(d["PA"] >= MIN_PA) & d["edge_dist_overall"].notna()]
    print(f"\nn eligible pitcher-seasons = {len(elig)}")

    print("\n=== Relationship with K_rate and BB_rate (concurrent) ===")
    for col in edge_cols + ["edge_dist_overall"]:
        ck = elig[col].corr(elig["K_rate"])
        cb = elig[col].corr(elig["BB_rate"])
        print(f"  {col:26s}  corr(K_rate)={ck:+.4f}   corr(BB_rate)={cb:+.4f}")

    print("\n=== Stickiness: year-over-year correlation ===")
    b = elig[["pitcher", "game_year", "edge_dist_overall", "n_overall"]].copy()
    b["game_year"] -= 1
    cp = elig.merge(b, on=["pitcher", "game_year"], suffixes=("1", "2"))
    print(f"  n couplets = {len(cp)}")
    print(f"  raw corr(edge_dist_overall_Y, _Y+1) = {cp['edge_dist_overall1'].corr(cp['edge_dist_overall2']):.4f}")

    print("\n=== Stability: n0 (via empirical reliability curve -- this is a continuous, not a rate,"
          " so the method-of-moments estimator's p(1-p) sampling model doesn't directly apply) ===")
    curve = mm.empirical_reliability_curve(cp, "edge_dist_overall1", "edge_dist_overall2",
                                             "n_overall1", "n_overall2")
    if len(curve):
        print(curve.round(4).to_string(index=False))
        n0_yoy = mm.fit_n0_from_curve(curve)
        print(f"  n0 (fit to YoY curve) = {n0_yoy:.1f}")
    else:
        print("  not enough bucket coverage for a curve -- reporting raw split-half only")
