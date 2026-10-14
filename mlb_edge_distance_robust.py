"""
mlb_edge_distance_robust.py -- does winsorizing/trimming/median-ing the signed
edge-distance metric improve its stability (n0), without losing the BB_rate
signal that made the raw mean worth using in the first place?

The raw mean's n0=587 (worst of everything tested) is a direct symptom of the
distribution's heavy right tail (max values past 15-18 zone-widths -- wild
pitches/mistakes) dominating the average until enough pitches wash them out.
Tests:
  - raw mean
  - winsorized mean, GLOBAL 1/99 and 5/95 pct bounds PER FAMILY (clip each
    PITCH's value before averaging -- caps true outlier pitches, same
    convention as horiz_dist/vert_dist elsewhere in this project)
  - trimmed mean, 5% each tail WITHIN each pitcher-season (a different, group-
    relative notion of robustness: drop each pitcher's own most extreme
    pitches, not a global cap)
  - median

n0 via two methods where each applies:
  - mlb_marcel.fit_shrinkage_params's sigma_within2/sigma_between2 (the correct
    method-of-moments analog for a continuous mean-of-per-pitch-values metric
    -- valid for raw/winsorized since those ARE simple per-pitch means; not
    meaningful for trimmed-mean/median, which aren't simple per-pitch averages)
  - the empirical YoY reliability curve (valid for all four regardless of
    aggregator, since it just fits n/(n+n0) to observed correlation-by-bucket)

Usage:
    python mlb_edge_distance_robust.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_kbb_model import MIN_PA, PANEL_PATH
from mlb_signed_edge_distance import signed_edge_distance

pd.set_option("display.width", 160, "display.max_columns", 40)

FAMILIES = ["FB", "BRK", "OFF"]


def trimmed_mean(s: pd.Series, trim: float = 0.05) -> float:
    lo, hi = s.quantile([trim, 1 - trim])
    return s[(s >= lo) & (s <= hi)].mean()


def evaluate(raw, panel, agg_name, value_col, groupby_fn, shrinkage_col=None):
    print(f"\n{'='*70}\n{agg_name}\n{'='*70}")
    season = raw.groupby(["pitcher", "game_year", "pitch_family"])[value_col].agg(
        [("value", groupby_fn), ("n", "size")]).reset_index()
    season = season[season["n"] >= mf.MIN_N_FAMILY]
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

    n0_mom = np.nan
    if shrinkage_col is not None:
        sw2, sb2, gm = mm.fit_shrinkage_params(raw, shrinkage_col, group_cols=("pitcher", "game_year"),
                                                 min_n=mf.MIN_N_FAMILY)
        n0_mom = sw2 / sb2
        print(f"  sigma_within2={sw2:.4f}  sigma_between2={sb2:.6f}  n0 (method-of-moments) = {n0_mom:.1f}")

    print(f"  n eligible = {len(elig)}, n couplets = {len(cp)}")
    print(f"  corr(K_rate) = {ck:+.4f}   corr(BB_rate) = {cb:+.4f}")
    print(f"  YoY raw corr = {yoy_raw:.4f}   n0 (YoY curve fit) = {n0_yoy:.1f}")
    if len(curve):
        print(f"  curve:\n{curve.round(4).to_string(index=False)}")
    return {"corr_K": ck, "corr_BB": cb, "yoy_raw": yoy_raw, "n0_yoy": n0_yoy, "n0_mom": n0_mom}


if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)
    raw = raw[raw["pitch_family"].isin(FAMILIES) & raw["plate_x"].notna() & raw["plate_z"].notna()].copy()
    raw["edge_dist"] = signed_edge_distance(raw["plate_x"].values, raw["plate_z"].values)
    print(f"\nedge_dist skewness by family:\n{raw.groupby('pitch_family')['edge_dist'].skew().round(2)}")

    # global (per-family) winsorization bounds -- cap true outlier PITCHES, computed
    # once from the whole pooled population, not per pitcher-season
    bounds_1_99 = raw.groupby("pitch_family")["edge_dist"].quantile([.01, .99]).unstack()
    bounds_5_95 = raw.groupby("pitch_family")["edge_dist"].quantile([.05, .95]).unstack()
    print(f"\nglobal winsorization bounds (1/99 pct):\n{bounds_1_99.round(3)}")
    print(f"global winsorization bounds (5/95 pct):\n{bounds_5_95.round(3)}")

    lo99 = raw["pitch_family"].map(bounds_1_99[0.01])
    hi99 = raw["pitch_family"].map(bounds_1_99[0.99])
    raw["edge_dist_w1"] = raw["edge_dist"].clip(lower=lo99, upper=hi99)
    lo95 = raw["pitch_family"].map(bounds_5_95[0.05])
    hi95 = raw["pitch_family"].map(bounds_5_95[0.95])
    raw["edge_dist_w5"] = raw["edge_dist"].clip(lower=lo95, upper=hi95)

    panel = pd.read_parquet(PANEL_PATH)
    results = {}
    results["raw_mean"] = evaluate(raw, panel, "raw_mean", "edge_dist", "mean", shrinkage_col="edge_dist")
    results["winsor_1_99"] = evaluate(raw, panel, "winsor_1_99", "edge_dist_w1", "mean", shrinkage_col="edge_dist_w1")
    results["winsor_5_95"] = evaluate(raw, panel, "winsor_5_95", "edge_dist_w5", "mean", shrinkage_col="edge_dist_w5")
    results["trimmed_5pct"] = evaluate(raw, panel, "trimmed_5pct", "edge_dist", trimmed_mean)
    results["median"] = evaluate(raw, panel, "median", "edge_dist", "median")

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    summary = pd.DataFrame(results).T
    print(summary.round(4).to_string())
