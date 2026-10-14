"""
fl_ros_live.py -- ROS snapshots for LIVE use (a season whose results are not yet known, e.g. 2027) with every coefficient fit on ALL completed seasons (2025 + 2026).

Differences from fl_ros_build.py (the 2026 backtest-honest snapshots, everything fit on 2025 only):
  * workload constants (k, c, inactive mean)       fit on 2025 + 2026 ROS rows
  * seasonal league-level correction (wOBA)        averaged over 2025 and 2026
  * pitcher K% / BB% regression constants          fit_constants(2026): all transitions through 2025 -> 2026
  * stuff / command coefficients                   STUFF_B["pooled"] (.958 / .0345)
  * hitter BB% in-season weight                    PROD["nu_bbr"] (.4, fit on 2025 rows)
  * pitchers: the PUBLISHED preseason K% / BB% is the prior (phi = 1), stuff / command terms x .5   (fl_ros_published_prior.py; suggestive: P(better) ~.9, not significant)
Nothing here can be validated out of sample until the season has been played; run with --year 2026 to check that the pooled build reproduces the clean 2026 snapshot closely (in-sample).

Requires the dated PA / pitch tables (fl_hitter_ros.PA_PATH / PITCH_PATH, fl_pitcher_ros) to contain the target season; the script stops with a clear message otherwise.

Usage:  python fl_ros_live.py --year 2027 --asof 2027-06-15 [--asof ...]
        python fl_ros_live.py --year 2026 --asof 2026-08-31 --check      (compare against output/ros_*_2026.parquet)
"""
from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np
import pandas as pd

import fl_hitter_ros as hr
import fl_innings_projection as ip
import fl_pitcher_ros as pr
import fl_pitcher_ros_stuff as pst
import fl_ros_production as rp
import fl_ros_workload as wl

COMPLETED = (2025, 2026)


def load_published(year):
    """Published preseason K% / BB% (name_clean-indexed) for the year being forecast.  2027: the production projection table; 2026 (check mode): the 2026-style projection built from full-2025 features."""
    if year == 2026:
        d = pd.read_parquet("output/pitcher_pre2026_published.parquet").dropna(subset=["name_clean"])
    else:
        d = pd.read_parquet("output/fl_2027_projections.parquet").rename(columns={"proj_K_rate": "pre_k", "proj_BB_rate": "pre_bb"}); d["name_clean"] = d["pitcher"].apply(ip.lastfirst_to_clean)
    d = d.dropna(subset=["name_clean"]).drop_duplicates("name_clean").set_index("name_clean")[["pre_k", "pre_bb"]]
    return d.assign(pre_k=d["pre_k"].clip(0.03, 0.6), pre_bb=d["pre_bb"].clip(0.01, 0.4))      # same bounds the ROS rates are clipped to (the published K% has a few out-of-range extrapolations)


def build(year, asofs, check=False):
    T_h = hr.Tables(); T_p = pr.Tables()
    have = set(T_h.pa["year"].unique())
    if year not in have:
        sys.exit(f"No {year} plate appearances in the dated PA table (have {sorted(have)}). Refresh the PA / pitch tables first.")
    ctx = rp.PitcherContext(T_p)
    D = wl.collect(T_h, T_p)
    wf_h = rp.workload_fit(D, "hit", COMPLETED); wf_p = rp.workload_fit(D, "pit", COMPLETED)
    consts = pr.fit_constants(max(COMPLETED)); n0s = pst.reliability(ctx.raw); B = rp.STUFF_B["pooled"]
    print(f"workload fits (pooled {COMPLETED}): hitters {wf_h}  pitchers {wf_p}\npitcher constants: " + ", ".join(f"{s}: delta {c['delta']} n0 {c['n0']:.0f}" for s, c in consts.items()) + f"\nstuff coefficients (K%, BB%): {B}")
    pub = load_published(year)
    hd = rp.display_names(T_h.pa, "batter"); pdn = rp.display_names(T_p.pa, "pitcher"); H, P = [], []
    for ds in asofs:
        asof = pd.Timestamp(ds)
        h = rp.hitter_snapshot(T_h, year, asof, wf_h, COMPLETED); h["asof"] = ds; h["name"] = hd.reindex(h.index).to_numpy(); H.append(h.reset_index())
        p = rp.pitcher_snapshot(T_p, ctx, year, asof, consts, n0s, B, wf_p, pub=pub, term_scale=0.5 if pub is not None else 1.0); p["asof"] = ds; p["name"] = pdn.reindex(p.index).to_numpy(); P.append(p.reset_index())
        print(f"snapshot built: {ds} {len(h)} hitters, {len(p)} pitchers", flush=True)
    H, P = pd.concat(H, ignore_index=True), pd.concat(P, ignore_index=True)
    if check:
        for tag, new, path, cols in (("hitters", H, f"output/ros_hitters_{year}.parquet", ["ros_woba", "ros_k", "ros_bb", "ros_pa", "ros_war"]),
                                     ("pitchers", P, f"output/ros_pitchers_{year}.parquet", ["ros_k", "ros_bb", "ros_ra9", "ros_ip", "ros_war"])):
            old = pd.read_parquet(path); keys = ["name_clean", "asof"]; m = old[keys + cols].merge(new[keys + cols], on=keys, suffixes=("_clean", "_pooled"))
            print(f"\n{tag}: pooled-coefficient build vs the clean {year} snapshot ({len(m)} matched rows)")
            for c in cols:
                d = m[f"{c}_pooled"] - m[f"{c}_clean"]; print(f"   {c:9s} mean diff {d.mean():+.5f}  mean |diff| {d.abs().mean():.5f}  max |diff| {d.abs().max():.5f}  corr {m[f'{c}_pooled'].corr(m[f'{c}_clean']):.4f}")
        return
    hp, pp = f"output/ros_hitters_live_{year}.parquet", f"output/ros_pitchers_live_{year}.parquet"
    H.to_parquet(hp, index=False); P.to_parquet(pp, index=False); print(f"saved {hp}, {pp} (not copied to the dashboard: the ROS view reads the 2026 files; point it at these when 2027 starts)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, required=True); ap.add_argument("--asof", action="append", required=True); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); build(a.year, a.asof, a.check)
