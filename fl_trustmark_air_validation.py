"""
fl_trustmark_air_validation.py -- does the Mississippi (TrustmarkPark) translation behave
the way it is built to?

The translation (fl_ra9_trustmark_projection.build_trustmark_adjustment) says: a pitcher's
home runs per batted ball at Trustmark are his usual rate x 0.624 (HR factor), non-HR air
balls fall in for hits 0.73 pp less often, and BOTH effects scale with his Air% (share of
batted balls at >= 10 degrees). This script checks, on every batted ball in the data
(visitors included, so the sample is far larger than Mississippi's own staff):

 1. the size of the park effect itself, all pitchers: HR per batted ball, HR per air ball,
    non-HR hit rate on air balls, wOBA on contact -- Trustmark vs everywhere else;
 2. the Air% interaction, OUT OF SAMPLE: pitchers grouped by their 2025 Air% (own history
    shrunk toward the mean); how do 2026 batted balls at Trustmark differ from elsewhere
    by Air% group, in the HR rate, hits on air balls, and wOBA on contact?
 3. the practical size: the spread of the Air%-driven shift across the roster vs. the
    shift's mean, i.e. how much of the Mississippi adjustment actually differs by pitcher.

Usage:  python fl_trustmark_air_validation.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRUST = "TrustmarkPark"


def load():
    b = pd.read_csv("output/batted_ball_event_residuals.csv").dropna(subset=["pitcher_id", "angle", "exit_speed"])
    b["pitcher_id"] = b["pitcher_id"].astype("int64").astype(str)
    b["venue"] = b["game_id"].str.split("-").str[1]
    b["is_T"] = b["venue"] == TRUST
    b["air"] = b["angle"] >= 10
    b["hr"] = b["play_result"].astype(str).str.lower().eq("home_run")
    b["hit_nonhr_air"] = b["air"] & b["is_hit"].astype(bool) & ~b["hr"]
    return b


def block(g, label):
    n = len(g)
    air = g[g["air"]]
    return {"group": label, "BBE": n, "HR/BBE": g["hr"].mean(), "HR/air ball": air["hr"].mean(),
            "non-HR hit/air ball": air["hit_nonhr_air"].mean(), "wOBA on contact": g["woba"].mean(), "air%": g["air"].mean()}


if __name__ == "__main__":
    b = load()
    print(f"batted balls: {len(b):,}; venues: {b['venue'].nunique()}; at {TRUST}: {int(b['is_T'].sum()):,}; "
          f"HR identified: {int(b['hr'].sum())}")
    assert b["is_T"].sum() > 1000, "TrustmarkPark venue token not found in game_id"

    print("\n1. THE PARK EFFECT, all pitchers (both seasons)")
    rows = [block(b[b["is_T"]], "Trustmark"), block(b[~b["is_T"]], "all other parks")]
    t = pd.DataFrame(rows).set_index("group")
    t.loc["Trustmark / other"] = t.loc["Trustmark"] / t.loc["all other parks"]
    print(t.round(4).to_string())
    # same split with year, to see stability
    for yr in (2025, 2026):
        y = b[b["year"] == yr]
        ra, rb = block(y[y["is_T"]], "T"), block(y[~y["is_T"]], "O")
        print(f"   {yr}: HR/air ball T {ra['HR/air ball']:.4f} vs other {rb['HR/air ball']:.4f} (ratio {ra['HR/air ball']/rb['HR/air ball']:.2f}); "
              f"non-HR hit/air ball {ra['non-HR hit/air ball']:.4f} vs {rb['non-HR hit/air ball']:.4f}; BBE at T = {ra['BBE']}")

    # runs-per-9 equivalent of the measured effect for a pitcher whose games are all at Trustmark
    ip_proxy = b.groupby(["pitcher_id", "year"]).size().mean()
    bip_per_9 = None
    print("\n   implied RUNS/9 saved by pitching ALL games at Trustmark (HR only, 1.4 runs per HR, using each other-park batted ball as the base):")
    other = b[~b["is_T"]]; T = b[b["is_T"]]
    d_hr = other["hr"].mean() - T["hr"].mean()
    print(f"   HR per batted ball {other['hr'].mean():.4f} -> {T['hr'].mean():.4f} (diff {d_hr:.4f}); at ~24.5 batted balls per 9 IP that is "
          f"{d_hr * 24.5 * 1.4:.2f} runs/9  [model: HR component of the full-park shift ~ -0.4]")

    print("\n2. THE AIR% INTERACTION, out of sample: 2025 Air% (own history, shrunk n0=150) -> 2026 batted balls")
    a25 = b[b["year"] == 2025].groupby("pitcher_id").agg(n=("air", "size"), air=("air", "mean")).reset_index()
    lg = np.average(a25["air"], weights=a25["n"])
    a25["air_s"] = (a25["n"] * a25["air"] + 150 * lg) / (a25["n"] + 150)
    b26 = b[b["year"] == 2026].merge(a25[["pitcher_id", "air_s"]], on="pitcher_id", how="inner")
    b26["tercile"] = pd.qcut(b26.drop_duplicates("pitcher_id").set_index("pitcher_id")["air_s"], 3, labels=["low Air%", "mid Air%", "high Air%"]).reindex(b26["pitcher_id"]).values
    rows = []
    for terc in ["low Air%", "mid Air%", "high Air%"]:
        g = b26[b26["tercile"] == terc]
        rt, ro = block(g[g["is_T"]], terc), block(g[~g["is_T"]], terc)
        rows.append({"2025 Air% group": terc, "pitchers": g["pitcher_id"].nunique(), "2025 Air% (mean)": g["air_s"].mean(),
                     "BBE at T / other": f"{rt['BBE']} / {ro['BBE']}",
                     "HR per BBE  T": rt["HR/BBE"], "other": ro["HR/BBE"], "T-other": rt["HR/BBE"] - ro["HR/BBE"],
                     "HR/air ratio T/other": rt["HR/air ball"] / ro["HR/air ball"],
                     "wOBAcon T-other": rt["wOBA on contact"] - ro["wOBA on contact"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("   (the model's prediction: the HR-per-ball EFFECT scales with Air%, so |T-other| should grow from low to high; "
          "the ratio should be about constant)")

    print("\n3. HOW MUCH OF THE MISSISSIPPI ADJUSTMENT DIFFERS BY PITCHER (the shipped roster)")
    r = pd.read_parquet("../projections-dashboard/data/fl_ra9_trustmark_projection.parquet")
    p = pd.read_parquet("../projections-dashboard/data/fl_2027_projections.parquet")[["pitcher_id", "PA"]]
    m = r.merge(p, on="pitcher_id"); ref = m[m["PA"] >= 30]
    sh = ref["trustmark_ra9_shift_half"]
    print(f"   half-season shift: mean {sh.mean():+.3f} runs/9, SD {sh.std():.3f}, 10th-90th pct {sh.quantile(.1):+.3f} .. {sh.quantile(.9):+.3f}; "
          f"SD of the projected RA9 itself {ref['proj_ra9_baseline'].std():.2f}")
    print(f"   so Air% moves a pitcher's Mississippi adjustment by about +/-{sh.std():.2f} runs/9 around {sh.mean():+.2f} "
          f"({sh.std()/ref['proj_ra9_baseline'].std():.0%} of the spread in projected RA9)")
