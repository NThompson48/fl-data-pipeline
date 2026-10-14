"""
fl_hitter_ifh_test.py -- is infield-hit rate a usable sprint-speed signal for hitters?

Definition: an infield hit = a SINGLE whose ball was fielded by an infielder / pitcher / catcher (fielded_by_pos in 1B, 2B, 3B, SS, P, C) -- or, where the fielder is not recorded,
a single that landed within 110 ft (bunt singles included).  Ground ball = launch angle < 10 degrees.  Because a slow or weakly hit grounder is more likely to be beaten out whoever
runs it, the speed signal is the infield-hit RESIDUAL: actual minus expected infield hits given the ball (exit velocity, launch angle, spray direction, batter side), the expectation from a
league-wide gradient-boosted model scored out-of-fold with hitters grouped.

A  Definition and base rates (by handedness).
B  Reliability: split-half (random halves of a hitter-season's ground balls) and 2025 -> 2026 correlation, for the raw rate and the residual.
C  Validity against independent speed indicators available for 2026 (published stolen bases per PA, grounded-into-double-play over expected, triples per PA).
D  Does it add to next-year wOBA / BABIP beyond the current model?  Year to year (2025 -> 2026) and a within-season PA split.

Usage:  python fl_hitter_ifh_test.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sqlalchemy import text

import fl_innings_projection as ip
from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
CACHE = "output/hitter_ifh_balls_2025_2026.parquet"
INFIELD = {"1B", "2B", "3B", "SS", "P", "C"}
rng = np.random.default_rng(3)


def load():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    d = read_sql_df(text("""select batter, year, game_id, date, play_result, exit_speed, angle, direction, distance, fielded_by_pos, tagged_hit_type, batter_side
        from pitches where game_type='reg_season' and year in (2025,2026) and play_result in ('single','double','triple','out','fielders_choice','error')"""))
    d["year"] = d["year"].astype(int)
    d.to_parquet(CACHE, index=False)
    return d


def prep(d):
    d = d[d["batter_side"].isin(["right", "left"])].copy()
    d["name_clean"] = d["batter"].apply(ip.lastfirst_to_clean)
    d["bunt"] = d["tagged_hit_type"].eq("bunt")
    d["tracked"] = d["exit_speed"].notna() & d["angle"].notna()
    d["gb"] = (d["angle"] < 10) | d["bunt"]
    single = d["play_result"] == "single"
    by_fielder = d["fielded_by_pos"].isin(INFIELD)
    short = d["fielded_by_pos"].isna() & (d["distance"] <= 110)
    d["ifh"] = (single & (by_fielder | short)).astype(int)
    side = d["batter_side"].map({"right": -1, "left": 1})
    d["pull_dir"] = d["direction"] * side
    d["lhh"] = (d["batter_side"] == "left").astype(int)
    return d


def expected_ifh(d):
    t = d[d["gb"] & d["tracked"]].copy()
    X = t[["exit_speed", "angle", "direction", "pull_dir", "lhh"]].fillna(0).to_numpy(); y = t["ifh"].to_numpy()
    groups = (t["batter"] + "|" + t["year"].astype(str)).to_numpy(); p = np.zeros(len(t))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=80, random_state=0).fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    d["xifh"] = np.nan; d.loc[t.index, "xifh"] = p
    return d


def season(d):
    g = d[d["gb"] & d["tracked"]].groupby(["name_clean", "year"]).agg(gb=("ifh", "size"), ifh=("ifh", "sum"), xifh=("xifh", "sum")).reset_index()
    allb = d.groupby(["name_clean", "year"]).agg(bip=("ifh", "size")).reset_index()
    s = g.merge(allb, on=["name_clean", "year"])
    s["ifh_gb"] = s["ifh"] / s["gb"]; s["xifh_gb"] = s["xifh"] / s["gb"]; s["resid_gb"] = s["ifh_gb"] - s["xifh_gb"]; s["ifh_bip"] = s["ifh"] / s["bip"]
    return s


def main():
    d = expected_ifh(prep(load()))
    s = season(d)
    print("A. DEFINITION AND BASE RATES")
    tr = d[d["tracked"]]
    print(f"   balls in play {len(d):,}; singles {int((d.play_result=='single').sum()):,}; infield hits {int(d.ifh.sum()):,} ({d.ifh.sum()/ (d.play_result=='single').sum():.1%} of singles, {d.ifh.sum()/len(d):.2%} of balls in play; "
          f"bunt singles {int(((d.play_result=='single') & d.bunt).sum())}); ground balls {int(d.gb.sum()):,}")
    print("   by handedness (per ground ball, tracked):")
    h = tr[tr["gb"]].groupby("batter_side").agg(gb=("ifh", "size"), ifh_gb=("ifh", "mean"), xifh=("xifh", "mean")); print(h.round(4).to_string())
    print("   by fielder (share of infield hits):", d[d.ifh == 1]["fielded_by_pos"].fillna("short, unassigned").value_counts(normalize=True).round(3).to_dict())

    print("\nB. RELIABILITY (hitter-seasons with >= 60 tracked ground balls)")
    g = d[d["gb"] & d["tracked"]].copy(); res = {"ifh_gb": [], "resid": []}
    for rep in range(30):
        g["half"] = rng.integers(0, 2, len(g))
        w = g.groupby(["name_clean", "year", "half"]).agg(n=("ifh", "size"), a=("ifh", "sum"), x=("xifh", "sum")).reset_index().pivot_table(index=["name_clean", "year"], columns="half", values=["n", "a", "x"]).dropna()
        w = w[(w[("n", 0)] >= 20) & (w[("n", 1)] >= 20)]
        r0, r1 = w[("a", 0)] / w[("n", 0)], w[("a", 1)] / w[("n", 1)]; e0, e1 = r0 - w[("x", 0)] / w[("n", 0)], r1 - w[("x", 1)] / w[("n", 1)]
        res["ifh_gb"].append(r0.corr(r1)); res["resid"].append(e0.corr(e1))
    a = s[(s["year"] == 2025) & (s["gb"] >= 40)].set_index("name_clean"); b = s[(s["year"] == 2026) & (s["gb"] >= 40)].set_index("name_clean"); j = a.join(b, lsuffix="_25", rsuffix="_26", how="inner")
    rows = []
    for lab, k, c in (("IFH per ground ball", "ifh_gb", "ifh_gb"), ("IFH residual per ground ball (actual - expected)", "resid", "resid_gb")):
        r = float(np.mean(res[k])); rows.append({"measure": lab, "split-half r": r, "full-season reliability": 2 * r / (1 + r), "y/y 2025->26": j[f"{c}_25"].corr(j[f"{c}_26"]), "n (y/y)": len(j)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    sd_obs = s[s["gb"] >= 60]["resid_gb"].std(); n_mean = s[s["gb"] >= 60]["gb"].mean(); p = s[s["gb"] >= 60]["ifh_gb"].mean()
    print(f"   SD across hitters of the residual per ground ball: observed {sd_obs:.4f} vs binomial noise {np.sqrt(p*(1-p)/n_mean):.4f} (at {n_mean:.0f} ground balls)")

    print("\nC. VALIDITY against independent speed indicators (2026)")
    w = read_sql_df(text("select player, pa, sb, cs, gdp, gdp_exp, wgdp_runs, bsr_runs from war_players where year=2026 and game_type='reg_season'"))
    w["name_clean"] = w["player"].apply(ip.lastfirst_to_clean)
    lp = pd.read_parquet("output/fl_hitter_panel.parquet"); lp = lp[lp["year"] == 2026].set_index("name_clean")
    v = s[(s["year"] == 2026) & (s["gb"] >= 60)].set_index("name_clean").join(w.set_index("name_clean")[["pa", "sb", "cs", "gdp", "gdp_exp", "bsr_runs"]]).join(lp[["x3b"]])
    v = v[v["pa"] >= 120]
    v["sb_pa"] = v["sb"].fillna(0) / v["pa"]; v["gdp_over"] = (v["gdp"] - v["gdp_exp"]) / v["pa"]; v["x3b_pa"] = v["x3b"] / v["pa"]; v["bsr_pa"] = v["bsr_runs"] / v["pa"]
    print(f"   n = {len(v)} hitters (>= 60 ground balls, >= 120 PA):  corr of IFH residual / IFH rate with  SB per PA: {v['resid_gb'].corr(v['sb_pa']):+.2f} / {v['ifh_gb'].corr(v['sb_pa']):+.2f};  "
          f"GIDP over expected (negative = fast): {v['resid_gb'].corr(v['gdp_over']):+.2f} / {v['ifh_gb'].corr(v['gdp_over']):+.2f};  triples per PA: {v['resid_gb'].corr(v['x3b_pa']):+.2f} / {v['ifh_gb'].corr(v['x3b_pa']):+.2f};  "
          f"published baserunning runs per PA: {v['resid_gb'].corr(v['bsr_pa']):+.2f} / {v['ifh_gb'].corr(v['bsr_pa']):+.2f}")
    s.to_parquet("output/hitter_ifh_seasons.parquet", index=False)
    d.to_parquet("output/hitter_ifh_balls_scored.parquet", index=False)


if __name__ == "__main__":
    main()
