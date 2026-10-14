"""
fl_hitter_ifh_test2.py -- infield-hit rate as a sprint-speed signal, 2026 only (the iScore play-by-play fielder is not available for 2025, so a year-to-year test is impossible
this season; the first version of this study mixed two definitions and is superseded).

Infield hit = a single fielded by 1B / 2B / 3B / SS / P / C (fielded_by_pos).  Population = 2026 tracked ground balls (launch angle < 10 or a bunt) with a recorded fielder (~88%).
Speed signal = actual minus expected infield hits, expectation from a league-wide gradient-boosted model of (exit velocity, launch angle, spray direction, spray pull side, batter side),
scored out-of-fold with hitters grouped.

A  Base rates and reliability: random-half and chronological-half split (does it persist through the season?), plus the venue effect.
B  Validity: correlation with stolen bases per PA, GIDP over expected, triples, published baserunning runs (all 2026).
C  Does it predict the REST of a hitter's PA?  Within-season PA split (known sample of m PA -> wOBA over the remaining PA), added on top of the current model
   (regressed wOBA + xwOBA nudge + EV90 power prior), hitter-grouped CV, standard inverse-variance weights, hitter-cluster bootstrap.

Usage:  python fl_hitter_ifh_test2.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sqlalchemy import text

import fl_eval_weights as ew
import fl_hitter_ev as hev
import fl_hitter_pa_split_test as ps
import fl_innings_projection as ip
from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
INFIELD = {"1B", "2B", "3B", "SS", "P", "C"}
CACHE = "output/hitter_pa_ifh_2026.parquet"
rng = np.random.default_rng(8)
N_DRAWS = 25
SIZES = (30, 60, 120, 200)


def load():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    d = read_sql_df(text("""select batter, game_id, pa_id, date, pitch_call, k_or_bb, play_result, woba, xwoba, exit_speed, angle, direction, batter_side, barrel, fielded_by_pos, tagged_hit_type
        from pitches where game_type='reg_season' and year=2026
          and (k_or_bb in ('strikeout','walk') or pitch_call='hit_by_pitch' or play_result in ('single','double','triple','home_run','out','fielders_choice','error','sacrifice'))"""))
    d["year"] = 2026
    d.to_parquet(CACHE, index=False)
    return d


def prepare():
    d = load()
    d["kind"] = np.where(d.k_or_bb.isin(["strikeout", "walk"]), d.k_or_bb, np.where(d.pitch_call == "hit_by_pitch", "hbp", d.play_result))
    d = d[(d["kind"] != "sacrifice") & d["batter_side"].isin(["right", "left"])].copy().reset_index(drop=True)
    d["name_clean"] = d["batter"].apply(ip.lastfirst_to_clean)
    d["woba"] = d["woba"].fillna(0.0)
    d["xw"] = d["xwoba"].fillna(d.groupby("kind")["xwoba"].transform("mean")).fillna(d["woba"])
    d["bip"] = d["kind"].isin(["single", "double", "triple", "out", "fielders_choice", "error"])
    d["tracked"] = d["bip"] & d["exit_speed"].notna() & d["angle"].notna() & (d["exit_speed"] <= 120)
    d["bunt"] = d["tagged_hit_type"].eq("bunt")
    d["elig"] = d["tracked"] & ((d["angle"] < 10) | d["bunt"]) & d["fielded_by_pos"].notna()
    d["ifh"] = ((d["kind"] == "single") & d["fielded_by_pos"].isin(INFIELD)).astype(int)
    side = d["batter_side"].map({"right": -1, "left": 1}); d["pull_dir"] = d["direction"] * side; d["lhh"] = (d["batter_side"] == "left").astype(int)
    t = d[d["elig"]]
    X = t[["exit_speed", "angle", "direction", "pull_dir", "lhh"]].fillna(0).to_numpy(); y = t["ifh"].to_numpy(); groups = t["batter"].to_numpy(); p = np.zeros(len(t))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=80, random_state=0).fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    d["xifh"] = np.nan; d.loc[t.index, "xifh"] = p
    d["res"] = d["ifh"] - d["xifh"]
    d["venue"] = d["game_id"].str.split("-").str[1]
    return d


def partAB(d):
    e = d[d["elig"]]
    print("A. BASE RATES AND RELIABILITY (2026)")
    print(f"   eligible ground balls {len(e):,}; infield hits {int(e.ifh.sum()):,} ({e.ifh.mean():.1%} of ground balls); by side: " + ", ".join(f"{k} {v:.1%}" for k, v in e.groupby('batter_side').ifh.mean().items()))
    print("   rate by recorded fielder (share of the infield hits):", e[e.ifh == 1].fielded_by_pos.value_counts(normalize=True).round(3).to_dict())
    res = {"rate": [], "resid": []}
    for _ in range(30):
        g = e.assign(h=rng.integers(0, 2, len(e))).groupby(["name_clean", "h"]).agg(n=("ifh", "size"), a=("ifh", "sum"), x=("xifh", "sum")).reset_index().pivot_table(index="name_clean", columns="h", values=["n", "a", "x"]).dropna()
        w = w_ = g[(g[("n", 0)] >= 20) & (g[("n", 1)] >= 20)]
        r0, r1 = w[("a", 0)] / w[("n", 0)], w[("a", 1)] / w[("n", 1)]
        res["rate"].append(r0.corr(r1)); res["resid"].append((r0 - w[("x", 0)] / w[("n", 0)]).corr(r1 - w[("x", 1)] / w[("n", 1)]))
    # chronological halves
    e = e.assign(dt=pd.to_datetime(e["date"]))
    med = e.groupby("name_clean")["dt"].transform("median")
    e["early"] = (e["dt"] <= med).astype(int)
    c = e.groupby(["name_clean", "early"]).agg(n=("ifh", "size"), a=("ifh", "sum"), x=("xifh", "sum")).reset_index().pivot_table(index="name_clean", columns="early", values=["n", "a", "x"]).dropna()
    c = c[(c[("n", 0)] >= 20) & (c[("n", 1)] >= 20)]
    cr = (c[("a", 0)] / c[("n", 0)]).corr(c[("a", 1)] / c[("n", 1)]); cres = ((c[("a", 0)] - c[("x", 0)]) / c[("n", 0)]).corr((c[("a", 1)] - c[("x", 1)]) / c[("n", 1)])
    rows = [{"measure": "IFH per ground ball", "random-half r": np.mean(res["rate"]), "full-season reliability": 2 * np.mean(res["rate"]) / (1 + np.mean(res["rate"])), "first half vs second half r": cr},
            {"measure": "IFH residual (actual - expected)", "random-half r": np.mean(res["resid"]), "full-season reliability": 2 * np.mean(res["resid"]) / (1 + np.mean(res["resid"])), "first half vs second half r": cres}]
    print(pd.DataFrame(rows).round(3).to_string(index=False) + f"\n   (hitters with >= 20 eligible ground balls in each half: {len(c)})")
    # n0 for the residual from random-half reliability
    half = e.groupby("name_clean").size().mean() / 2
    r = float(np.mean(res["resid"])); n0 = half * (1 - r) / r
    print(f"   implied n0 for the residual: {n0:.0f} ground balls (mean half size {half:.0f})")
    # venue effect: odd/even game split of venue means
    v = e.assign(odd=(pd.factorize(e["game_id"])[0] % 2)).groupby(["venue", "odd"]).agg(n=("res", "size"), m=("res", "mean")).reset_index().pivot(index="venue", columns="odd", values=["n", "m"]).dropna()
    vv = v[("m", 0)].corr(v[("m", 1)]); sd = np.sqrt(max(np.cov(v[("m", 0)], v[("m", 1)])[0, 1], 0))
    print(f"   venue effect on the residual (odd vs even games, {len(v)} venues): corr {vv:.2f}, SD of true venue effect {sd:.4f} per ground ball (hitter-level SD of the residual {e.groupby('name_clean').res.mean().std():.4f})")
    return n0


def partB(d):
    print("\nB. VALIDITY against independent speed indicators (hitters with >= 60 eligible ground balls and >= 120 PA)")
    e = d[d["elig"]].groupby("name_clean").agg(gb=("ifh", "size"), rate=("ifh", "mean"), res=("res", "mean")).reset_index()
    w = read_sql_df(text("select player, pa, sb, cs, gdp, gdp_exp, bsr_runs from war_players where year=2026 and game_type='reg_season'")); w["name_clean"] = w["player"].apply(ip.lastfirst_to_clean)
    lp = pd.read_parquet("output/fl_hitter_panel.parquet"); lp = lp[lp["year"] == 2026][["name_clean", "x3b"]]
    v = e.merge(w[["name_clean", "pa", "sb", "gdp", "gdp_exp", "bsr_runs"]], on="name_clean").merge(lp, on="name_clean")
    v = v[(v["gb"] >= 60) & (v["pa"] >= 120)].copy()
    v["sb_pa"] = v["sb"].fillna(0) / v["pa"]; v["gdp_over"] = (v["gdp"] - v["gdp_exp"]) / v["pa"]; v["x3b_pa"] = v["x3b"] / v["pa"]; v["bsr_pa"] = v["bsr_runs"] / v["pa"]
    rows = [{"speed indicator": lab, "corr w/ IFH residual": v["res"].corr(v[c]), "corr w/ raw IFH rate": v["rate"].corr(v[c])} for lab, c in
            (("stolen bases per PA", "sb_pa"), ("published baserunning runs per PA", "bsr_pa"), ("GIDP over expected per PA (negative = fast)", "gdp_over"), ("triples per PA", "x3b_pa"))]
    print(f"   n = {len(v)}"); print(pd.DataFrame(rows).round(3).to_string(index=False))


def partC(d, n0_res):
    print("\nC. PREDICTING THE REST OF A HITTER'S PA (2026 only; within-season PA split; add-on on top of the current model)")
    GAMMA, B_EV, N0W = 0.485, 0.00377, 440.0
    ev, lg_ev = hev.ev90_table(years=(2026,))
    evd = ev.set_index("name_clean")
    lgx = d["xw"].mean(); lgw = d["woba"].mean()
    rows = []
    for (b, nm), g in d.groupby(["batter", "name_clean"]):
        n = len(g)
        if n < 150:
            continue
        woba, xw, elig, res, tracked = g["woba"].to_numpy(), g["xw"].to_numpy(), g["elig"].to_numpy(), g["res"].to_numpy(), g["tracked"].to_numpy()
        evv = np.where(tracked, g["exit_speed"].to_numpy(), np.nan)
        for m in SIZES:
            if n - m < 30:
                continue
            for _ in range(N_DRAWS):
                idx = rng.permutation(n); a, o = idx[:m], idx[m:]
                ne = int(elig[a].sum()); ev_k = evv[a][~np.isnan(evv[a])]
                rows.append({"batter": b, "m": m, "n_out": n - m, "w_in": woba[a].mean(), "xw_in": xw[a].mean(), "y": woba[o].mean(), "n_elig": ne,
                             "res_mean": np.nansum(res[a]) / ne if ne >= 3 else np.nan, "bbe": len(ev_k), "ev90": np.quantile(ev_k, 0.9) if len(ev_k) >= 3 else np.nan})
    D = pd.DataFrame(rows)
    print(f"   hitter-seasons with >= 150 PA: {D['batter'].nunique()}; draws per size {N_DRAWS}")
    for m in SIZES:
        S = D[D["m"] == m].reset_index(drop=True)
        y = S["y"].to_numpy(); base0 = lgw + m * (S["w_in"].to_numpy() - lgw) / (m + N0W)
        xs = (m * S["xw_in"] + 200 * lgx) / (m + 200)
        evd_ = np.where(S["ev90"].isna(), 0.0, (S["ev90"].to_numpy() - lg_ev) * S["bbe"].to_numpy() / (S["bbe"].to_numpy() + hev.N0_EV90))
        base = base0 + GAMMA * (xs.to_numpy() - base0) + B_EV * evd_ * N0W / (m + N0W)         # current model, applied to the known sample
        wt, a_, b_ = ew.weights_from_base(y, base, S["n_out"].to_numpy())
        z = np.where(S["res_mean"].isna(), 0.0, S["res_mean"].fillna(0).to_numpy() * S["n_elig"].to_numpy() / (S["n_elig"].to_numpy() + n0_res)); z = z / z.std()
        groups = S["batter"].to_numpy()
        p_ifh = ps.cv_group(y, base, z[:, None], wt, groups)
        p0 = ps.cv_group(y, base, np.zeros((len(y), 0)), wt, groups)         # FAIR reference: the current model with the same cross-validated intercept
        ub = pd.Series(groups).unique(); idx_by = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array)
        dd = []
        for _ in range(400):
            pick = rng.choice(ub, len(ub)); ii = np.concatenate([idx_by[u] for u in pick]); dd.append(ps.rm(y[ii], p_ifh[ii], wt[ii]) - ps.rm(y[ii], p0[ii], wt[ii]))
        A = np.column_stack([np.ones(len(y)), z]); bb = np.linalg.lstsq(A * np.sqrt(wt)[:, None], (y - base) * np.sqrt(wt), rcond=None)[0]
        print(f"   known sample {m:3d} PA: current-model RMSE {ps.rm(y, base, wt):.4f}, with a CV intercept {ps.rm(y, p0, wt):.4f}; adding the IFH residual changes the intercept-only reference by {np.mean(dd):+.5f}  90% [{np.percentile(dd,5):+.5f}, {np.percentile(dd,95):+.5f}]; "
              f"coefficient {bb[1]:+.4f} wOBA per SD of shrunk residual")


def main():
    d = prepare()
    n0 = partAB(d)
    partB(d)
    partC(d, n0)


if __name__ == "__main__":
    main()
