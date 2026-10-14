"""
fl_bridge_unified_vs_byleague.py -- head-to-head: the shipped by-league bridge vs
the unified one, with the two ingredients separated:

    INPUTS     shipped (latest-season K%, league-specific BB% averaging, no
               shrinkage)          vs   unified (shrink n0=50; K% decay .25 and
               staleness damping .7^(gap-1); BB% flat decay)
    STRUCTURE  by-league lines (a separate intercept+slope per league; shipped)
               vs  one pooled line   vs   pooled line + league intercepts

Scored on the same 513 debut pairs (College 249, Pioneer 73, MiLB 191) with
leave-one-out predictions (a pair is always predicted by a model that never saw
it), PA-weighted. Reported as % reduction in squared error vs. guessing that
league's own mean FL rate, and RMSE in percentage points, overall and per league.
Also: how much the roster bridge values would actually move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_bridge_recency_league_study as st
import fl_bridge_unified_test as un

LEAGUES = ["College", "Pioneer", "MiLB"]


def wls_loo(y, X, w):
    XtWX = X.T @ (X * w[:, None]); beta = np.linalg.solve(XtWX, X.T @ (w * y))
    e = y - X @ beta; h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(XtWX), X)
    return y - e / (1 - h)


def loo_mean(y, w):
    return (np.sum(w * y) - w * y) / (np.sum(w) - w)


def build(pairs, lgr):
    out = {}
    for metric, yname, col, dec, rho in (("K", "k_rate", "k", un.POLICY["dK"], 0.7), ("BB", "bb_rate", "bb", un.POLICY["dBB"], 1.0)):
        d = {"y": [], "w": [], "lg": [], "ship": [], "uni": []}
        for lg in LEAGUES:
            p = pairs[lg]; sd, sn = st.SHIPPED[(lg, metric)]
            d["y"].append(p[yname].values); d["w"].append(p["pa"].values.astype(float)); d["lg"].append(np.full(len(p), lg))
            d["ship"].append(np.array([un.source_feature(h, col, sd, sn, lgr[lg][col]) for h in p["hist"]]) - lgr[lg][col])
            u = np.array([un.source_feature(h, col, dec, un.POLICY["n0"], lgr[lg][col]) for h in p["hist"]]) - lgr[lg][col]
            d["uni"].append(u * rho ** (p["gap"].values - 1))
        out[metric] = {k: np.concatenate(v) for k, v in d.items()}
    return out


def evaluate(d):
    y, w, lg = d["y"], d["w"], d["lg"]
    I = np.column_stack([(lg == l).astype(float) for l in LEAGUES])
    # per-league baseline = that league's own LOO mean
    base = np.zeros(len(y))
    for l in LEAGUES:
        m = lg == l; base[m] = loo_mean(y[m], w[m])
    rows = []
    for inp in ("ship", "uni"):
        x = d[inp]
        preds = {}
        p = np.zeros(len(y))
        for l in LEAGUES:                      # by-league lines
            m = lg == l; p[m] = wls_loo(y[m], np.column_stack([np.ones(m.sum()), x[m]]), w[m])
        preds["by-league lines (shipped structure)"] = p
        preds["one pooled line"] = wls_loo(y, np.column_stack([np.ones(len(y)), x]), w)
        preds["pooled line + league intercepts"] = wls_loo(y, np.column_stack([I, x]), w)
        for st_name, pr in preds.items():
            r = {"inputs": "shipped" if inp == "ship" else "unified", "structure": st_name}
            sse0, sse = np.sum(w * (y - base) ** 2), np.sum(w * (y - pr) ** 2)
            r["overall SSE cut %"] = round(100 * (1 - sse / sse0), 1)
            r["RMSE_w pp"] = round(np.sqrt(sse / w.sum()) * 100, 2)
            for l in LEAGUES:
                m = lg == l
                r[l] = round(100 * (1 - np.sum(w[m] * (y[m] - pr[m]) ** 2) / np.sum(w[m] * (y[m] - base[m]) ** 2)), 1)
            rows.append(r)
    rows.append({"inputs": "(guess)", "structure": "league mean", "overall SSE cut %": 0.0,
                 "RMSE_w pp": round(np.sqrt(np.sum(w * (y - base) ** 2) / w.sum()) * 100, 2), **{l: 0.0 for l in LEAGUES}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()
    srcs = st.source_tables(); lgr = {lg: st.lg_rates(s) for lg, s in srcs.items()}
    debuts = st.fl_debuts(raw)
    pairs = {lg: st.build_pairs(debuts, s, lg) for lg, s in srcs.items()}
    data = build(pairs, lgr)
    for metric in ("K", "BB"):
        print(f"\n{metric}% -- LOO, % reduction in squared error vs each league's own-mean guess (higher = better); n={len(data[metric]['y'])}")
        print(evaluate(data[metric]).to_string(index=False))

    # practical impact on the 2027 roster
    print("\nROSTER IMPACT: shipped by-league bridge values vs unified (PA>=1, current_year=2026)")
    proj = pd.read_parquet("../projections-dashboard/data/fl_2027_projections.parquet")
    names = dict(zip(proj["pitcher_id"].astype(str), proj["pitcher"]))
    uni = un.Unified(srcs, lgr, un.POLICY, {"k": 0.7, "bb": 1.0}).for_pitchers(list(names), list(names.values()), 2026).set_index("pitcher_id")
    # pooled mappings fit on all pairs (u -> FL rate), as in fl_bridge_unified_test part 1
    maps = {}
    for metric in ("K", "BB"):
        d = data[metric]; X = np.column_stack([np.ones(len(d["y"])), d["uni"]]); w = d["w"]
        maps[metric] = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * d["y"]))
    j = proj.set_index(proj["pitcher_id"].astype(str)).join(uni)
    j["k_uni"] = maps["K"][0] + maps["K"][1] * j["u_k"].fillna(0); j["bb_uni"] = maps["BB"][0] + maps["BB"][1] * j["u_bb"].fillna(0)
    for metric, a, b, sc in (("K%", "fl_k_rate_bridge", "k_uni", "k_bridge_source"), ("BB%", "fl_bb_rate_bridge", "bb_uni", "bb_bridge_source")):
        m = j[sc] != "none"
        diff = (j.loc[m, b] - j.loc[m, a]) * 100
        print(f"  {metric}: {int(m.sum())} matched; shipped bridge SD {j.loc[m, a].std()*100:.2f}pp vs unified SD {j.loc[m, b].std()*100:.2f}pp; "
              f"corr {j.loc[m, a].corr(j.loc[m, b]):.2f}; |difference| median {diff.abs().median():.2f}pp, 90th pct {diff.abs().quantile(.9):.2f}pp, max {diff.abs().max():.1f}pp")
        print(f"     source winner changed for {(j.loc[m, 'src_' + ('k' if metric=='K%' else 'bb')].str.lower().replace({'milb':'milb'}) != j.loc[m, sc]).mean():.0%} of matched pitchers")
    show = j.loc[j["pitcher"].isin(["Johnson, Preston", "Smith, Jackson", "Gardner, Sam", "Stil, Matt"]), ["pitcher", "k_bridge_source", "fl_k_rate_bridge", "k_uni", "src_k", "bb_bridge_source", "fl_bb_rate_bridge", "bb_uni", "src_bb"]]
    print(show.round(3).to_string(index=False))
