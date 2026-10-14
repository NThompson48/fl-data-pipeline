"""
fl_bridge_recency_league_study.py -- audit and test of how the other-league
bridge treats RECENCY and LEAGUE.

What the shipped code does (audited from the modules, not assumed):
                      College (FanGraphs)    Pioneer (StatsCrew)     Affiliated MiLB
  seasons available   2023-2026              2021-2025 (no 2026)     2018-2026 (MLB Stats API)
  K% input            latest season          latest season           latest season
  BB% input           TBF x 0.5^lag avg      TBF-weighted flat avg   TBF-weighted flat avg
  shrink by TBF?      no                     no                      no
  bridge fit          plain OLS              PA-weighted WLS         PA-weighted WLS
  collision handling  upid                   none (name only)        MLBAM id
  pairs               debuts, latest <= Y-1  debuts, latest <= Y-1   debuts, latest <= Y-1
  roster match        single best of 3 sources by R2 x TBF x 0.5^(2026 - latest season)

This script rebuilds the debut-pair data for all three leagues WITH their
source histories so every choice can be varied on equal footing, then answers:
  A. Pair sample: sizes, source-gap distribution, TBF, matching quality.
  B. Recency: for each league and for one consistent policy across leagues, how
     do the K% and BB% inputs behave under decay d (latest-only .. flat) and
     TBF shrinkage n0? (leave-one-out, PA-weighted, vs the league-mean guess)
  C. Staleness: does the translation weaken with the gap between the source
     season and the FL debut year?
  D. League: separate bridges vs one pooled bridge with shared slope.

Usage:  python fl_bridge_recency_league_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_college_bridge as cb
import fl_pioneer_bridge as pb
import fl_milb_bridge as mb

DEBUT_YEARS = [(2025, bt.START_2025, bt.END_2025), (2026, bt.START_2026, bt.END_2026)]
DECAYS = (0.0, 0.25, 0.5, 0.75, 1.0)
N0S = (0, 50, 100, 200, 400)
SHIPPED = {("College", "K"): (0.0, 0), ("College", "BB"): (0.5, 0),
           ("Pioneer", "K"): (0.0, 0), ("Pioneer", "BB"): (1.0, 0),
           ("MiLB", "K"): (0.0, 0), ("MiLB", "BB"): (1.0, 0)}


# --------------------------------------------------------------------------- #
#  source tables -> common schema (kid, name_clean, season, tbf, k, bb)
# --------------------------------------------------------------------------- #
def source_tables():
    cy = cb.load_all_college_years()
    college = pd.DataFrame({"kid": cy["upid"].astype(str), "name_clean": cy["name_clean"], "season": cy["season"],
                            "tbf": cy["tbf"], "k": cy["k_pct"], "bb": cy["bb_pct"]})
    py = pb.load_all_pioneer_years()
    pioneer = pd.DataFrame({"kid": py["name_clean"], "name_clean": py["name_clean"], "season": py["year"],
                            "tbf": py["tbf"], "k": py["k_pct"], "bb": py["bb_pct"]})
    my = mb.load_all_milb_years()
    milb = pd.DataFrame({"kid": my["pitcher"].astype(str), "name_clean": my["name_clean"], "season": my["year"],
                         "tbf": my["tbf"], "k": my["k_pct"], "bb": my["bb_pct"]})
    return {"College": college, "Pioneer": pioneer, "MiLB": milb}


def fl_debuts(raw):
    seen, rows = set(), []
    names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    for fl_year, s, e in DEBUT_YEARS:
        k = fp.build_period_kbb(fp.filter_period(raw, s, e), min_pa=1)
        k = k[~k["pitcher_id"].isin(seen)]; seen |= set(k["pitcher_id"])
        k = k.merge(names, on="pitcher_id", how="left")
        k["name_clean"] = k["pitcher"].apply(cb._fl_name_clean); k["fl_year"] = fl_year
        k["pitcher_id"] = k["pitcher_id"].astype(str)
        rows.append(k)
    d = pd.concat(rows, ignore_index=True)
    return d.sort_values("PA", ascending=False).drop_duplicates(["name_clean", "fl_year"], keep="first")


def build_pairs(debuts, src, league):
    """One row per FL debut with a source history at or before fl_year-1; `hist` kept for re-aggregation."""
    by_name = {n: g for n, g in src.groupby("name_clean")}
    rows = []
    for _, r in debuts.iterrows():
        g = by_name.get(r["name_clean"])
        if g is None:
            continue
        g = g[g["season"] <= r["fl_year"] - 1]
        if g.empty:
            continue
        if g["kid"].nunique() > 1:   # same-name collision -> the person whose history is most recent (ties: total TBF)
            lat = g.groupby("kid").agg(ls=("season", "max"), tb=("tbf", "sum")).sort_values(["ls", "tb"], ascending=False)
            g = g[g["kid"] == lat.index[0]]
        rows.append({"league": league, "pitcher_id": r["pitcher_id"], "fl_year": r["fl_year"], "pa": r["PA"],
                     "k_rate": r["K_rate"], "bb_rate": r["BB_rate"], "kid": g["kid"].iloc[0],
                     "id_match": str(g["kid"].iloc[0]) == r["pitcher_id"],
                     "latest": int(g["season"].max()), "hist": g[["season", "tbf", "k", "bb"]].reset_index(drop=True)})
    out = pd.DataFrame(rows)
    out["gap"] = out["fl_year"] - out["latest"]
    out["tbf_latest"] = out["hist"].apply(lambda h: float(h.loc[h["season"].idxmax(), "tbf"]))
    out["tbf_multi"] = out["hist"].apply(lambda h: float(h["tbf"].sum()))
    out["n_years"] = out["hist"].apply(len)
    return out


def feature(hist, col, decay, n0, lg):
    """TBF x decay^lag weighted average (decay 0 -> latest season only), then shrunk toward the league mean by
    n0 pseudo-batters: (sum w*r + n0*lg) / (sum w + n0) with w = TBF-style weights on the TBF scale."""
    lag = hist["season"].max() - hist["season"]
    w = hist["tbf"] * np.where(lag == 0, 1.0, decay ** lag) if decay > 0 else hist["tbf"] * (lag == 0)
    if decay == 0:
        w = hist["tbf"] * (lag == 0)
    return (float((w * hist[col]).sum()) + n0 * lg) / (float(w.sum()) + n0)


# --------------------------------------------------------------------------- #
#  leave-one-out WLS (closed form) and weighted R2 against the LOO league-mean guess
# --------------------------------------------------------------------------- #
def loo_r2(y, X, w):
    y, w = np.asarray(y, float), np.asarray(w, float)
    X = np.column_stack([np.ones(len(y)), X]) if X.ndim else np.ones((len(y), 1))
    XtWX = X.T @ (X * w[:, None])
    beta = np.linalg.solve(XtWX, X.T @ (w * y))
    e = y - X @ beta
    h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(XtWX), X)
    e_loo = e / (1 - h)
    base = (np.sum(w * y) - w * y) / (np.sum(w) - w)
    return 1 - np.sum(w * e_loo ** 2) / np.sum(w * (y - base) ** 2)


def lg_rates(src):
    return {"k": np.average(src["k"], weights=src["tbf"]), "bb": np.average(src["bb"], weights=src["tbf"])}


def main():
    raw = fp.load_fl_pitches_cached()
    srcs = source_tables(); debuts = fl_debuts(raw)
    print(f"FL debut pitchers (2025+2026 first appearances): {len(debuts)}")
    pairs = {lg: build_pairs(debuts, s, lg) for lg, s in srcs.items()}
    lgr = {lg: lg_rates(s) for lg, s in srcs.items()}

    # ---------------- A. sample audit ----------------
    print("\nA. PAIR SAMPLES")
    rows = []
    for lg, p in pairs.items():
        s = srcs[lg]
        rows.append({"league": lg, "source seasons": f"{s['season'].min()}-{s['season'].max()}", "source rows": len(s),
                     "pairs": len(p), "gap=1": int((p.gap == 1).sum()), "gap=2": int((p.gap == 2).sum()), "gap>=3": int((p.gap >= 3).sum()),
                     "median TBF latest": int(p.tbf_latest.median()), "median TBF total": int(p.tbf_multi.median()),
                     "multi-year share": round((p.n_years > 1).mean(), 2), "median FL PA": int(p.pa.median()),
                     "league K%/BB% (source)": f"{lgr[lg]['k']:.3f}/{lgr[lg]['bb']:.3f}"})
    print(pd.DataFrame(rows).to_string(index=False))
    mp = pairs["MiLB"]
    print(f"   MiLB match quality: FL id == MLBAM id for {mp.id_match.sum()}/{len(mp)} pairs; the other {int((~mp.id_match).sum())} are NAME-only matches")
    print(f"   overlap: pitchers matched in >1 league: " +
          str({f"{a}&{b}": len(set(pairs[a].pitcher_id) & set(pairs[b].pitcher_id)) for a, b in (("College", "Pioneer"), ("College", "MiLB"), ("Pioneer", "MiLB"))}))

    # ---------------- B. recency / shrinkage grid ----------------
    def feats(lg, metric, d, n0):
        p = pairs[lg]
        col = "k" if metric == "K" else "bb"
        return np.array([feature(h, col, d, n0, lgr[lg][col]) for h in p["hist"]])

    print("\nB. RECENCY x TBF-SHRINKAGE: LOO weighted R2 vs league-mean guess (rows = decay; cols = shrinkage n0 in batters; * = shipped)")
    best_cfg = {}
    for metric in ("K", "BB"):
        yname = "k_rate" if metric == "K" else "bb_rate"
        for lg in pairs:
            p = pairs[lg]; y = p[yname].values; w = p["pa"].values.astype(float)
            grid = pd.DataFrame(index=[f"d={d}" for d in DECAYS], columns=[f"n0={n}" for n in N0S], dtype=float)
            for d in DECAYS:
                for n0 in N0S:
                    grid.loc[f"d={d}", f"n0={n0}"] = loo_r2(y, feats(lg, metric, d, n0), w)
            sd, sn = SHIPPED[(lg, metric)]
            print(f"\n  {lg} {metric}%  (n={len(p)}; shipped d={sd}, n0={sn}: {grid.loc[f'd={sd}', f'n0={sn}']:.3f}; best {grid.values.max():.3f})")
            print(grid.round(3).to_string())
            best_cfg[(lg, metric)] = grid

    print("\n  POOLED, ONE CONSISTENT POLICY across the three leagues (league-specific intercept + slope; LOO over all pairs; weights PA)")
    for metric in ("K", "BB"):
        yname = "k_rate" if metric == "K" else "bb_rate"
        rows = []
        for d in DECAYS:
            r = {"decay": d}
            for n0 in N0S:
                Y, W, Xs = [], [], []
                for i, lg in enumerate(pairs):
                    p = pairs[lg]; f = feats(lg, metric, d, n0)
                    Y.append(p[yname].values); W.append(p["pa"].values.astype(float))
                    cols = np.zeros((len(p), 3)); cols[:, i] = 1; Xs.append(np.column_stack([cols, f[:, None] * cols]))
                X = np.vstack(Xs); Yv = np.concatenate(Y); Wv = np.concatenate(W)
                # drop the explicit intercept column (dummies carry it): reuse loo_r2 by adding ones then removing collinearity via lstsq
                Xf = X[:, :]  # 3 league intercepts + 3 league slopes
                XtWX = Xf.T @ (Xf * Wv[:, None]); beta = np.linalg.solve(XtWX, Xf.T @ (Wv * Yv)); e = Yv - Xf @ beta
                h = Wv * np.einsum("ij,jk,ik->i", Xf, np.linalg.inv(XtWX), Xf); el = e / (1 - h)
                base = (np.sum(Wv * Yv) - Wv * Yv) / (np.sum(Wv) - Wv)
                r[f"n0={n0}"] = round(1 - np.sum(Wv * el ** 2) / np.sum(Wv * (Yv - base) ** 2), 3)
            rows.append(r)
        print(f"\n  {metric}% pooled LOO R2_w (rows = decay):"); print(pd.DataFrame(rows).set_index("decay").to_string())

    # ---------------- C. staleness ----------------
    print("\nC. STALENESS: translation by gap (source season -> FL debut year), shipped inputs; PA-weighted OLS per league x gap bucket")
    rows = []
    for lg, p in pairs.items():
        for metric, yname, col in (("K", "k_rate", "k"), ("BB", "bb_rate", "bb")):
            d_, n_ = SHIPPED[(lg, metric)]
            f = np.array([feature(h, col, d_, n_, lgr[lg][col]) for h in p["hist"]])
            for lab, mask in (("gap=1", p.gap == 1), ("gap>=2", p.gap >= 2)):
                m = mask.values
                if m.sum() < 15:
                    continue
                X = np.column_stack([np.ones(m.sum()), f[m]]); w = p["pa"].values[m].astype(float); y = p[yname].values[m]
                beta = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))
                rows.append({"league": lg, "metric": metric, "bucket": lab, "n": int(m.sum()), "corr": round(np.corrcoef(f[m], y)[0, 1], 3),
                             "slope": round(beta[1], 3), "mean FL rate": round(np.average(y, weights=w), 4), "mean source rate": round(np.average(f[m], weights=w), 4)})
    print(pd.DataFrame(rows).to_string(index=False))
    # pooled interaction test: does gap>=2 change the slope, controlling league intercept+slope?
    print("\n   pooled test: FL_rate ~ league intercepts + feature*league slopes + gap>=2 shift + feature x (gap>=2):")
    for metric, yname, col in (("K", "k_rate", "k"), ("BB", "bb_rate", "bb")):
        Y, W, X = [], [], []
        for i, lg in enumerate(pairs):
            p = pairs[lg]; d_, n_ = SHIPPED[(lg, metric)]
            f = np.array([feature(h, col, d_, n_, lgr[lg][col]) for h in p["hist"]]) - lgr[lg][col]
            g2 = (p.gap.values >= 2).astype(float)
            cols = np.zeros((len(p), 3)); cols[:, i] = 1
            X.append(np.column_stack([cols, f[:, None] * cols, g2, f * g2])); Y.append(p[yname].values); W.append(p["pa"].values.astype(float))
        X, Y, W = np.vstack(X), np.concatenate(Y), np.concatenate(W)
        XtWX = X.T @ (X * W[:, None]); beta = np.linalg.solve(XtWX, X.T @ (W * Y)); e = Y - X @ beta
        sig2 = np.sum(W * e ** 2) / (len(Y) - X.shape[1]); cov = sig2 * np.linalg.inv(XtWX); se = np.sqrt(np.diag(cov))
        print(f"   {metric}%: gap>=2 shift {beta[-2]:+.4f} (t={beta[-2]/se[-2]:+.1f}); slope change {beta[-1]:+.3f} (t={beta[-1]/se[-1]:+.1f});"
              f" league slopes {np.round(beta[3:6], 3)}")

    # ---------------- D. league structure ----------------
    print("\nD. LEAGUE STRUCTURE (shipped inputs; LOO weighted R2 over all pairs pooled)")
    for metric, yname, col in (("K", "k_rate", "k"), ("BB", "bb_rate", "bb")):
        Y, W, F, L = [], [], [], []
        for i, lg in enumerate(pairs):
            p = pairs[lg]; d_, n_ = SHIPPED[(lg, metric)]
            f = np.array([feature(h, col, d_, n_, lgr[lg][col]) for h in p["hist"]])
            Y.append(p[yname].values); W.append(p["pa"].values.astype(float)); F.append(f); L.append(np.full(len(p), i))
        Y, W, F, L = (np.concatenate(a) for a in (Y, W, F, L))
        I = np.column_stack([(L == i).astype(float) for i in range(3)])
        Fdev = F - np.array([lgr[lg][col] for lg in pairs])[L]
        designs = {"one line for all leagues (no league terms)": np.column_stack([np.ones(len(Y)), F]),
                   "league intercepts, common slope (raw source rate)": np.column_stack([I, F]),
                   "league intercepts, common slope (source rate relative to its league mean)": np.column_stack([I, Fdev]),
                   "league intercepts + league slopes (shipped structure)": np.column_stack([I, I * F[:, None]])}
        print(f"\n  {metric}%:")
        for nm, X in designs.items():
            XtWX = X.T @ (X * W[:, None]); beta = np.linalg.solve(XtWX, X.T @ (W * Y)); e = Y - X @ beta
            h = W * np.einsum("ij,jk,ik->i", X, np.linalg.inv(XtWX), X); el = e / (1 - h)
            base = (np.sum(W * Y) - W * Y) / (np.sum(W) - W)
            print(f"    {nm:78s} LOO R2_w {1 - np.sum(W * el ** 2) / np.sum(W * (Y - base) ** 2):+.3f}   params {np.round(beta, 3)}")

    return pairs, lgr


if __name__ == "__main__":
    main()
