"""
fl_hitter_history_depth_test.py -- does using several prior seasons beat using only the last one?

Model: lg_next + sum_j d^j n_j (x_j - lg_j) / (sum_j d^j n_j + n0), j = 0 (last season) ... K-1, d = weight per extra year back, n0 = PA of league-average performance mixed in.
K = 1 is naive shrinkage; K = 2 is the current model's direct half (d = .75).  For each K the pair (d, n0) is fit on the OTHER three transitions (leave-one-transition-out), standard
inverse-variance weights (frozen), park-neutral wOBA, next-year PA >= 100.  Hitters with fewer than K seasons of history simply use what exists, so in 2022 -> 23 every K is the same model.
Part 1  wOBA: RMSE by K, overall / by transition / by prior-season sample size, paired bootstrap.
Part 2  The same for K%, BB% and HR/PA (the components the blend uses).
Part 3  What d and n0 the data want for each K (fit on all transitions), i.e. how fast old seasons should fade.

Usage:  python fl_hitter_history_depth_test.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev

warnings.filterwarnings("ignore")
rng = np.random.default_rng(6)
KMAX = 4
DELTAS = (0.25, 0.5, 0.75, 1.0)


def build(q, stat):
    den = bt.DEN[stat]
    lg = bt.league(q, stat)
    base = bt.transitions(q, stat)[["name_clean", "t", "x_next", "n_next", "w"]]
    base["lg_next"] = base["t"].map(lambda t: lg[t + 1])
    by = {y: q[q["year"] == y].set_index("name_clean") for y in range(2022, 2027)}
    for j in range(KMAX):
        xs, ns, ls = [], [], []
        for nm, t in zip(base["name_clean"], base["t"]):
            y = t - j
            if y >= 2022 and y in lg.index and nm in by[y].index and pd.notna(by[y].at[nm, stat]) and by[y].at[nm, den] > 0:
                xs.append(by[y].at[nm, stat]); ns.append(float(by[y].at[nm, den])); ls.append(lg[y])
            else:
                xs.append(0.0); ns.append(0.0); ls.append(0.0)
        base[f"x{j}"], base[f"n{j}"], base[f"l{j}"] = xs, ns, ls
    return base[base["n0"] > 0].reset_index(drop=True)


def predict(d, K, delta, n0):
    num = np.zeros(len(d)); den = np.zeros(len(d))
    for j in range(K):
        wj = delta ** j
        num += wj * d[f"n{j}"].to_numpy() * (d[f"x{j}"].to_numpy() - d[f"l{j}"].to_numpy()); den += wj * d[f"n{j}"].to_numpy()
    return d["lg_next"].to_numpy() + num / (den + n0)


def wse(d, p):
    w = d["w"].to_numpy(); return float(np.sum(w * (d["x_next"].to_numpy() - p) ** 2) / w.sum())


def fit(d, K):
    best = None
    for delta in (DELTAS if K > 1 else (1.0,)):
        r = minimize_scalar(lambda lg: wse(d, predict(d, K, delta, np.exp(lg))), bounds=(np.log(5), np.log(20000)), method="bounded")
        if best is None or r.fun < best[0]:
            best = (float(r.fun), delta, float(np.exp(r.x)))
    return best[1], best[2]


def loto(d, K):
    pr = np.zeros(len(d)); par = {}
    for t in sorted(d["t"].unique()):
        tr, te = d[d["t"] != t], d[d["t"] == t]
        delta, n0 = fit(tr, K); par[t] = (delta, n0); pr[(d["t"] == t).to_numpy()] = predict(te, K, delta, n0)
    return pr, par


def rm(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def boot(y, A, B, w, g, n=1000):
    ug = pd.Series(g).unique(); by = pd.Series(np.arange(len(y))).groupby(g).apply(np.array); out = []
    for _ in range(n):
        ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); out.append(rm(y[ii], A[ii], w[ii]) - rm(y[ii], B[ii], w[ii]))
    return float(np.mean(out)), float(np.percentile(out, 5)), float(np.percentile(out, 95))


def part(q, stat, label, detail):
    d = build(q, stat)
    y = d["x_next"].to_numpy(); w = d["w"].to_numpy(); g = d["name_clean"].to_numpy()
    P, PAR = {}, {}
    for K in range(1, KMAX + 1):
        P[K], PAR[K] = loto(d, K)
    avail = (d[[f"n{j}" for j in range(KMAX)]] > 0).sum(axis=1)
    print(f"\n{label}: n = {len(d)} hitter-transitions; seasons of history available: " + ", ".join(f"{k}: {int((avail==k).sum())}" for k in range(1, KMAX + 1)))
    rows = [{"model": "league average", "RMSE": rm(y, d["lg_next"].to_numpy(), w)}]
    for K in range(1, KMAX + 1):
        r = {"model": f"{K} season{'s' if K > 1 else ''}", "RMSE": rm(y, P[K], w)}
        if detail:
            for t in sorted(d["t"].unique()):
                m = (d["t"] == t).to_numpy(); r[f"t={t}"] = rm(y[m], P[K][m], w[m])
        rows.append(r)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    for K in (2, 3, 4):
        m, lo, hi = boot(y, P[K], P[K - 1], w, g)
        print(f"   {K} seasons minus {K-1}: {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]")
    m, lo, hi = boot(y, P[4], P[1], w, g); print(f"   4 seasons minus 1: {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]")
    if detail:
        d["bk"] = pd.cut(d["n0"], [0, 100, 250, 10 ** 5], labels=["< 100 PA", "100-249", "250+"])
        print("   by the hitter's PA in his LAST season (RMSE):")
        rows = []
        for b in ("< 100 PA", "100-249", "250+"):
            m = (d["bk"] == b).to_numpy(); r = {"last-season PA": b, "n": int(m.sum()), "league avg": rm(y[m], d["lg_next"].to_numpy()[m], w[m])}
            for K in range(1, KMAX + 1):
                r[f"K={K}"] = rm(y[m], P[K][m], w[m])
            rows.append(r)
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        print("   fitted (weight per extra year d, n0) on ALL transitions:", {K: tuple(round(v, 2) for v in fit(d, K)) for K in range(1, KMAX + 1)})
    return d


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]
    # the n0 column in `build` is the LAST season's PA/AB (j=0)
    print("PART 1: park-neutral wOBA")
    part(q, "woba", "wOBA", True)
    print("\nPART 2: components")
    for s, lab in (("k", "K%"), ("bbr", "BB%"), ("hrr", "HR/PA (park-neutral)")):
        part(q, s, lab, False)


if __name__ == "__main__":
    main()
