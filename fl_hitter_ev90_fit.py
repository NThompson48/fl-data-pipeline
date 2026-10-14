"""
fl_hitter_ev90_fit.py -- fit the production coefficient for the EV90 "power prior" term in raw units, jointly with the xwOBA nudge.

Same within-season PA-split design as fl_hitter_pa_split_test.py (hitter-seasons with >= 150 PA; known sample m = 30 / 60 / 120 / 200 random PA; target = wOBA over the rest;
standard inverse-variance weights per m).  Model on the base's miss:   y - base = b0 + g * (shrunk xwOBA - base) + b * x * n0w / (m + n0w)
with x = (EV90_known - league EV90) * balls / (balls + 14.3)  in mph (EV90 shrunk by its reliability, n0 = 14.3 batted balls), n0w = 440 (the wOBA regression constant): the
"informed prior" form, so the term fades as the sample grows.  Pooled over the four sample sizes; standard errors from a bootstrap over hitter-seasons.

Usage:  python fl_hitter_ev90_fit.py
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_pa_split_test as ps

warnings.filterwarnings("ignore")
N0_EV90 = 14.26
rng = np.random.default_rng(5)


def main():
    d = ps.prep()
    ps.SIZES = (30, 60, 120, 200)
    D = ps.draws(d)
    t = d[d["tracked"]].groupby(["batter", "year"])["ev"].agg(n="size", ev90=lambda s: np.quantile(s, 0.9)).reset_index()
    lgv = t[t["n"] >= 50].groupby("year")["ev90"].mean()
    print("league EV90 (hitter-seasons with >= 50 tracked balls):", lgv.round(2).to_dict(), "| SD across hitters", t[t["n"] >= 50]["ev90"].std().round(2))
    Xs, Ys, Ws, G = [], [], [], []
    for m in ps.SIZES:
        S = D[D["m"] == m].reset_index(drop=True)
        y = S["y"].to_numpy(); base = (S["lg_w"] + m * (S["w_in"] - S["lg_w"]) / (m + ps.N0_WOBA)).to_numpy()
        wt, _, _ = ew.weights_from_base(y, base, S["n_out"].to_numpy())
        xs = (m * S["xw_in"] + ps.N0_X * S["lg_x"]) / (m + ps.N0_X)
        zx = (xs - base).to_numpy()
        lg = S["year"].map(lgv).to_numpy()
        x = np.where(S["ev90"].isna(), 0.0, (S["ev90"].to_numpy() - lg) * S["bbe"].to_numpy() / (S["bbe"].to_numpy() + N0_EV90))
        Xs.append(np.column_stack([np.ones(len(y)), zx, x * ps.N0_WOBA / (m + ps.N0_WOBA)])); Ys.append(y - base); Ws.append(wt)
        G.append((S["batter"] + "|" + S["year"].astype(str)).to_numpy())
    X = np.vstack(Xs); Y = np.concatenate(Ys); W = np.concatenate(Ws); G = np.concatenate(G)
    fit = lambda idx: np.linalg.lstsq(X[idx] * np.sqrt(W[idx])[:, None], Y[idx] * np.sqrt(W[idx]), rcond=None)[0]
    b = fit(np.arange(len(Y)))
    ub = pd.Series(G).unique(); by = pd.Series(np.arange(len(G))).groupby(G).apply(np.array)
    bs = np.array([fit(np.concatenate([by[u] for u in rng.choice(ub, len(ub))])) for _ in range(300)])
    print(f"pooled rows {len(Y):,}; joint fit: intercept {b[0]:+.4f}, nudge gamma {b[1]:.3f} (SE {bs[:,1].std():.3f}), EV90 term {b[2]:.5f} wOBA per mph of shrunk EV90 (SE {bs[:,2].std():.5f}) at zero sample")
    print(f"   -> one SD of hitter EV90 ({t[t['n']>=50]['ev90'].std():.2f} mph) is worth {b[2]*t[t['n']>=50]['ev90'].std():+.4f} wOBA before the (1 - reliability) fade; at 300 PA of evidence the fade factor is {440/740:.2f}")
    for m, (i0, i1) in zip(ps.SIZES, np.cumsum([0] + [len(x) for x in Ys])[:-1:1].reshape(-1, 1).tolist() and [(a, c) for a, c in zip(np.cumsum([0] + [len(x) for x in Ys])[:-1], np.cumsum([len(x) for x in Ys]))]):
        idx = np.arange(i0, i1); bm = fit(idx)
        print(f"   m={m:3d}: gamma {bm[1]:.2f}, EV90 {bm[2]:.5f} per mph")


if __name__ == "__main__":
    main()
