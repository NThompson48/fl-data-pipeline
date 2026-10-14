"""
fl_hitter_pa_quality_test.py -- do better hitters get more playing time than the PA model gives them?  (Top-end calibration of projected hitter WAR: the eight hitters who finished >= 4 WAR in the 2026 backtest were projected
at 230-320 PA and played 405-481.)
PA per team game, returners (any PA next year), 2022->23 ... 2025->26, leave-one-transition-out:
  A (current)  PA_next/G_next = a + b PA_t/G_t
  B            + c x q,  q = the hitter's projected wOBA deviation (the production three-season regression, n0 fit on the training transitions, minus next year's league level; no Trackman terms, so it exists for all 5 seasons)
  C            + c x q + d x (PA_t/G_t) x q       (quality matters more for full-timers)
Equal weights (playing time is not a binomial rate), hitter-cluster bootstrap, RMSE and MAE (reporting standard); also bias by quality decile and for the top decile (the group the user asked about), and the share of returners
whose PA_next exceeds 400 at 102 games' length.
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_baseline_backtest as bt
import fl_hitter_depth3_test as d3
import fl_hitter_model_dev as dev
import fl_hitter_projection as hp

warnings.filterwarnings("ignore")
G = hp.GAMES


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; bt.MIN_NEXT = 1; fr = d3.frame(q, "woba"); delta = hp.SPEC["woba"][0]
    pm = dev.pa_model(p); r = pm.dropna(subset=["pa_next"]).copy(); r["g_t"] = r["t"].map(G); r["g_next"] = r["t"].map(lambda t: G[t + 1]); r["x_pg"] = r["pa_t"] / r["g_t"]; r["y_pg"] = r["pa_next"] / r["g_next"]
    parts = []
    for T in (2022, 2023, 2024, 2025):
        bt.MIN_NEXT = 100; tr100 = d3.frame(q, "woba"); tr100 = tr100[tr100["t"] != T]; bt.MIN_NEXT = 1
        n0 = d3.fit_n0(tr100, delta, 3); te = fr[fr["t"] == T].copy(); te["q"] = d3.pred(te, n0, delta, 3) - te["lg_next"]; parts.append(te[["name_clean", "t", "q"]])
    r = r.merge(pd.concat(parts), on=["name_clean", "t"], how="inner"); print(f"{len(r)} returning hitter-transitions with a projected wOBA deviation; q SD {r['q'].std():.4f}; corr(q, PA_next/G) {r['q'].corr(r['y_pg']):+.2f}, corr(PA_t/G, PA_next/G) {r['x_pg'].corr(r['y_pg']):+.2f}")
    r["xq"] = r["x_pg"] * r["q"]; specs = {"A": ["x_pg"], "B": ["x_pg", "q"], "C": ["x_pg", "q", "xq"]}; out = {k: [] for k in specs}; Y, I, F, Q, GN = [], [], [], [], []
    for T in (2022, 2023, 2024, 2025):
        tr, te = r[r["t"] != T], r[r["t"] == T]
        for k, cols in specs.items():
            X = np.column_stack([np.ones(len(tr))] + [tr[c] for c in cols]); b = np.linalg.lstsq(X, tr["y_pg"].to_numpy(), rcond=None)[0]
            out[k].append(np.maximum(te["g_next"].to_numpy() * (b[0] + np.column_stack([te[c] for c in cols]) @ b[1:]), 20))
        Y.append(te["pa_next"].to_numpy()); I.append(te["name_clean"].to_numpy()); F.append(np.full(len(te), T)); Q.append(te["q"].to_numpy()); GN.append(te["g_next"].to_numpy())
    y, i, f, qq = map(np.concatenate, (Y, I, F, Q)); P = {k: np.concatenate(v) for k, v in out.items()}; w = np.ones(len(y)); rows = []
    for k in ("B", "C"):
        rows.append({"model": f"{k} vs A (current)", **er.compare(y, P["A"], P[k], w, i, n=800)})
    print("PA next year, LOTO, returners (old = A, new = B / C)\n" + er.table(rows, 2))
    top = qq >= np.quantile(qq, 0.9); reg = np.concatenate(GN) is not None
    print(f"\nTOP DECILE of projected wOBA quality (n = {top.sum()}): mean actual PA {y[top].mean():.0f}; model A projects {P['A'][top].mean():.0f} (bias {y[top].mean() - P['A'][top].mean():+.0f}), B {P['B'][top].mean():.0f} ({y[top].mean() - P['B'][top].mean():+.0f}), C {P['C'][top].mean():.0f} ({y[top].mean() - P['C'][top].mean():+.0f})")
    dec = pd.qcut(pd.Series(qq).rank(method="first"), 5, labels=["Q1 (worst)", "Q2", "Q3", "Q4", "Q5 (best)"])
    t = pd.DataFrame({"q": dec, "actual PA": y, "A": P["A"], "B": P["B"], "C": P["C"]}).groupby("q").mean().round(0); print("\nmean PA by projected-quality quintile (actual vs models):\n" + t.to_string())
    for k in ("A", "B", "C"):
        pass
    X = np.column_stack([np.ones(len(r)), r["x_pg"], r["q"]]); b = np.linalg.lstsq(X, r["y_pg"].to_numpy(), rcond=None)[0]; res = r["y_pg"] - X @ b
    print(f"\nfinal fit B: PA/game_next = {b[0]:.3f} + {b[1]:.3f} PA_t/G_t + {b[2]:.1f} q   (q in wOBA points: +.010 wOBA = +{b[2] * 0.01 * 102:.1f} PA at 102 games); resid SD {res.std(ddof=3):.3f}/game = {res.std(ddof=3) * 102:.1f} PA")
    print("example (102-game season, 470 PA): " + ", ".join(f"q = {qv:+.3f}: {102 * (b[0] + b[1] * 470 / 102 + b[2] * qv):.0f} PA" for qv in (-0.01, 0.0, 0.01, 0.02)))


if __name__ == "__main__":
    main()
