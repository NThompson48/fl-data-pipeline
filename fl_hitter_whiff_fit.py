"""
fl_hitter_whiff_fit.py -- coefficient for the Whiff% "contact-skill prior" term in the hitter K% projection, in raw units, and a re-check of the K% history weights.

Within-season PA split (fl_hitter_k_study.py design: hitter-seasons with >= 150 PA, known sample m = 30 / 60 / 120 / 200 PA, target = K% over the rest, standard inverse-variance
weights): y - base = b0 + B x (shrunk Whiff% - league Whiff%) x n0w / (m + n0w), Whiff% = whiffs per swing shrunk by n0 = 52.5 swings; base = lg + m (K%_known - lg)/(m + 79.3).
Pooled over the four sample sizes with an intercept (fair reference); bootstrap over hitter-seasons for the standard error.  Part 2: leave-one-transition-out K% RMSE for prior-season weights d.

Usage:  python fl_hitter_whiff_fit.py
"""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_baseline_backtest as bt
import fl_hitter_k_study as ks
import fl_hitter_model_dev as dev

warnings.filterwarnings("ignore")
rng = np.random.default_rng(9)


def main():
    a = ks.pa_table(); s = ks.season_stats(a)
    with contextlib.redirect_stdout(io.StringIO()):
        n0 = ks.partA(a, s)
    lg25 = s[s.year == 2025]["whiff"].sum() / s[s.year == 2025]["swings"].sum(); lg26 = s[s.year == 2026]["whiff"].sum() / s[s.year == 2026]["swings"].sum()
    print(f"n0 (swings) for Whiff%: {n0['Whiff%']:.1f} | league Whiff% 2025 {lg25:.4f}, 2026 {lg26:.4f}")
    D = ks.draws(a, s, n0)
    Xs, Ys, Ws, G, M = [], [], [], [], []
    for m in sorted(D["m"].unique()):
        S = D[D["m"] == m].reset_index(drop=True)
        y = S["y"].to_numpy(); base = (S["lg_k"] + m * (S["k_in"] - S["lg_k"]) / (m + ks.N0_K)).to_numpy()
        wt, _, _ = ew.weights_from_base(y, base, S["n_out"].to_numpy())
        x = np.where(S["Whiff%"].isna(), 0.0, S["Whiff%"].to_numpy()) * ks.N0W / (m + ks.N0W)
        Xs.append(np.column_stack([np.ones(len(y)), x])); Ys.append(y - base); Ws.append(wt); G.append((S["name_clean"] + "|" + S["year"].astype(str)).to_numpy()); M.append(np.full(len(y), m))
    X = np.vstack(Xs); Y = np.concatenate(Ys); W = np.concatenate(Ws); G = np.concatenate(G); M = np.concatenate(M)
    fit = lambda idx: np.linalg.lstsq(X[idx] * np.sqrt(W[idx])[:, None], Y[idx] * np.sqrt(W[idx]), rcond=None)[0]
    b = fit(np.arange(len(Y)))
    ug = pd.Series(G).unique(); by = pd.Series(np.arange(len(G))).groupby(G).apply(np.array)
    bs = np.array([fit(np.concatenate([by[u] for u in rng.choice(ug, len(ug))])) for _ in range(300)])
    print(f"pooled rows {len(Y):,}: intercept {b[0]:+.4f}; B = {b[1]:.4f} K% per unit of shrunk Whiff% deviation (SE {bs[:,1].std():.4f}) -> 1 point of Whiff% (0.01) = {b[1]*0.01*100:.2f} points of K% at zero sample")
    for m in sorted(set(M)):
        print(f"   m={int(m):3d}: B = {fit(np.where(M == m)[0])[1]:.4f}")
    sd_w = s[(s["pa"] >= 100)]["Whiff%"].std()
    print(f"   one SD of hitter Whiff% ({sd_w:.3f}) = {b[1]*sd_w*100:+.2f} points of K% before the fade (a hitter with 300 PA of evidence: x{ks.N0W/(300+ks.N0W):.2f})")

    print("\nK% history weights: leave-one-transition-out RMSE of the regression by prior-season weight d (standard weights)")
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]
    d = bt.transitions(q, "k")
    rows = []
    for delta in (0.25, 0.5, 0.75, 1.0):
        E, n0s = bt.loto(d, delta)
        rows.append({"d (weight on the season before)": delta, "RMSE": bt.rmse(E, E["pred"]), "pooled n0": bt.fit_n0(d, delta)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
