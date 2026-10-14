"""
fl_hitter_forward_validation.py -- a clean, non-leaky forward test of the three Trackman/pitch-level terms (EV90 + xwOBA nudge for wOBA, Whiff% for K%, Swing% / Chase% for BB%).

Why: the reverse direction (2026 -> 2025) predicts the past from the future, and the earlier forward tests used coefficients fit on within-season draws of BOTH years, i.e. including 2026 outcomes.
Design:
  Stage 1  coefficients are fit ONLY on 2025 data: within-season PA splits of 2025 hitter-seasons (known sample m = 30 / 60 / 120 / 200 PA -> outcome over the rest), pooled over m, with an intercept,
           exactly the production forms (informed-prior fade n0w / (PA_eff + n0w); Whiff% / Swing% / Chase% shrunk by their own reliability; EV90 shrunk n0 14.26 balls).  Variable league levels from 2025 only.
  Stage 2  the frozen coefficients are applied to 2025 -> 2026 on the production bases (K%: two seasons, n0 79.3, d .75; BB%: n0 146.6, d .5; wOBA: the shipped blend); 2026 outcomes are used only to score.
  Bootstrap (5,000 replicates) resamples BOTH stages: the 2025 hitter-seasons that fit the coefficient and the hitters being forecast.  Each model gets its own constant fit on the same resample (fair reference).
  Reported: mean change in RMSE vs the base, 90% / 95% interval, and the share of replicates in which the term helps.  Also a thin-history cut (prior-season PA < 150, next-year PA >= 50).

Usage:  python fl_hitter_forward_validation.py
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_baseline_backtest as bt
import fl_hitter_bb_study as bb
import fl_hitter_contact_vars_test as cv
import fl_hitter_ev as hev
import fl_hitter_k_study as ks
import fl_hitter_model_dev as dev
import fl_hitter_pa_split_test as ps
import fl_hitter_wobacon_test as wc

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026)
R_FIT, R_EVAL = 300, 5000
N0K, N0BB, N0W = 79.3, 146.6, 440.0
SIZES = (30, 60, 120, 200)


def wfit(X, Y, W, idx):
    return np.linalg.lstsq(X[idx] * np.sqrt(W[idx])[:, None], Y[idx] * np.sqrt(W[idx]), rcond=None)[0]


def boot_fit(X, Y, W, G, R=R_FIT):
    """Coefficient replicates: resample hitter-seasons (clusters) and refit."""
    ug = pd.Series(G).unique(); by = pd.Series(np.arange(len(G))).groupby(G).apply(np.array)
    full = wfit(X, Y, W, np.arange(len(Y)))
    reps = np.array([wfit(X, Y, W, np.concatenate([by[u] for u in rng.choice(ug, len(ug))])) for _ in range(R)])
    return full, reps


def stage1_k(a25, s25, n0):
    D = ks.draws(a25, s25, n0)
    Xs, Ys, Ws, G = [], [], [], []
    for m in SIZES:
        S = D[D["m"] == m]; y = S["y"].to_numpy(); base = (S["lg_k"] + m * (S["k_in"] - S["lg_k"]) / (m + N0K)).to_numpy()
        wt, _, _ = ew.weights_from_base(y, base, S["n_out"].to_numpy())
        Xs.append(np.column_stack([np.ones(len(y)), np.where(S["Whiff%"].isna(), 0.0, S["Whiff%"].to_numpy()) * N0K / (m + N0K)])); Ys.append(y - base); Ws.append(wt); G.append((S["name_clean"]).to_numpy())
    return boot_fit(np.vstack(Xs), np.concatenate(Ys), np.concatenate(Ws), np.concatenate(G)), D["name_clean"].nunique()


def stage1_bb(a25, s25, n0, var):
    D = bb.draws(a25, s25, n0)
    Xs, Ys, Ws, G = [], [], [], []
    for m in SIZES:
        S = D[D["m"] == m]; y = S["y"].to_numpy(); base = (S["lg_bb"] + m * (S["bb_in"] - S["lg_bb"]) / (m + N0BB)).to_numpy()
        wt, _, _ = ew.weights_from_base(y, base, S["n_out"].to_numpy())
        cols = [np.where(S[v].isna(), 0.0, S[v].to_numpy()) * N0BB / (m + N0BB) for v in var]
        Xs.append(np.column_stack([np.ones(len(y))] + cols)); Ys.append(y - base); Ws.append(wt); G.append(S["name_clean"].to_numpy())
    return boot_fit(np.vstack(Xs), np.concatenate(Ys), np.concatenate(Ws), np.concatenate(G)), D["name_clean"].nunique()


def stage1_woba(d25):
    ps.SIZES = SIZES
    D = ps.draws(d25)
    t = d25[d25["tracked"]].groupby(["batter", "year"])["ev"].agg(n="size", ev90=lambda s: np.quantile(s, 0.9)).reset_index()
    lg_ev = float(t[t["n"] >= 50]["ev90"].mean())
    Xs, Ys, Ws, G = [], [], [], []
    for m in SIZES:
        S = D[D["m"] == m].reset_index(drop=True); y = S["y"].to_numpy(); base = (S["lg_w"] + m * (S["w_in"] - S["lg_w"]) / (m + N0W)).to_numpy()
        wt, _, _ = ew.weights_from_base(y, base, S["n_out"].to_numpy())
        xs = (m * S["xw_in"] + ps.N0_X * S["lg_x"]) / (m + ps.N0_X); zx = xs.to_numpy() - base
        x = np.where(S["ev90"].isna(), 0.0, (S["ev90"].to_numpy() - lg_ev) * S["bbe"].to_numpy() / (S["bbe"].to_numpy() + hev.N0_EV90)) * N0W / (m + N0W)
        Xs.append(np.column_stack([np.ones(len(y)), zx, x])); Ys.append(y - base); Ws.append(wt); G.append((S["batter"] + "|" + S["year"].astype(str)).to_numpy())
    return boot_fit(np.vstack(Xs), np.concatenate(Ys), np.concatenate(Ws), np.concatenate(G)), D["batter"].nunique(), lg_ev


def rm(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def evaluate(y, base, terms_full, terms_fn, reps, w, label, n_eval=R_EVAL):
    """terms_fn(coef_vector) -> additive term per row (n,). Constant fit on each resample for BOTH the model and the reference."""
    n = len(y)
    def delta(idx, coef):
        t = terms_fn(coef)[idx]; r0 = base[idx]; m1 = r0 + t
        c0 = np.average(y[idx] - r0, weights=w[idx]); c1 = np.average(y[idx] - m1, weights=w[idx])
        return rm(y[idx], m1 + c1, w[idx]) - rm(y[idx], r0 + c0, w[idx])
    point = delta(np.arange(n), terms_full)
    ds = np.empty(n_eval)
    for i in range(n_eval):
        idx = rng.integers(0, n, n); ds[i] = delta(idx, reps[rng.integers(0, len(reps))])
    return {"term": label, "n": n, "point (frozen coef)": point, "bootstrap mean": ds.mean(), "90% interval": f"[{np.percentile(ds,5):+.5f}, {np.percentile(ds,95):+.5f}]",
            "95% interval": f"[{np.percentile(ds,2.5):+.5f}, {np.percentile(ds,97.5):+.5f}]", "P(term helps)": float((ds < 0).mean())}


def main():
    # ---------------------------------------------------------------- data
    a = ks.pa_table(); s = ks.season_stats(a)
    with contextlib.redirect_stdout(io.StringIO()):
        n0k = ks.partA(a, s); n0bb = bb.partA(a, s)
    a25, s25 = a[a["year"] == 2025], s[s["year"] == 2025]
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]
    out = []

    # ---------------------------------------------------------------- K%: Whiff%
    (full, reps), nfit = stage1_k(a25, s25, n0k)
    print(f"Stage 1 (2025 only), K% Whiff% term: B = {full[1]:.3f} (bootstrap SE {reps[:,1].std():.3f}; {nfit} hitter-seasons in the fit); production value fit on 2025-26 was 0.605")
    for label, minp, minn, thin in (("all hitters (2025 PA >= 20, 2026 PA >= 100)", 20, 100, None), ("thin history (2025 PA 10-149, 2026 PA >= 50)", 10, 50, 150)):
        bt.MIN_NEXT = minn
        d = bt.transitions(q, "k"); bt.MIN_NEXT = 100
        d = d[(d["t"] == 2025) & (d["n_t"] >= minp)].copy()
        if thin:
            d = d[d["n_t"] < thin]
        base = bt.predict(d, N0K, 0.75).to_numpy(); y = d["x_next"].to_numpy(); w = d["w"].to_numpy(); pa_eff = (d["n_t"] + 0.75 * d["n_prev"]).to_numpy()
        st = s25.set_index("name_clean").reindex(d["name_clean"])
        lg_wh = s25["whiff"].sum() / s25["swings"].sum()
        sw = st["swings"].to_numpy(float); xw = st["Whiff%"].to_numpy(float); dev_ = np.where(np.isnan(xw), 0.0, (xw - lg_wh) * sw / (sw + n0k["Whiff%"]))
        fn = lambda coef, dev_=dev_, pa_eff=pa_eff: coef[1] * dev_ * N0K / (pa_eff + N0K)
        r = evaluate(y, base, full, fn, reps, w, f"K%: Whiff% | {label}", R_EVAL); out.append(r)

    # ---------------------------------------------------------------- BB%: Swing%, Chase%, both
    for var in (["Swing%"], ["Chase%"], ["Swing%", "Chase%"]):
        (full, reps), nfit = stage1_bb(a25, s25, n0bb, var)
        print(f"Stage 1 (2025 only), BB% {' + '.join(var)}: B = {np.round(full[1:], 3).tolist()} (bootstrap SE {np.round(reps[:,1:].std(axis=0), 3).tolist()}; {nfit} hitters)")
        for label, minp, minn, thin in (("all hitters", 20, 100, None), ("thin history", 10, 50, 150)):
            bt.MIN_NEXT = minn
            d = bt.transitions(q, "bbr"); bt.MIN_NEXT = 100
            d = d[(d["t"] == 2025) & (d["n_t"] >= minp)].copy()
            if thin:
                d = d[d["n_t"] < thin]
            base = bt.predict(d, N0BB, 0.5).to_numpy(); y = d["x_next"].to_numpy(); w = d["w"].to_numpy(); pa_eff = (d["n_t"] + 0.5 * d["n_prev"]).to_numpy()
            st = s25.set_index("name_clean").reindex(d["name_clean"]); cols = []
            for v in var:
                nu, de = bb.VARS[v]; lgv = s25[nu].sum() / s25[de].sum(); den = st[de].to_numpy(float); x = st[v].to_numpy(float)
                cols.append(np.where(np.isnan(x), 0.0, (x - lgv) * den / (den + n0bb[v])))
            fn = lambda coef, cols=cols, pa_eff=pa_eff: sum(coef[1 + i] * cols[i] for i in range(len(cols))) * N0BB / (pa_eff + N0BB)
            out.append(evaluate(y, base, full, fn, reps, w, f"BB%: {' + '.join(var)} | {label}", R_EVAL))

    # ---------------------------------------------------------------- wOBA: nudge + EV90
    d25 = ps.prep(); d25 = d25[d25["year"] == 2025]
    (full, reps), nfit, lg_ev = stage1_woba(d25)
    print(f"Stage 1 (2025 only), wOBA nudge + EV90: gamma = {full[1]:.3f} (SE {reps[:,1].std():.3f}), B_EV90 = {full[2]:.5f} per mph (SE {reps[:,2].std():.5f}); {nfit} hitters; production (2025-26): .485 / .00377")
    with contextlib.redirect_stdout(io.StringIO()):
        J, q2, xc, _w = wc.main()
        d2 = cv.flags(pd.read_parquet(cv.BBE)); s2 = cv.season_vars(d2)
    T = J[J["t"] == 2025].copy()
    dd = bt.transitions(q, "woba"); dd = dd[dd["t"] == 2025][["name_clean", "n_t", "n_prev"]]
    T = T.merge(dd, on="name_clean", how="left")
    a_ = s2[s2["year"] == 2025].set_index("name_clean")[["n", "ev90"]].rename(columns={"n": "bbe", "ev90": "ev25"})
    qi = q2[q2["year"] == 2025].set_index("name_clean")
    T = T.join(a_, on="name_clean"); T["xw25"] = T["name_clean"].map(qi["xwoba"]); T["pa25"] = T["name_clean"].map(qi["pa"]); T = T.dropna(subset=["xw25"])
    lgx = np.average(T["xw25"], weights=T["pa25"]); xs = ((T["pa25"] * T["xw25"] + ps.N0_X * lgx) / (T["pa25"] + ps.N0_X)).to_numpy()
    base = T["shipped_blend"].to_numpy(); y = T["x_next"].to_numpy(); w = T["w"].to_numpy(); pa_eff = (T["n_t"] + 0.75 * T["n_prev"]).to_numpy()
    bbe = T["bbe"].fillna(0).to_numpy(); ev = np.where(T["ev25"].isna(), 0.0, (T["ev25"].to_numpy() - lg_ev) * bbe / (bbe + hev.N0_EV90))
    zx = xs - base
    out.append(evaluate(y, base, full, lambda c: c[1] * zx + c[2] * ev * N0W / (pa_eff + N0W), reps, w, "wOBA: xwOBA nudge + EV90 | all hitters (2025 PA >= 50, 2026 PA >= 100)", R_EVAL))
    out.append(evaluate(y, base, full, lambda c: c[2] * ev * N0W / (pa_eff + N0W), reps, w, "wOBA: EV90 alone (nudge off) | all hitters", R_EVAL))
    out.append(evaluate(y, base, full, lambda c: c[1] * zx, reps, w, "wOBA: xwOBA nudge alone | all hitters", R_EVAL))

    R = pd.DataFrame(out)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print("\nFORWARD VALIDATION 2025 -> 2026, coefficients fit on 2025 only, 5,000 two-stage bootstrap replicates; change in RMSE vs the production base (negative = the term helps)")
    print(R.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
