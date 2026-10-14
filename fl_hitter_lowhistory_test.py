"""
fl_hitter_lowhistory_test.py -- for hitters with LITTLE prior Frontier League history, does xwOBA (or a subset of PullAir/OppoGB/Barrel/MaxEV/EV90) help,
and does either beat the other?

History = total FL plate appearances in every season up to the input year (2022-26 box-score lines + 2025-26 Trackman), not just the last season.
Groups: < 100 PA, 100-299, 300+.  To get usable thin groups the target only needs 30 PA next year (the earlier tests required 100 and left 8-11 hitters in the
thin cell); the standard inverse-variance weights (fl_eval_weights) already discount noisy small-PA targets.  Base = regressed wOBA blend (LOTO, rebuilt with the
30-PA floor).  Add-ons are fit on ALL hitters (cross-validated) and scored by group; both directions (2025 -> 2026, 2026 -> 2025) are shown and pooled.

Usage:  python fl_hitter_lowhistory_test.py
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_baseline_backtest as bt
import fl_hitter_contact_vars_test as cv
import fl_hitter_wobacon_test as wc
import fl_hitter_xwoba_vs_vars_test as xv

warnings.filterwarnings("ignore")
GROUPS = [("<100 prior FL PA", 0, 100), ("100-299", 100, 300), ("300+", 300, 10 ** 6)]
MODELS = {"xwOBA nudge (shipped)": None, "EV90": ["ev90"], "EV90 + PullAir": ["ev90", "pull_air"], "Barrel + PullAir": ["barrel", "pull_air"],
          "OppoGB + MaxEV": ["oppo_gb", "maxev"], "all five": ["pull_air", "oppo_gb", "barrel", "maxev", "ev90"]}
GAMMA = 0.57


def hist_pa(q, year):
    g = q[q["year"] <= year].groupby("name_clean")["pa"].sum()
    return g


def run_dir(label, T, Z, base, y, wt, hist):
    pr = {"base": base}
    zx = T["xw_s"].to_numpy() - base
    pr["xwOBA nudge (shipped)"], g = xv.cv_nudge(y, base, zx, wt)
    for k, cols in MODELS.items():
        if cols is None:
            continue
        pr[k] = xv.cv_predict(y, base, np.column_stack([Z[c] for c in cols]), wt)
    pr["xwOBA nudge + EV90 + PullAir"] = xv.cv_predict(y, base, np.column_stack([zx, Z["ev90"], Z["pull_air"]]), wt)
    grp = np.zeros(len(y), dtype=int)
    for i, (_, lo, hi) in enumerate(GROUPS):
        grp[(hist >= lo) & (hist < hi)] = i + 1
    rows = []
    for k, p in pr.items():
        r = {"model": k}
        for i, (lab, lo, hi) in enumerate(GROUPS):
            m = grp == i + 1
            if m.sum() < 3:
                r[f"{lab} (n={m.sum()})"] = np.nan; continue
            r[f"{lab} (n={m.sum()})"] = xv.rmse(y[m], p[m], wt[m]) - xv.rmse(y[m], base[m], wt[m]) if k != "base" else xv.rmse(y[m], base[m], wt[m])
        r["all"] = xv.rmse(y, p, wt) - xv.rmse(y, base, wt) if k != "base" else xv.rmse(y, base, wt)
        rows.append(r)
    print(f"\n{label}: n = {len(y)}  (nudge gamma {g:.2f}); 'base' row = RMSE level, other rows = change in RMSE vs base (negative = better)")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    return pr, grp


def main():
    bt.MIN_NEXT = 30                                   # thin groups need thin targets; the inverse-variance weights handle the noise
    d = cv.flags(pd.read_parquet(cv.BBE)); s = cv.season_vars(d)
    with contextlib.redirect_stdout(io.StringIO()):
        n0, _ = cv.partA(d, s)
        J, q, xc, w = wc.main()
    # ---- forward ----------------------------------------------------------------------------------------------------
    T = J[J["t"] == 2025].copy()
    a = s[(s["year"] == 2025) & (s["n"] >= 15)].set_index("name_clean")
    T = T.join(a[["n"] + list(cv.VARS)].rename(columns={"n": "bbe"}), on="name_clean").dropna(subset=["pull_air"])
    qi = q[q["year"] == 2025].set_index("name_clean")
    T["xw"] = T["name_clean"].map(qi["xwoba"]); T["pa_in"] = T["name_clean"].map(qi["pa"]); T = T.dropna(subset=["xw"]); T = T[T["pa_in"] >= 10]
    lgx = np.average(T["xw"], weights=T["pa_in"]); T["xw_s"] = (T["pa_in"] * T["xw"] + 200 * lgx) / (T["pa_in"] + 200)
    ZF = {}
    for v in cv.VARS:
        x = T[v].to_numpy(float); mu = np.average(x, weights=T["bbe"]); sh = T["bbe"].to_numpy() / (T["bbe"].to_numpy() + n0[v]) if v in cv.RATE_VARS else np.ones(len(T))
        z = (x - mu) * sh; ZF[v] = z / z.std()
    yF = T["x_next"].to_numpy(); bF = T["shipped_blend"].to_numpy(); wF = T["w"].to_numpy(); hF = T["name_clean"].map(hist_pa(q, 2025)).to_numpy()
    prF, gF = run_dir("FORWARD 2025 -> 2026 (history through 2025)", T, ZF, bF, yF, wF, hF)

    # ---- reverse -----------------------------------------------------------------------------------------------------
    q26 = q[q["year"] == 2026].set_index("name_clean"); q25 = q[q["year"] == 2025].set_index("name_clean")
    a = s[(s["year"] == 2026) & (s["n"] >= 15)].set_index("name_clean")
    R = a[["n"] + list(cv.VARS)].rename(columns={"n": "bbe"}).join(q26[["woba_n", "pa", "xwoba"]].rename(columns={"woba_n": "w_in", "pa": "pa_in", "xwoba": "xw"})) \
        .join(q25[["woba_n", "pa"]].rename(columns={"woba_n": "x_next", "pa": "n_next"}), how="inner").dropna()
    R = R[(R["pa_in"] >= 10) & (R["n_next"] >= 30)].reset_index()
    lg_in = np.average(R["w_in"], weights=R["pa_in"]); lg_out = np.average(R["x_next"], weights=R["n_next"])
    bR = (lg_out + (R["pa_in"] * (R["w_in"] - lg_in)) / (R["pa_in"] + 440.0)).to_numpy(); yR = R["x_next"].to_numpy()
    wR, _a, _b = ew.weights_from_base(yR, bR, R["n_next"].to_numpy())
    lgx = np.average(R["xw"], weights=R["pa_in"]); R["xw_s"] = (R["pa_in"] * R["xw"] + 200 * lgx) / (R["pa_in"] + 200)
    ZR = {}
    for v in cv.VARS:
        x = R[v].to_numpy(float); mu = np.average(x, weights=R["bbe"]); sh = R["bbe"].to_numpy() / (R["bbe"].to_numpy() + n0[v]) if v in cv.RATE_VARS else np.ones(len(R))
        z = (x - mu) * sh; ZR[v] = z / z.std()
    hR = R["pa_in"].to_numpy()      # reverse: the "history" is the 2026 sample the projection is built from (earlier years are the target)
    prR, gR = run_dir("REVERSE 2026 -> 2025 (groups by 2026 PA, the sample behind the projection)", R, ZR, bR, yR, wR, hR)

    # ---- pooled thin-history comparison with bootstrap ----------------------------------------------------------------------------
    print("\nPOOLED (forward + reverse), hitters with < 100 prior FL PA: paired bootstrap of the RMSE change (90% interval)")
    mF = gF == 1; mR = gR == 1
    y = np.concatenate([yF[mF], yR[mR]]); wt = np.concatenate([wF[mF], wR[mR]])
    P = {k: np.concatenate([prF[k][mF], prR[k][mR]]) for k in prF}
    rng = np.random.default_rng(0); n = len(y)
    print(f"   pooled n = {n} (forward {int(mF.sum())}, reverse {int(mR.sum())})")
    for a_, b_ in (("xwOBA nudge (shipped)", "base"), ("EV90 + PullAir", "base"), ("EV90", "base"), ("all five", "base"), ("EV90 + PullAir", "xwOBA nudge (shipped)"),
                   ("all five", "xwOBA nudge (shipped)"), ("OppoGB + MaxEV", "xwOBA nudge (shipped)"), ("xwOBA nudge + EV90 + PullAir", "xwOBA nudge (shipped)")):
        dd = []
        for _ in range(2000):
            i = rng.integers(0, n, n); dd.append(xv.rmse(y[i], P[a_][i], wt[i]) - xv.rmse(y[i], P[b_][i], wt[i]))
        print(f"   {a_:30s} vs {b_:24s}: {np.mean(dd):+.5f}  [{np.percentile(dd,5):+.5f}, {np.percentile(dd,95):+.5f}]")


if __name__ == "__main__":
    main()
