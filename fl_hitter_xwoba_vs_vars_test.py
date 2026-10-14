"""
fl_hitter_xwoba_vs_vars_test.py -- do PullAir%, OppoGB%, Barrel%, Max EV and EV90 beat (or add to) the xwOBA nudge, and do they help more when the
hitter's prior sample is small?

Base = the shipped regressed wOBA blend (no Trackman).  Add-ons, all cross-validated (10-fold x 20 repeats) on 2025 -> 2026 and scored with the standard
inverse-variance weights (fl_eval_weights):
    xwOBA nudge      base + g * (shrunk xwOBA - base)                      (shipped)
    five variables   base + b0 + sum b_j z_j        z_j = the variable shrunk by its own reliability, standardised
    xwOBA + five     both
    ... each also with an interaction  z_j x (1 - r),  r = PA / (PA + 440) = how much the hitter's own wOBA is trusted: "do the skills matter more when
    the outcomes are thin?"
Reported overall and by prior-season sample size (PA in 2025: 20-99, 100-249, 250+), with paired-bootstrap intervals, then the reverse direction
(2026 -> 2025).  Cells are small (about 35-60 hitters), so differences below ~.0003 are noise.

Usage:  python fl_hitter_xwoba_vs_vars_test.py
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import fl_eval_weights as ew
import fl_hitter_contact_vars_test as cv
import fl_hitter_wobacon_test as wc

warnings.filterwarnings("ignore")
V5 = ["pull_air", "oppo_gb", "barrel", "maxev", "ev90"]
GROUPS = [("prior PA 20-99", 20, 100), ("prior PA 100-249", 100, 250), ("prior PA 250+", 250, 10_000)]
N0_WOBA = 440.0


def cv_predict(y, base, X, wt, reps=20):
    """Cross-validated base + X b (OLS with intercept on the base's miss, weighted); returns the averaged out-of-fold prediction."""
    if X is None:
        return base.copy()
    pr = np.zeros(len(y))
    for rep in range(reps):
        for tri, tei in KFold(10, shuffle=True, random_state=rep).split(X):
            A = np.column_stack([np.ones(len(tri)), X[tri]])
            b = np.linalg.lstsq(A * np.sqrt(wt[tri])[:, None], (y - base)[tri] * np.sqrt(wt[tri]), rcond=None)[0]
            pr[tei] += (base[tei] + np.column_stack([np.ones(len(tei)), X[tei]]) @ b) / reps
    return pr


def cv_nudge(y, base, z, wt, reps=20):
    pr = np.zeros(len(y)); gs = []
    for rep in range(reps):
        for tri, tei in KFold(10, shuffle=True, random_state=rep).split(z):
            g = np.sum(wt[tri] * z[tri] * (y - base)[tri]) / np.sum(wt[tri] * z[tri] ** 2); gs.append(g)
            pr[tei] += (base[tei] + g * z[tei]) / reps
    return pr, float(np.mean(gs))


def rmse(y, p, wt):
    return float(np.sqrt(np.average((y - p) ** 2, weights=wt)))


def build(J, q, s, n0, year_in=2025, min_bbe=30):
    T = J[J["t"] == year_in].copy()
    a = s[(s["year"] == year_in) & (s["n"] >= min_bbe)].set_index("name_clean")
    T = T.join(a[["n"] + list(cv.VARS)].rename(columns={"n": "bbe"}), on="name_clean").dropna(subset=["pull_air"])
    qi = q[q["year"] == year_in].set_index("name_clean")
    T["xw"] = T["name_clean"].map(qi["xwoba"]); T["pa_in"] = T["name_clean"].map(qi["pa"])
    T = T.dropna(subset=["xw"])
    T = T[T["pa_in"] >= 20]
    lgxw = np.average(T["xw"], weights=T["pa_in"])
    T["xw_s"] = (T["pa_in"] * T["xw"] + 200 * lgxw) / (T["pa_in"] + 200)
    Z = {}
    for v in cv.VARS:
        x = T[v].to_numpy(float); mu = np.average(x, weights=T["bbe"])
        sh = T["bbe"].to_numpy() / (T["bbe"].to_numpy() + n0[v]) if v in cv.RATE_VARS else np.ones(len(T))
        z = (x - mu) * sh; Z[v] = z / z.std()
    return T, Z


def run(T, Z, base, y, wt, title):
    rel = (T["pa_in"] / (T["pa_in"] + N0_WOBA)).to_numpy()
    z5 = np.column_stack([Z[v] for v in V5])
    z5i = np.column_stack([z5, z5 * (1 - rel)[:, None]])
    zx = (T["xw_s"].to_numpy() - base)
    preds = {"base only": base}
    preds["xwOBA nudge (shipped)"], g = cv_nudge(y, base, zx, wt)
    preds["five variables"] = cv_predict(y, base, z5, wt)
    preds["EV90 + Barrel% only"] = cv_predict(y, base, np.column_stack([Z["ev90"], Z["barrel"]]), wt)
    preds["xwOBA nudge + five variables"] = cv_predict(y, base, np.column_stack([zx, z5]), wt)
    preds["five variables x (1 - reliability)"] = cv_predict(y, base, z5i, wt)
    preds["xwOBA nudge x (1 - reliability)"] = cv_predict(y, base, np.column_stack([zx, zx * (1 - rel)]), wt)
    print(f"\n{title}: n = {len(y)}   (xwOBA nudge gamma {g:.2f})")
    grp = np.zeros(len(y), dtype=int)
    for i, (lab, lo, hi) in enumerate(GROUPS):
        grp[(T["pa_in"].to_numpy() >= lo) & (T["pa_in"].to_numpy() < hi)] = i + 1
    rows = []
    for k, p in preds.items():
        r = {"model": k, "RMSE all": rmse(y, p, wt), "change vs base": rmse(y, p, wt) - rmse(y, base, wt)}
        for i, (lab, lo, hi) in enumerate(GROUPS):
            m = grp == i + 1
            r[f"{lab} (n={m.sum()})"] = rmse(y[m], p[m], wt[m]) - rmse(y[m], base[m], wt[m]) if m.sum() > 5 else np.nan
        rows.append(r)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    rng = np.random.default_rng(0)
    print("   paired bootstrap, change in RMSE (negative = better), 90% interval:")
    for a_, b_ in (("xwOBA nudge (shipped)", "base only"), ("five variables", "base only"), ("five variables", "xwOBA nudge (shipped)"), ("xwOBA nudge + five variables", "xwOBA nudge (shipped)")):
        d = []
        for _ in range(1500):
            i = rng.integers(0, len(y), len(y)); d.append(rmse(y[i], preds[a_][i], wt[i]) - rmse(y[i], preds[b_][i], wt[i]))
        print(f"     {a_:32s} vs {b_:24s}: {np.mean(d):+.5f}  [{np.percentile(d,5):+.5f}, {np.percentile(d,95):+.5f}]")
    # by group, the two headline comparisons
    for lab, a_, b_ in (("five variables vs xwOBA nudge", "five variables", "xwOBA nudge (shipped)"),):
        out = []
        for i, (glab, lo, hi) in enumerate(GROUPS):
            m = np.where(grp == i + 1)[0]
            d = []
            for _ in range(1500):
                j = rng.choice(m, len(m)); d.append(rmse(y[j], preds[a_][j], wt[j]) - rmse(y[j], preds[b_][j], wt[j]))
            out.append(f"{glab}: {np.mean(d):+.5f} [{np.percentile(d,5):+.5f}, {np.percentile(d,95):+.5f}]")
        print(f"   {lab}, by group -> " + "; ".join(out))


def main():
    d = cv.flags(pd.read_parquet(cv.BBE)); s = cv.season_vars(d)
    with contextlib.redirect_stdout(io.StringIO()):
        n0, _ = cv.partA(d, s)
        J, q, xc, w = wc.main()
    T, Z = build(J, q, s, n0)
    base = T["shipped_blend"].to_numpy(); y = T["x_next"].to_numpy(); wt = T["w"].to_numpy()
    run(T, Z, base, y, wt, "FORWARD 2025 -> 2026 (prior PA >= 20, balls tracked >= 30; next-year PA >= 100)")

    # reverse: 2026 -> 2025
    q26 = q[q["year"] == 2026].set_index("name_clean"); q25 = q[q["year"] == 2025].set_index("name_clean")
    a = s[(s["year"] == 2026) & (s["n"] >= 30)].set_index("name_clean")
    R = a[["n"] + list(cv.VARS)].rename(columns={"n": "bbe"}).join(q26[["woba_n", "pa", "xwoba"]].rename(columns={"woba_n": "w_in", "pa": "pa_in", "xwoba": "xw"})) \
        .join(q25[["woba_n", "pa"]].rename(columns={"woba_n": "x_next", "pa": "n_next"}), how="inner").dropna()
    R = R[(R["pa_in"] >= 20) & (R["n_next"] >= 100)].reset_index()
    lg_in = np.average(R["w_in"], weights=R["pa_in"]); lg_out = np.average(R["x_next"], weights=R["n_next"])
    base_r = (lg_out + (R["pa_in"] * (R["w_in"] - lg_in)) / (R["pa_in"] + N0_WOBA)).to_numpy(); y_r = R["x_next"].to_numpy()
    wt_r, _a, _b = ew.weights_from_base(y_r, base_r, R["n_next"].to_numpy())
    lgxw = np.average(R["xw"], weights=R["pa_in"]); R["xw_s"] = (R["pa_in"] * R["xw"] + 200 * lgxw) / (R["pa_in"] + 200)
    Zr = {}
    for v in cv.VARS:
        x = R[v].to_numpy(float); mu = np.average(x, weights=R["bbe"]); sh = R["bbe"].to_numpy() / (R["bbe"].to_numpy() + n0[v]) if v in cv.RATE_VARS else np.ones(len(R))
        z = (x - mu) * sh; Zr[v] = z / z.std()
    run(R, Zr, base_r, y_r, wt_r, "REVERSE 2026 -> 2025 (simple base: own wOBA regressed, n0 440)")


if __name__ == "__main__":
    main()
