"""
fl_hitter_subset_test.py -- PullAir%, OppoGB%, Barrel%, Max EV and EV90 overlap; which SUBSET (if any) adds next-year information?

1  Redundancy: correlations among the five (hitter-seasons with >= 80 tracked balls).
2  All 31 subsets, cross-validated, in BOTH directions (2025 -> 2026 and 2026 -> 2025), as an add-on to (a) the regressed wOBA blend (replacing xwOBA) and
   (b) the shipped xwOBA nudge (adding to it).  Ranked by the average of the two directions; with 31 looks, the best row is optimistic.
3  Honest selection: nested cross-validation -- the subset is chosen inside each training fold (inner 5-fold CV over all 31) and scored on the held-out fold.
4  The same for next-year HR/PA (which drives the Mississippi park fit).
Standard inverse-variance weights throughout (fl_eval_weights).  Usage: python fl_hitter_subset_test.py
"""
from __future__ import annotations

import contextlib
import io
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import fl_eval_weights as ew
import fl_hitter_contact_vars_test as cv
import fl_hitter_wobacon_test as wc
import fl_hitter_xwoba_vs_vars_test as xv

warnings.filterwarnings("ignore")
V5 = ["pull_air", "oppo_gb", "barrel", "maxev", "ev90"]
LAB = {"pull_air": "PullAir", "oppo_gb": "OppoGB", "barrel": "Barrel", "maxev": "MaxEV", "ev90": "EV90"}
SUBSETS = [c for r in range(1, 6) for c in itertools.combinations(V5, r)]
GAMMA = 0.57          # shipped xwOBA nudge


def fit_pred(y, base, X, wt, tr, te):
    A = np.column_stack([np.ones(len(tr)), X[tr]])
    b = np.linalg.lstsq(A * np.sqrt(wt[tr])[:, None], (y - base)[tr] * np.sqrt(wt[tr]), rcond=None)[0]
    return base[te] + np.column_stack([np.ones(len(te)), X[te]]) @ b


def cv_sub(y, base, X, wt, reps=10):
    pr = np.zeros(len(y))
    for rep in range(reps):
        for tr, te in KFold(10, shuffle=True, random_state=rep).split(X):
            pr[te] += fit_pred(y, base, X, wt, tr, te) / reps
    return pr


def rm(y, p, wt):
    return float(np.sqrt(np.average((y - p) ** 2, weights=wt)))


def nested(y, base, Zs, wt, reps=3):
    """Subset chosen by inner 5-fold CV (min weighted MSE over all 31 subsets) inside each outer training fold; scored on the outer test fold."""
    pr = np.zeros(len(y)); chosen = []
    for rep in range(reps):
        for tr, te in KFold(10, shuffle=True, random_state=100 + rep).split(y):
            best, bs = None, np.inf
            for sub in SUBSETS:
                X = Zs[list(sub)].to_numpy()
                e = 0.0
                for itr, ite in KFold(5, shuffle=True, random_state=7).split(tr):
                    a, b = tr[itr], tr[ite]
                    p = fit_pred(y, base, X, wt, a, b)
                    e += np.sum(wt[b] * (y[b] - p) ** 2)
                if e < bs:
                    bs, best = e, sub
            chosen.append(best)
            pr[te] += fit_pred(y, base, Zs[list(best)].to_numpy(), wt, tr, te) / reps
    return pr, chosen


def main():
    d = cv.flags(pd.read_parquet(cv.BBE)); s = cv.season_vars(d)
    with contextlib.redirect_stdout(io.StringIO()):
        n0, _ = cv.partA(d, s)
        J, q, xc, w = wc.main()
    # ---- 1 redundancy ----------------------------------------------------------------------------------------------
    x = s[s["n"] >= 80][V5]
    print("1. REDUNDANCY: correlations among the five (hitter-seasons with >= 80 tracked balls, n=%d)" % len(x))
    print(x.rename(columns=LAB).corr().round(2).rename(columns=LAB).to_string())

    # ---- datasets (wOBA) ---------------------------------------------------------------------------------------------
    T, Z = xv.build(J, q, s, n0)
    yF = T["x_next"].to_numpy(); wF = T["w"].to_numpy(); bF = T["shipped_blend"].to_numpy()
    zxF = T["xw_s"].to_numpy() - bF; b2F = bF + GAMMA * zxF
    q26 = q[q["year"] == 2026].set_index("name_clean"); q25 = q[q["year"] == 2025].set_index("name_clean")
    a = s[(s["year"] == 2026) & (s["n"] >= 30)].set_index("name_clean")
    R = a[["n"] + list(cv.VARS)].rename(columns={"n": "bbe"}).join(q26[["woba_n", "pa", "xwoba", "hrr"]].rename(columns={"woba_n": "w_in", "pa": "pa_in", "xwoba": "xw", "hrr": "hr_in"})) \
        .join(q25[["woba_n", "pa", "hrr"]].rename(columns={"woba_n": "x_next", "pa": "n_next", "hrr": "hr_next"}), how="inner").dropna()
    R = R[(R["pa_in"] >= 20) & (R["n_next"] >= 100)].reset_index()
    lg_in = np.average(R["w_in"], weights=R["pa_in"]); lg_out = np.average(R["x_next"], weights=R["n_next"])
    bR = (lg_out + (R["pa_in"] * (R["w_in"] - lg_in)) / (R["pa_in"] + 440.0)).to_numpy(); yR = R["x_next"].to_numpy()
    wR, _a, _b = ew.weights_from_base(yR, bR, R["n_next"].to_numpy())
    lgx = np.average(R["xw"], weights=R["pa_in"]); R["xw_s"] = (R["pa_in"] * R["xw"] + 200 * lgx) / (R["pa_in"] + 200)
    b2R = bR + GAMMA * (R["xw_s"].to_numpy() - bR)
    ZR = {}
    for v in cv.VARS:
        xx = R[v].to_numpy(float); mu = np.average(xx, weights=R["bbe"]); sh = R["bbe"].to_numpy() / (R["bbe"].to_numpy() + n0[v]) if v in cv.RATE_VARS else np.ones(len(R))
        z = (xx - mu) * sh; ZR[v] = z / z.std()
    ZF = pd.DataFrame({v: Z[v] for v in V5}); ZRd = pd.DataFrame({v: ZR[v] for v in V5})

    # ---- 2 all subsets, both directions ---------------------------------------------------------------------------------
    rows = []
    for sub in SUBSETS:
        r = {"subset": "+".join(LAB[v] for v in sub), "k": len(sub)}
        for tag, (y, base, base2, Zd, wt) in {"fwd": (yF, bF, b2F, ZF, wF), "rev": (yR, bR, b2R, ZRd, wR)}.items():
            X = Zd[list(sub)].to_numpy()
            r[f"{tag} replace xwOBA"] = rm(y, cv_sub(y, base, X, wt), wt) - rm(y, base, wt)
            r[f"{tag} add to xwOBA"] = rm(y, cv_sub(y, base2, X, wt), wt) - rm(y, base2, wt)
        r["avg replace"] = (r["fwd replace xwOBA"] + r["rev replace xwOBA"]) / 2
        r["avg add"] = (r["fwd add to xwOBA"] + r["rev add to xwOBA"]) / 2
        rows.append(r)
    t = pd.DataFrame(rows)
    ref = {"fwd": (rm(yF, b2F, wF) - rm(yF, bF, wF)), "rev": (rm(yR, b2R, wR) - rm(yR, bR, wR))}
    print(f"\n2. ALL 31 SUBSETS, cross-validated (change in RMSE; negative = better).  Reference: the shipped xwOBA nudge alone changes RMSE by {ref['fwd']:+.4f} (2025->26) and {ref['rev']:+.4f} (2026->25)")
    print("   'replace' = subset instead of the nudge (compare with the reference); 'add' = subset on top of the nudge (compare with 0)")
    pd.set_option("display.width", 250)
    cols = ["subset", "fwd replace xwOBA", "rev replace xwOBA", "avg replace", "fwd add to xwOBA", "rev add to xwOBA", "avg add"]
    print("   best 8 as a replacement:"); print(t.sort_values("avg replace").head(8)[cols].round(4).to_string(index=False))
    print("   best 8 as an add-on to xwOBA:"); print(t.sort_values("avg add").head(8)[cols].round(4).to_string(index=False))
    named = t[t["subset"].isin(["EV90", "Barrel", "EV90+PullAir", "Barrel+PullAir", "EV90+OppoGB", "Barrel+OppoGB", "MaxEV"])]
    print("   a few named subsets:"); print(named[cols].round(4).to_string(index=False))
    print(f"   subsets that beat the shipped nudge as a replacement in BOTH directions: {int(((t['fwd replace xwOBA'] < ref['fwd']) & (t['rev replace xwOBA'] < ref['rev'])).sum())} of 31; "
          f"that improve on the nudge when added in BOTH directions: {int(((t['fwd add to xwOBA'] < 0) & (t['rev add to xwOBA'] < 0)).sum())} of 31")

    # ---- 3 nested selection -----------------------------------------------------------------------------------------------
    print("\n3. HONEST SELECTION: subset chosen inside each training fold (nested CV), scored on held-out hitters (change in RMSE vs the base shown)")
    for tag, (y, base, base2, Zd, wt) in {"2025 -> 2026": (yF, bF, b2F, ZF, wF), "2026 -> 2025": (yR, bR, b2R, ZRd, wR)}.items():
        pr, ch = nested(y, base, Zd, wt)
        pr2, ch2 = nested(y, base2, Zd, wt)
        top = pd.Series(["+".join(LAB[v] for v in c) for c in ch]).value_counts().head(3).to_dict()
        print(f"   {tag}: base {rm(y, base, wt):.4f}; xwOBA nudge {rm(y, base2, wt):.4f}; nested-chosen subset REPLACING the nudge {rm(y, pr, wt):.4f} ({rm(y, pr, wt)-rm(y, base, wt):+.4f}); "
              f"chosen ON TOP of the nudge {rm(y, pr2, wt):.4f} ({rm(y, pr2, wt)-rm(y, base2, wt):+.4f}); most often chosen: {top}")

    # ---- 4 HR/PA target ---------------------------------------------------------------------------------------------------
    print("\n4. NEXT-YEAR HR/PA (park-neutral): best single / pair subsets, both directions, base = regressed own HR/PA (n0 137)")
    nx = q26["hrr"]; pa26 = q26["pa"]
    Tf = T.copy(); Tf["hr_in"] = Tf["name_clean"].map(q25["hrr"]); Tf["pa_in"] = Tf["name_clean"].map(q25["pa"])
    Tf["y_hr"] = Tf["name_clean"].map(nx); Tf["n_hr"] = Tf["name_clean"].map(pa26)
    Tf = Tf.dropna(subset=["hr_in", "y_hr", "n_hr"]); Zf = ZF.loc[Tf.index] if False else pd.DataFrame({v: Z[v] for v in V5}, index=T.index).loc[Tf.index]
    lgi = np.average(Tf["hr_in"], weights=Tf["pa_in"]); lgo = np.average(Tf["y_hr"], weights=Tf["n_hr"])
    bf = (lgo + (Tf["pa_in"] * (Tf["hr_in"] - lgi)) / (Tf["pa_in"] + 137.0)).to_numpy(); yf = Tf["y_hr"].to_numpy(); wf, _, _ = ew.weights_from_base(yf, bf, Tf["n_hr"].to_numpy())
    lgi = np.average(R["hr_in"], weights=R["pa_in"]); lgo = np.average(R["hr_next"], weights=R["n_next"])
    br = (lgo + (R["pa_in"] * (R["hr_in"] - lgi)) / (R["pa_in"] + 137.0)).to_numpy(); yr = R["hr_next"].to_numpy(); wr, _, _ = ew.weights_from_base(yr, br, R["n_next"].to_numpy())
    rows = []
    for sub in SUBSETS:
        if len(sub) > 2:
            continue
        r = {"subset": "+".join(LAB[v] for v in sub)}
        r["fwd"] = rm(yf, cv_sub(yf, bf, Zf[list(sub)].to_numpy(), wf), wf) - rm(yf, bf, wf)
        r["rev"] = rm(yr, cv_sub(yr, br, ZRd[list(sub)].to_numpy(), wr), wr) - rm(yr, br, wr)
        r["avg"] = (r["fwd"] + r["rev"]) / 2
        rows.append(r)
    print(f"   n = {len(yf)} (fwd), {len(yr)} (rev); base RMSE {rm(yf, bf, wf):.4f} / {rm(yr, br, wr):.4f}")
    print(pd.DataFrame(rows).sort_values("avg").head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
