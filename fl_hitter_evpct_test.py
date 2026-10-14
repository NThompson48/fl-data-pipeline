"""
fl_hitter_evpct_test.py -- which exit-velocity summary is the best "power prior": the 50th, 60th, ... 90th, 95th, 98th percentile, the single hardest ball, or a hard-hit share?

Same within-season PA-split design as fl_hitter_pa_split_test.py (369 hitter-seasons with >= 150 PA; known sample m = 30 / 60 / 120 / 200 random PA, target = wOBA over the rest,
add-on fit by 10-fold CV grouped by hitter, standard inverse-variance weights, hitter-cluster bootstrap), but with ONE exit-velocity summary at a time.  Each summary is
shrunk by its own reliability: n0 (batted balls) from random-half split reliability of full seasons, so noisy extremes (98th, max) are penalised fairly.
Also reported: full-season reliability, 2025 -> 2026 correlation, and same-season correlation with HR/PA and wOBAcon.

Usage:  python fl_hitter_evpct_test.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import fl_eval_weights as ew
import fl_hitter_pa_split_test as ps

warnings.filterwarnings("ignore")
rng = np.random.default_rng(11)
SUMMARIES = {f"EV{p}": ("q", p / 100) for p in (50, 60, 70, 75, 80, 85, 90, 95, 98)}
SUMMARIES["MaxEV"] = ("q", 1.0)
SUMMARIES["HardHit95%"] = ("hh", 95.0)
SUMMARIES["HardHit100%"] = ("hh", 100.0)


def summ(ev, kind, p):
    if kind == "q":
        return float(np.quantile(ev, p))
    return float((ev >= p).mean())


def reliability_n0(d):
    """Per summary: n0 (balls) from random-half split reliability on full hitter-seasons with >= 60 tracked balls."""
    res = {k: [] for k in SUMMARIES}; half = []
    for (b, y), g in d[d["tracked"]].groupby(["batter", "year"]):
        if len(g) < 60:
            continue
        ev = g["ev"].to_numpy()
        for _ in range(12):
            i = rng.permutation(len(ev)); h = len(ev) // 2; a, o = ev[i[:h]], ev[i[h:2 * h]]
            half.append(h)
            for k, (kind, p) in SUMMARIES.items():
                res[k].append((b, y, summ(a, kind, p), summ(o, kind, p)))
    out = {}
    for k, r in res.items():
        x = pd.DataFrame(r, columns=["b", "y", "a", "o"])
        rr = x["a"].corr(x["o"])
        out[k] = (float(rr), max(np.mean(half) * (1 - rr) / rr, 1.0))
    return out


def main():
    d = ps.prep()
    rel = reliability_n0(d)
    # season-level stats for y/y and same-season descriptors
    t = d[d["tracked"]]
    S = t.groupby(["batter", "year"])["ev"].apply(lambda s: pd.Series({k: summ(s.to_numpy(), kd, p) for k, (kd, p) in SUMMARIES.items()} | {"n": len(s)})).unstack().reset_index()
    out = d.groupby(["batter", "year"]).agg(pa=("woba", "size"), hr=("kind", lambda s: (s == "home_run").sum())).reset_index()
    cn = d.groupby(["batter", "year"]).apply(lambda g: pd.Series({"wcon": g.loc[g["kind"].isin(["single", "double", "triple", "home_run", "out", "fielders_choice", "error"]), "woba"].mean()})).reset_index()
    S = S.merge(out, on=["batter", "year"]).merge(cn, on=["batter", "year"])
    S["hrpa"] = S["hr"] / S["pa"]
    a = S[(S["year"] == 2025) & (S["n"] >= 50)].set_index("batter"); b = S[(S["year"] == 2026) & (S["n"] >= 50)].set_index("batter")
    j = a.join(b, lsuffix="_25", rsuffix="_26", how="inner")
    big = S[(S["n"] >= 80) & (S["pa"] >= 100)]
    desc = []
    for k in SUMMARIES:
        desc.append({"summary": k, "split-half r": rel[k][0], "n0 (balls)": rel[k][1], "y/y 2025->26": j[f"{k}_25"].corr(j[f"{k}_26"]), "corr w/ HR/PA": big[k].corr(big["hrpa"]), "corr w/ wOBAcon": big[k].corr(big["wcon"])})
    print("DESCRIPTIVE: reliability, stability, and what each summary tracks (hitter-seasons with >= 80 tracked balls and >= 100 PA)")
    print(pd.DataFrame(desc).round(3).to_string(index=False))

    # ---- PA-split horse race --------------------------------------------------------------------------------------------
    ps.SIZES = (30, 60, 120, 200)
    rows = []
    for m in ps.SIZES:
        recs = []
        for (bt_, y), g in d.groupby(["batter", "year"]):
            n = len(g)
            if n < 150 or n - m < 30:
                continue
            woba = g["woba"].to_numpy(); tr = g["tracked"].to_numpy(); ev = g["ev"].to_numpy()
            for r in range(ps.N_DRAWS):
                idx = rng.permutation(n); a_, o = idx[:m], idx[m:]
                evk = ev[a_][tr[a_]]
                row = {"batter": bt_, "year": y, "n_out": n - m, "w_in": woba[a_].mean(), "y": woba[o].mean(), "bbe": len(evk),
                       "lg_w": d.loc[d["year"] == y, "woba"].mean()}
                for k, (kind, p) in SUMMARIES.items():
                    row[k] = summ(evk, kind, p) if len(evk) >= 3 else np.nan
                recs.append(row)
        D = pd.DataFrame(recs)
        y_ = D["y"].to_numpy(); base = (D["lg_w"] + m * (D["w_in"] - D["lg_w"]) / (m + ps.N0_WOBA)).to_numpy()
        wt, _, _ = ew.weights_from_base(y_, base, D["n_out"].to_numpy())
        groups = (D["batter"] + "|" + D["year"].astype(str)).to_numpy()
        lg = {k: S.loc[S["pa"] >= 100].groupby("year")[k].mean() for k in SUMMARIES}
        preds = {}
        for k in SUMMARIES:
            x = D[k].to_numpy(float); lgk = D["year"].map(lg[k]).to_numpy(float)
            sh = D["bbe"].to_numpy() / (D["bbe"].to_numpy() + rel[k][1])
            z = np.where(np.isnan(x), 0.0, (x - lgk) * sh); z = z / z.std()
            preds[k] = ps.cv_group(y_, base, z[:, None], wt, groups)
        ub = pd.Series(groups).unique(); idx_by = pd.Series(np.arange(len(y_))).groupby(groups).apply(np.array)
        bs = {k: [] for k in SUMMARIES}; dv90 = {k: [] for k in SUMMARIES}
        for _ in range(300):
            pick = rng.choice(ub, len(ub)); ii = np.concatenate([idx_by[u] for u in pick])
            rb = ps.rm(y_[ii], base[ii], wt[ii]); r90 = ps.rm(y_[ii], preds["EV90"][ii], wt[ii])
            for k in SUMMARIES:
                r = ps.rm(y_[ii], preds[k][ii], wt[ii]); bs[k].append(r - rb); dv90[k].append(r - r90)
        for k in SUMMARIES:
            rows.append({"summary": k, "m": m, "vs base": np.mean(bs[k]), "vs EV90": np.mean(dv90[k]), "lo": np.percentile(dv90[k], 5), "hi": np.percentile(dv90[k], 95)})
    R = pd.DataFrame(rows)
    piv = R.pivot(index="summary", columns="m", values="vs base")
    piv["avg"] = piv.mean(axis=1)
    order = list(SUMMARIES)
    print("\nCHANGE IN wOBA RMSE vs the regressed wOBA (negative = better), by known-sample size in PA; one summary at a time, shrunk by its own reliability")
    print(piv.loc[order].round(4).to_string())
    pv = R.pivot(index="summary", columns="m", values="vs EV90"); pv["avg"] = pv.mean(axis=1)
    print("\nDIFFERENCE vs EV90 (negative = better than EV90), averaged over the 4 sample sizes; per-size 90% intervals exclude 0 where flagged *")
    flag = R.assign(sig=((R["lo"] > 0) | (R["hi"] < 0))).pivot(index="summary", columns="m", values="sig")
    pv2 = pv.loc[order].round(4).astype(str)
    for k in order:
        for m in ps.SIZES:
            if k != "EV90" and bool(flag.loc[k, m]):
                pv2.loc[k, m] = pv2.loc[k, m] + "*"
    print(pv2.to_string())


if __name__ == "__main__":
    main()
