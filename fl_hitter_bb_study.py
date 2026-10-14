"""
fl_hitter_bb_study.py -- does Swing% (and the swing-decision measures behind it) work as an informative prior for hitter BB%?

Current BB% model: regress to the league over the last two seasons (n0 147 PA, prior-season weight .5).  Candidates from every pitch seen (2025-26):
Swing% (swings per pitch), ZSwing% (swings per pitch in the zone), Chase% (swings per pitch outside), FPSwing% (first-pitch swing rate), Take-K% (called strikes per pitch).
A  Reliability, 2025 -> 2026 correlation, same-season correlation with BB% (and K%).
B  Within-season PA split (hitter-seasons >= 150 PA; known sample m = 30 / 60 / 120 / 200 PA -> BB% over the rest); every model, including the reference, carries the same
   cross-validated intercept; hitter-grouped CV; standard inverse-variance weights; hitter-cluster bootstrap.  Faded (informed-prior) form: sum b z x n0w / (m + n0w).
C  Year to year, 2025 -> 2026 (production two-season BB% base) and 2026 -> 2025 (one-season base), same fair reference.
Usage:  python fl_hitter_bb_study.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import fl_eval_weights as ew
import fl_hitter_baseline_backtest as bt
import fl_hitter_k_study as ks
import fl_hitter_model_dev as dev
import fl_hitter_pa_split_test as ps

warnings.filterwarnings("ignore")
rng = np.random.default_rng(55)
N0_BB = 146.6
VARS = {v: ks.VARS[v] for v in ("Swing%", "ZSwing%", "Chase%", "FPSwing%", "Take-K%")}
SUBSETS = {"Swing%": ["Swing%"], "Chase%": ["Chase%"], "ZSwing%": ["ZSwing%"], "Swing% + Chase%": ["Swing%", "Chase%"], "ZSwing% + Chase%": ["ZSwing%", "Chase%"], "all five": list(VARS)}
COLS = ["swings", "whiff", "pitches", "o_pitches", "z_pitches", "o_swing", "z_swing", "o_contact", "z_contact", "cstrike", "fp_pitches", "fp_swing", "k", "bb", "pa"]


def cvk(y, base, X, wt, reps=15):
    pr = np.zeros(len(y))
    for rep in range(reps):
        for tr, te in KFold(10, shuffle=True, random_state=rep).split(y):
            A = np.column_stack([np.ones(len(tr)), X[tr]]) if X.shape[1] else np.ones((len(tr), 1))
            b = np.linalg.lstsq(A * np.sqrt(wt[tr])[:, None], (y - base)[tr] * np.sqrt(wt[tr]), rcond=None)[0]
            pr[te] += (base[te] + (np.column_stack([np.ones(len(te)), X[te]]) if X.shape[1] else np.ones((len(te), 1))) @ b) / reps
    return pr


def partA(a, s):
    print("A. RELIABILITY AND WHAT EACH TRACKS (hitter-seasons with >= 100 PA)")
    big = a.groupby(["name_clean", "year"]).filter(lambda g: len(g) >= 100)
    res = {v: [] for v in VARS}; res["BB%"] = []; dens = {v: [] for v in VARS}
    for rep in range(20):
        h = big.assign(h=rng.integers(0, 2, len(big))).groupby(["name_clean", "year", "h"])[COLS].sum().reset_index()
        for v, (nu, de) in VARS.items():
            h[v] = h[nu] / h[de].replace(0, np.nan)
        h["BB%"] = h["bb"] / h["pa"]
        w = h.pivot_table(index=["name_clean", "year"], columns="h", values=list(VARS) + ["BB%", "pitches", "o_pitches", "z_pitches", "fp_pitches", "pa"]).dropna()
        for v in list(VARS) + ["BB%"]:
            res[v].append(w[(v, 0)].corr(w[(v, 1)]))
        for v, (nu, de) in VARS.items():
            dens[v].append(float(w[(de, 0)].mean()))
    a25 = s[(s["year"] == 2025) & (s["pa"] >= 100)].set_index("name_clean"); a26 = s[(s["year"] == 2026) & (s["pa"] >= 100)].set_index("name_clean")
    j = a25.join(a26, lsuffix="_25", rsuffix="_26", how="inner")
    big_s = s[s["pa"] >= 100]; rows = []; n0 = {}
    for v in list(VARS) + ["BB%"]:
        r = float(np.mean(res[v]))
        if v in VARS:
            n0[v] = max(np.mean(dens[v]) * (1 - r) / r, 1.0)
        col = "bbpct" if v == "BB%" else v
        rows.append({"measure": v, "split-half r": r, "full-season reliability": 2 * r / (1 + r), "n0 (own denominator)": n0.get(v, N0_BB), "y/y 2025->26": j[f"{col}_25"].corr(j[f"{col}_26"]),
                     "corr w/ BB%": big_s[col].corr(big_s["bbpct"]), "corr w/ K%": big_s[col].corr(big_s["kpct"]), "last yr -> next BB%": j[f"{col}_25"].corr(j["bbpct_26"])})
    print(pd.DataFrame(rows).round(3).to_string(index=False) + f"\n   (n = {len(j)} hitters in both years)")
    return n0


def draws(a, s, n0, sizes=(30, 60, 120, 200), ndraw=25):
    lgv = {v: s[VARS[v][0]].sum() / s[VARS[v][1]].sum() for v in VARS}; lg_bb = a.groupby("year")["bb"].mean(); rows = []
    for (nm, y), g in a.groupby(["name_clean", "year"]):
        n = len(g)
        if n < 150:
            continue
        arr = {c: g[c].to_numpy() for c in COLS}; bb = g["bb"].to_numpy()
        for m in sizes:
            if n - m < 30:
                continue
            for _ in range(ndraw):
                idx = rng.permutation(n); ia, io_ = idx[:m], idx[m:]
                row = {"name_clean": nm, "year": y, "m": m, "n_out": n - m, "bb_in": bb[ia].mean(), "y": bb[io_].mean(), "lg_bb": lg_bb[y]}
                for v, (nu, de) in VARS.items():
                    num, den = arr[nu][ia].sum(), arr[de][ia].sum(); x = num / den if den > 0 else np.nan
                    row[v] = np.nan if np.isnan(x) else (x - lgv[v]) * den / (den + n0[v])
                rows.append(row)
    return pd.DataFrame(rows)


def partB(D):
    print("\nB. WITHIN-SEASON PA SPLIT: known sample m PA -> BB% over the rest (fair reference: same CV intercept in every model)")
    out = []
    for m in sorted(D["m"].unique()):
        S = D[D["m"] == m].reset_index(drop=True)
        y = S["y"].to_numpy(); base = (S["lg_bb"] + m * (S["bb_in"] - S["lg_bb"]) / (m + N0_BB)).to_numpy()
        wt, _, _ = ew.weights_from_base(y, base, S["n_out"].to_numpy()); groups = (S["name_clean"] + "|" + S["year"].astype(str)).to_numpy()
        Z = {v: np.where(S[v].isna(), 0.0, S[v].to_numpy()) for v in VARS}; sd = {v: Z[v].std() for v in VARS}; fade = N0_BB / (m + N0_BB)
        ref = ps.cv_group(y, base, np.zeros((len(y), 0)), wt, groups)
        ug = pd.Series(groups).unique(); by = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array)
        for name, vs in SUBSETS.items():
            pr = ps.cv_group(y, base, np.column_stack([Z[v] / sd[v] * fade for v in vs]), wt, groups); d = []
            for _ in range(250):
                ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); d.append(ps.rm(y[ii], pr[ii], wt[ii]) - ps.rm(y[ii], ref[ii], wt[ii]))
            out.append({"m": m, "subset": name, "change": np.mean(d), "lo": np.percentile(d, 5), "hi": np.percentile(d, 95), "ref": ps.rm(y, ref, wt)})
    R = pd.DataFrame(out)
    print("   reference RMSE by m:", R.groupby("m")["ref"].first().round(4).to_dict())
    piv = R.pivot(index="subset", columns="m", values="change"); piv["avg"] = piv.mean(axis=1)
    print("   change in BB% RMSE vs the intercept-only reference (negative = better):"); print(piv.loc[list(SUBSETS)].round(5).to_string())
    sig = R.assign(s=lambda x: (x["lo"] > 0) | (x["hi"] < 0)).pivot(index="subset", columns="m", values="s")
    print("   90% interval excludes 0:"); print(sig.loc[list(SUBSETS)].to_string())


def year_to_year(s, n0):
    print("\nC. YEAR TO YEAR (change in BB% RMSE vs the intercept-only reference; 90% interval)")
    q = dev.load(); q["woba"] = q["woba_n"]
    for label, t_in, t_out, two in (("2025 -> 2026 (production two-season base)", 2025, 2026, True), ("2026 -> 2025 (reverse, one-season base)", 2026, 2025, False)):
        if two:
            d = bt.transitions(q, "bbr"); d = d[d["t"] == t_in].copy()
            base = bt.predict(d, N0_BB, 0.5).to_numpy(); y = d["x_next"].to_numpy(); wt = d["w"].to_numpy(); names = d["name_clean"].to_numpy(); pa_eff = (d["n_t"] + 0.5 * d["n_prev"]).to_numpy()
        else:
            a_ = q[q["year"] == t_in].set_index("name_clean"); b_ = q[q["year"] == t_out].set_index("name_clean")
            j = a_[["bbr", "pa"]].join(b_[["bbr", "pa"]], lsuffix="_in", rsuffix="_out", how="inner"); j = j[(j["pa_in"] >= 10) & (j["pa_out"] >= 100)]
            lgi = np.average(j["bbr_in"], weights=j["pa_in"]); lgo = np.average(j["bbr_out"], weights=j["pa_out"])
            base = (lgo + j["pa_in"] * (j["bbr_in"] - lgi) / (j["pa_in"] + N0_BB)).to_numpy(); y = j["bbr_out"].to_numpy(); names = j.index.to_numpy(); pa_eff = j["pa_in"].to_numpy()
            wt, _, _ = ew.weights_from_base(y, base, j["pa_out"].to_numpy())
        st = s[s["year"] == t_in].set_index("name_clean").reindex(names)
        lgv = {v: s.loc[s["year"] == t_in, VARS[v][0]].sum() / s.loc[s["year"] == t_in, VARS[v][1]].sum() for v in VARS}
        Z = {}
        for v, (nu, de) in VARS.items():
            den = st[de].to_numpy(float); x = st[v].to_numpy(float); Z[v] = np.where(np.isnan(x), 0.0, (x - lgv[v]) * den / (den + n0[v])); Z[v] = Z[v] / Z[v].std()
        fade = N0_BB / (pa_eff + N0_BB); ref = cvk(y, base, np.zeros((len(y), 0)), wt)
        print(f"\n  {label}: n = {len(y)}; base RMSE {ps.rm(y, base, wt):.4f}; reference with intercept {ps.rm(y, ref, wt):.4f}")
        rows = []
        for name, vs in SUBSETS.items():
            for form in ("faded", "unfaded"):
                pr = cvk(y, base, np.column_stack([Z[v] * (fade if form == "faded" else 1.0) for v in vs]), wt); dd = []
                for _ in range(500):
                    ii = rng.integers(0, len(y), len(y)); dd.append(ps.rm(y[ii], pr[ii], wt[ii]) - ps.rm(y[ii], ref[ii], wt[ii]))
                rows.append({"subset": name, "form": form, "change": np.mean(dd), "interval": f"[{np.percentile(dd,5):+.5f}, {np.percentile(dd,95):+.5f}]"})
        print(pd.DataFrame(rows).round(5).to_string(index=False))


def main():
    a = ks.pa_table(); s = ks.season_stats(a)
    print(f"PA {len(a):,}; walks {a['bb'].mean():.1%}; hitter-seasons >= 150 PA: {int((s['pa'] >= 150).sum())}")
    n0 = partA(a, s)
    D = draws(a, s, n0)
    partB(D)
    year_to_year(s, n0)


if __name__ == "__main__":
    main()
