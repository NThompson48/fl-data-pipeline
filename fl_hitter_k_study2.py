"""
fl_hitter_k_study2.py -- K%: which discipline subset, which functional form, and does it hold year to year?

1  Forms (within-season PA split, fair reference = same CV intercept for every model):
     faded      sum b_v z_v x n0w / (m + n0w)      the informed-prior form used for EV90 (influence fades as his own PA accumulate)
     unfaded    sum b_v z_v                         a constant-strength adjustment
   and subsets: Whiff%, Whiff%+Chase%, Whiff%+Chase%+ZContact%, SwStr%+Chase%, all nine.
2  Year to year, the real test: 2025 -> 2026 on top of the PRODUCTION two-season K% base (n0 79, prior-season weight .75), and the reverse 2026 -> 2025 with a one-season base;
   fair intercept in every model; paired bootstrap over hitters.
3  Fitted coefficients (raw units) for the chosen form, pooled over sample sizes.

Usage:  python fl_hitter_k_study2.py
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
rng = np.random.default_rng(77)
SUBSETS = {"Whiff%": ["Whiff%"], "Whiff% + Chase%": ["Whiff%", "Chase%"], "Whiff% + Chase% + ZContact%": ["Whiff%", "Chase%", "ZContact%"], "SwStr% + Chase%": ["SwStr%", "Chase%"],
           "all nine": list(ks.VARS)}


def cvk(y, base, X, wt, reps=15):
    pr = np.zeros(len(y))
    for rep in range(reps):
        for tr, te in KFold(10, shuffle=True, random_state=rep).split(y):
            A = np.column_stack([np.ones(len(tr)), X[tr]]) if X.shape[1] else np.ones((len(tr), 1))
            b = np.linalg.lstsq(A * np.sqrt(wt[tr])[:, None], (y - base)[tr] * np.sqrt(wt[tr]), rcond=None)[0]
            pr[te] += (base[te] + (np.column_stack([np.ones(len(te)), X[te]]) if X.shape[1] else np.ones((len(te), 1))) @ b) / reps
    return pr


def part1(D):
    print("1. FORMS AND SUBSETS (within-season PA split; change in K% RMSE vs the intercept-only reference; negative = better)")
    rows = []
    for m in sorted(D["m"].unique()):
        S = D[D["m"] == m].reset_index(drop=True)
        y = S["y"].to_numpy(); base = (S["lg_k"] + m * (S["k_in"] - S["lg_k"]) / (m + ks.N0_K)).to_numpy()
        wt, _, _ = ew.weights_from_base(y, base, S["n_out"].to_numpy()); groups = (S["name_clean"] + "|" + S["year"].astype(str)).to_numpy()
        Z = {v: np.where(S[v].isna(), 0.0, S[v].to_numpy()) for v in ks.VARS}; sd = {v: Z[v].std() for v in ks.VARS}
        fade = ks.N0W / (m + ks.N0W)
        ref = ps.cv_group(y, base, np.zeros((len(y), 0)), wt, groups)
        ug = pd.Series(groups).unique(); by = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array)
        for name, vs in SUBSETS.items():
            for form in ("faded", "unfaded"):
                X = np.column_stack([Z[v] / sd[v] * (fade if form == "faded" else 1.0) for v in vs])
                pr = ps.cv_group(y, base, X, wt, groups)
                d = []
                for _ in range(200):
                    ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); d.append(ps.rm(y[ii], pr[ii], wt[ii]) - ps.rm(y[ii], ref[ii], wt[ii]))
                rows.append({"m": m, "subset": name, "form": form, "change": np.mean(d), "lo": np.percentile(d, 5), "hi": np.percentile(d, 95), "ref RMSE": ps.rm(y, ref, wt)})
    R = pd.DataFrame(rows)
    piv = R.pivot_table(index=["subset", "form"], columns="m", values="change")
    piv["avg"] = piv.mean(axis=1)
    print(piv.round(5).to_string())
    print("   reference RMSE by m:", R.groupby("m")["ref RMSE"].first().round(4).to_dict())
    return R


def year_to_year(a, s, n0):
    print("\n2. YEAR TO YEAR with the production K% base (change in K% RMSE vs the intercept-only reference; 90% interval; hitter-cluster bootstrap)")
    q = dev.load(); q["woba"] = q["woba_n"]
    for label, t_in, t_out, two_season in (("2025 -> 2026 (production two-season base)", 2025, 2026, True), ("2026 -> 2025 (reverse, one-season base)", 2026, 2025, False)):
        if two_season:
            d = bt.transitions(q, "k"); d = d[d["t"] == t_in].copy()
            base = bt.predict(d, ks.N0_K, 0.75).to_numpy(); y = d["x_next"].to_numpy(); wt = d["w"].to_numpy(); names = d["name_clean"].to_numpy(); pa_eff = (d["n_t"] + 0.75 * d["n_prev"]).to_numpy()
        else:
            a_ = q[q["year"] == t_in].set_index("name_clean"); b_ = q[q["year"] == t_out].set_index("name_clean")
            j = a_[["k", "pa"]].join(b_[["k", "pa"]], lsuffix="_in", rsuffix="_out", how="inner"); j = j[(j["pa_in"] >= 10) & (j["pa_out"] >= 100)]
            lgi = np.average(j["k_in"], weights=j["pa_in"]); lgo = np.average(j["k_out"], weights=j["pa_out"])
            base = (lgo + j["pa_in"] * (j["k_in"] - lgi) / (j["pa_in"] + ks.N0_K)).to_numpy(); y = j["k_out"].to_numpy(); names = j.index.to_numpy(); pa_eff = j["pa_in"].to_numpy()
            wt, _, _ = ew.weights_from_base(y, base, j["pa_out"].to_numpy())
        st = s[s["year"] == t_in].set_index("name_clean").reindex(names)
        lgv = {v: s.loc[s["year"] == t_in, ks.VARS[v][0]].sum() / s.loc[s["year"] == t_in, ks.VARS[v][1]].sum() for v in ks.VARS}
        Z = {}
        for v, (nu, de) in ks.VARS.items():
            den = st[de].to_numpy(float); x = st[v].to_numpy(float)
            Z[v] = np.where(np.isnan(x), 0.0, (x - lgv[v]) * den / (den + n0[v])); Z[v] = Z[v] / Z[v].std()
        fade = ks.N0W / (pa_eff + ks.N0W)
        ref = cvk(y, base, np.zeros((len(y), 0)), wt)
        print(f"\n  {label}: n = {len(y)}; base RMSE {ps.rm(y, base, wt):.4f}; reference with intercept {ps.rm(y, ref, wt):.4f}")
        ug = names; by = {u: i for i, u in enumerate(names)}
        rows = []
        for name, vs in SUBSETS.items():
            for form in ("faded", "unfaded"):
                X = np.column_stack([Z[v] * (fade if form == "faded" else 1.0) for v in vs]); pr = cvk(y, base, X, wt)
                dd = []
                for _ in range(500):
                    ii = rng.integers(0, len(y), len(y)); dd.append(ps.rm(y[ii], pr[ii], wt[ii]) - ps.rm(y[ii], ref[ii], wt[ii]))
                rows.append({"subset": name, "form": form, "change": np.mean(dd), "interval": f"[{np.percentile(dd,5):+.5f}, {np.percentile(dd,95):+.5f}]"})
        print(pd.DataFrame(rows).round(5).to_string(index=False))


def main():
    a = ks.pa_table(); s = ks.season_stats(a)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        n0 = ks.partA(a, s)
    D = ks.draws(a, s, n0)
    part1(D)
    year_to_year(a, s, n0)


if __name__ == "__main__":
    main()
