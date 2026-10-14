"""
fl_hitter_pa_split_test.py -- the thin-sample question with enough hitters: does xwOBA (or a subset of PullAir% / OppoGB% / Barrel% / MaxEV / EV90) predict a hitter's
OTHER plate appearances better than his regressed wOBA, when the sample behind the projection is small?

Design (no survivorship, no league drift): every hitter-season with >= 150 PA (2025 and 2026).  For a sample size m in {30, 60, 120}, draw m of his PA at random as the
"known" sample (25 independent draws); the target is his wOBA over the remaining PA (>= 30).  From the known sample only:
    base      lg + m (wOBA_known - lg) / (m + 440)          the regressed wOBA (same-season league lg)
    nudge     base + g (shrunk xwOBA_known - base)          shrink n0 = 200, as shipped
    subsets   base + b0 + sum b z                           PullAir%, OppoGB%, Barrel%, MaxEV, EV90 computed from the known sample's batted balls, each shrunk by its own reliability
Add-ons are fit by 10-fold cross-validation GROUPED BY HITTER (a hitter's draws never straddle train and test) and scored with the standard inverse-variance weights
(law fit on the base model's residuals at each m, then frozen).  Intervals: bootstrap over hitters (all of a hitter's draws move together).
This is a same-season test of what the Trackman skills add at a given sample size; it does not replace the year-to-year tests (aging, role and park changes are absent here).

Usage:  python fl_hitter_pa_split_test.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import fl_eval_weights as ew

warnings.filterwarnings("ignore")
PA = "output/hitter_pa_2025_2026.parquet"
N_DRAWS = 25
SIZES = (30, 60, 120, 200)
N0_WOBA, N0_X = 440.0, 200.0
N0_VAR = {"pull_air": 55, "oppo_gb": 68, "barrel": 42, "maxev": 46, "ev90": 15}
MODELS = {"nudge": None, "EV90": ["ev90"], "EV90+PullAir": ["ev90", "pull_air"], "Barrel+PullAir": ["barrel", "pull_air"], "OppoGB+MaxEV": ["oppo_gb", "maxev"],
          "all five": ["pull_air", "oppo_gb", "barrel", "maxev", "ev90"], "nudge+EV90+PullAir": "combo",
          "EV90 (prior-weighted)": "prior:ev90", "EV90+PullAir (prior-weighted)": "prior:ev90,pull_air", "nudge+EV90 (prior-weighted)": "prior:nudge,ev90"}
rng = np.random.default_rng(2026)


def prep():
    d = pd.read_parquet(PA)
    d = d[d["kind"] != "sacrifice"].copy()
    d["woba"] = d["woba"].fillna(0.0)
    bip = d["kind"].isin(["single", "double", "triple", "home_run", "out", "fielders_choice", "error"])
    d["xw"] = d["xwoba"]
    d["xw"] = d["xw"].fillna(d.groupby("kind")["xw"].transform("mean")).fillna(d["woba"])
    d["tracked"] = bip & d["exit_speed"].notna() & d["angle"].notna() & (d["exit_speed"] <= 120)
    side = d["batter_side"].map({"right": -1, "left": 1}).fillna(0)
    spray = d["direction"] * side
    air = d["angle"] >= 10
    d["pull_air"] = (d["tracked"] & (spray > 15) & air).astype(float)
    d["oppo_gb"] = (d["tracked"] & (spray < -15) & ~air).astype(float)
    d["barrel"] = (d["tracked"] & (d["barrel"] == 1)).astype(float)
    d["ev"] = np.where(d["tracked"], d["exit_speed"], np.nan)
    d = d[d["batter_side"].isin(["right", "left"])]
    return d


def draws(d):
    rows = []
    lg = d.groupby("year").agg(w=("woba", "mean"), x=("xw", "mean"))
    lgv = {v: d.loc[d["tracked"]].groupby("year")[v].mean() for v in ("pull_air", "oppo_gb", "barrel")}
    lg_ev = d.loc[d["tracked"]].groupby("year")["ev"].agg(mean="mean")
    for (b, y), g in d.groupby(["batter", "year"]):
        n = len(g)
        if n < 150:
            continue
        woba = g["woba"].to_numpy(); xw = g["xw"].to_numpy(); tr = g["tracked"].to_numpy()
        feats = {v: g[v].to_numpy() for v in ("pull_air", "oppo_gb", "barrel")}; ev = g["ev"].to_numpy()
        for m in SIZES:
            if n - m < 30:
                continue
            for r in range(N_DRAWS):
                idx = rng.permutation(n); a, o = idx[:m], idx[m:]
                bbe = int(tr[a].sum())
                evk = ev[a][tr[a]]
                row = {"batter": b, "year": y, "m": m, "draw": r, "n_out": n - m, "w_in": woba[a].mean(), "xw_in": xw[a].mean(), "y": woba[o].mean(), "bbe": bbe,
                       "lg_w": lg.loc[y, "w"], "lg_x": lg.loc[y, "x"]}
                for v in ("pull_air", "oppo_gb", "barrel"):
                    row[v] = feats[v][a][tr[a]].mean() if bbe >= 3 else np.nan
                row["maxev"] = evk.max() if bbe >= 3 else np.nan
                row["ev90"] = np.quantile(evk, 0.9) if bbe >= 3 else np.nan
                rows.append(row)
    D = pd.DataFrame(rows)
    ref = d[d["tracked"]]
    for v in ("pull_air", "oppo_gb", "barrel"):
        D[f"lg_{v}"] = D["year"].map(ref.groupby("year")[v].mean())
    D["lg_maxev"] = D["year"].map(ref.groupby(["batter", "year"])["ev"].max().groupby("year").mean())
    D["lg_ev90"] = D["year"].map(ref.groupby(["batter", "year"])["ev"].quantile(0.9).groupby("year").mean())
    return D


def zs(D, cols_needed):
    Z = {}
    for v in ("pull_air", "oppo_gb", "barrel", "maxev", "ev90"):
        x = D[v].to_numpy(float); lg = D[f"lg_{v}"].to_numpy(float)
        sh = D["bbe"].to_numpy() / (D["bbe"].to_numpy() + N0_VAR[v])
        z = np.where(np.isnan(x), 0.0, (x - lg) * sh)
        Z[v] = z / np.std(z)
    return Z


def cv_group(y, base, X, wt, groups):
    pr = np.zeros(len(y))
    for tr, te in GroupKFold(10).split(X, y, groups):
        A = np.column_stack([np.ones(len(tr)), X[tr]])
        b = np.linalg.lstsq(A * np.sqrt(wt[tr])[:, None], (y - base)[tr] * np.sqrt(wt[tr]), rcond=None)[0]
        pr[te] = base[te] + np.column_stack([np.ones(len(te)), X[te]]) @ b
    return pr


def rm(y, p, wt):
    return float(np.sqrt(np.average((y - p) ** 2, weights=wt)))


def main():
    d = prep()
    D = draws(d)
    print(f"hitter-seasons with >= 150 PA: {D.groupby(['batter','year']).ngroups}; draws per size {N_DRAWS}; total rows {len(D):,}")
    Zall = zs(D, None)
    out = []
    for m in SIZES:
        mask = (D["m"] == m).to_numpy()
        S = D[mask].reset_index(drop=True)
        Z = {v: Zall[v][mask] for v in Zall}
        y = S["y"].to_numpy(); base = (S["lg_w"] + m * (S["w_in"] - S["lg_w"]) / (m + N0_WOBA)).to_numpy()
        wt, a, b = ew.weights_from_base(y, base, S["n_out"].to_numpy())
        xs = (m * S["xw_in"] + N0_X * S["lg_x"]) / (m + N0_X)
        zx = (xs - base).to_numpy()
        groups = (S["batter"] + "|" + S["year"].astype(str)).to_numpy()
        # nudge: one-parameter, grouped CV
        pr = {"base": base}
        p = np.zeros(len(y)); gs = []
        for tr, te in GroupKFold(10).split(zx, y, groups):
            g = np.sum(wt[tr] * zx[tr] * (y - base)[tr]) / np.sum(wt[tr] * zx[tr] ** 2); gs.append(g); p[te] = base[te] + g * zx[te]
        pr["nudge"] = p
        for k, cols in MODELS.items():
            if k == "nudge":
                continue
            if isinstance(cols, str) and cols.startswith("prior:"):
                names = cols[6:].split(",")
                X = np.column_stack([(zx if c == "nudge" else Z[c]) for c in names]) * (N0_WOBA / (m + N0_WOBA))     # informed prior: weight (1 - reliability)
            elif cols == "combo":
                X = np.column_stack([zx, Z["ev90"], Z["pull_air"]])
            else:
                X = np.column_stack([Z[c] for c in cols])
            pr[k] = cv_group(y, base, X, wt, groups)
        # hitter-cluster bootstrap
        ub = pd.Series(groups).unique(); idx_by = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array)
        bs = {k: [] for k in pr if k != "base"}; bs_vs = {k: [] for k in pr if k not in ("base", "nudge")}
        for _ in range(400):
            pick = rng.choice(ub, len(ub)); ii = np.concatenate([idx_by[u] for u in pick])
            rb = rm(y[ii], base[ii], wt[ii]); rn = rm(y[ii], pr["nudge"][ii], wt[ii])
            for k in bs:
                r = rm(y[ii], pr[k][ii], wt[ii]); bs[k].append(r - rb)
                if k in bs_vs: bs_vs[k].append(r - rn)
        row_base = rm(y, base, wt)
        print(f"\nKNOWN SAMPLE = {m} PA   ({len(S)//N_DRAWS} hitter-seasons x {N_DRAWS} draws; targets average {S['n_out'].mean():.0f} PA; law a={a:.5f} b={b:.3f}; nudge gamma {np.mean(gs):.2f}; base RMSE {row_base:.4f})")
        rows = []
        for k in bs:
            r = {"model": k, "change vs regressed wOBA": np.mean(bs[k]), "90% interval": f"[{np.percentile(bs[k],5):+.4f}, {np.percentile(bs[k],95):+.4f}]"}
            if k in bs_vs:
                r["vs nudge"] = np.mean(bs_vs[k]); r["vs nudge 90% interval"] = f"[{np.percentile(bs_vs[k],5):+.4f}, {np.percentile(bs_vs[k],95):+.4f}]"
            rows.append(r)
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        # coefficient interpretation for EV90 alone (full-sample fit): wOBA per +1 SD of shrunk EV90
        X1 = Z["ev90"][:, None]; A = np.column_stack([np.ones(len(y)), X1])
        bb = np.linalg.lstsq(A * np.sqrt(wt)[:, None], (y - base) * np.sqrt(wt), rcond=None)[0]
        print(f"   EV90 add-on fit on all rows: intercept {bb[0]:+.4f}, {bb[1]:+.4f} wOBA per +1 SD of (shrunk) EV90 in the known sample")


if __name__ == "__main__":
    main()
