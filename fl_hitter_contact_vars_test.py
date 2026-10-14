"""
fl_hitter_contact_vars_test.py -- do PullAir%, OppoGB%, Barrel%, Max EV and 90th-percentile EV add anything to the hitter projection?

Per hitter-season, from every tracked batted ball in play (HR included; exit velocity above 120 mph treated as a tracking error):
  PullAir%  share of batted balls that are air balls (launch angle >= 10) hit to the hitter's pull side (spray beyond 15 degrees)
  OppoGB%   share of batted balls that are ground balls (angle < 10) hit to the opposite field (beyond 15 degrees)
  Barrel%   share flagged as a barrel (Trackman flag: EV >= 95.5 at a matching launch angle)
  MaxEV     hardest tracked batted ball          EV90   90th percentile exit velocity           (AvgEV for reference)
Spray sign: right-handed hitters' home runs average -12.6 degrees, left-handed +15.4, so pull = direction beyond -15 (R) / +15 (L).

A  Reliability: split-half (random halves of each hitter's balls) and 2025 -> 2026 correlation.
B  What each one describes in the same season (hitter-seasons with >= 80 batted balls): correlation with wOBAcon, HR/PA, BABIP.
C  Does it predict NEXT year beyond the shipped model? 2025 -> 2026 (n ~ 100): correlation with the shipped model's miss, and a cross-validated
   add-on  y = base + b * z  (z = the variable, shrunk by its own reliability) for next-year wOBA, wOBAcon and HR/PA;
   then the same in reverse (2026 -> 2025) with a simple base, as a stability check.

Usage:  python fl_hitter_contact_vars_test.py
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import fl_eval_weights as ew
import fl_hitter_wobacon_test as wc
import fl_innings_projection as ip

warnings.filterwarnings("ignore")
BBE = "output/hitter_bbe_2025_2026.parquet"
VARS = {"pull_air": "PullAir%", "oppo_gb": "OppoGB%", "barrel": "Barrel%", "maxev": "MaxEV", "ev90": "EV90", "avgev": "AvgEV"}
RATE_VARS = ["pull_air", "oppo_gb", "barrel"]
MIN_BBE = 50


def flags(d):
    d = d[d["exit_speed"] <= 120].copy()
    side = d["batter_side"].map({"right": -1, "left": 1}).fillna(0)           # pull side sign of `direction`
    d["spray"] = d["direction"] * side                                        # + = pulled
    d["air"] = d["angle"] >= 10
    d["pull_air"] = ((d["spray"] > 15) & d["air"]).astype(float)
    d["oppo_gb"] = ((d["spray"] < -15) & ~d["air"]).astype(float)
    d["barrel"] = d["barrel"].fillna(0).astype(float)
    d["name_clean"] = d["batter"].apply(ip.lastfirst_to_clean)
    return d[d["batter_side"].isin(["right", "left"])]


def season_vars(d):
    g = d.groupby(["name_clean", "year"])
    s = g.agg(n=("exit_speed", "size"), pull_air=("pull_air", "mean"), oppo_gb=("oppo_gb", "mean"), barrel=("barrel", "mean"), maxev=("exit_speed", "max"),
              avgev=("exit_speed", "mean")).reset_index()
    s = s.merge(g["exit_speed"].quantile(0.9).rename("ev90").reset_index(), on=["name_clean", "year"])
    return s


def partA(d, s):
    print("A. RELIABILITY")
    rng = np.random.default_rng(0)
    res = {v: [] for v in VARS}; sizes = []
    for rep in range(30):
        d["half"] = rng.integers(0, 2, len(d))
        g = d.groupby(["name_clean", "year", "half"])
        h = g.agg(n=("exit_speed", "size"), pull_air=("pull_air", "mean"), oppo_gb=("oppo_gb", "mean"), barrel=("barrel", "mean"), maxev=("exit_speed", "max"), avgev=("exit_speed", "mean"))
        h = h.join(g["exit_speed"].quantile(0.9).rename("ev90")).reset_index()
        w = h.pivot_table(index=["name_clean", "year"], columns="half", values=["n"] + list(VARS)).dropna()
        w = w[(w[("n", 0)] >= 25) & (w[("n", 1)] >= 25)]
        sizes.append(w[[("n", 0), ("n", 1)]].to_numpy().mean())
        for v in VARS:
            res[v].append(w[(v, 0)].corr(w[(v, 1)]))
    a = s[(s["year"] == 2025) & (s["n"] >= MIN_BBE)].set_index("name_clean"); b = s[(s["year"] == 2026) & (s["n"] >= MIN_BBE)].set_index("name_clean")
    j = a.join(b, lsuffix="_25", rsuffix="_26", how="inner")
    m = float(np.mean(sizes)); rows = []; n0 = {}
    for v, lab in VARS.items():
        r = float(np.mean(res[v])); sb = 2 * r / (1 + r)
        n0[v] = max(m * (1 - r) / r, 1.0) if v in RATE_VARS else np.nan
        rows.append({"variable": lab, "split-half r (halves ~%d balls)" % round(m): r, "full-season reliability (SB)": sb, "y/y corr 2025->2026": j[f"{v}_25"].corr(j[f"{v}_26"]), "n hitters (y/y)": len(j)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return n0, j


def partB(s, q):
    print("\nB. WHAT EACH DESCRIBES THE SAME SEASON (hitter-seasons with >= 80 batted balls; park-neutral outcomes)")
    x = s[s["n"] >= 80].merge(q[["name_clean", "year", "wcon_n", "hrr", "babip", "iso", "k", "bbr", "woba_n", "xwoba", "pa"]], on=["name_clean", "year"], how="inner")
    x = x[x["pa"] >= 100]
    rows = []
    for v, lab in list(VARS.items()) + [("xwoba", "xwOBA (reference)")]:
        rows.append({"variable": lab, "n": x[v].notna().sum(), "wOBAcon": x[v].corr(x["wcon_n"]), "HR/PA": x[v].corr(x["hrr"]), "ISO": x[v].corr(x["iso"]), "BABIP": x[v].corr(x["babip"]),
                     "K%": x[v].corr(x["k"]), "wOBA": x[v].corr(x["woba_n"])})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def cv_add(y, base, z, wt, extra=None):
    """10-fold x20 cross-validated add-on  y ~ base + z @ b  (OLS on the base's miss, weighted); returns RMSE and mean coefficients."""
    X = z if z.ndim == 2 else z[:, None]
    pr = np.zeros(len(y)); co = []
    for rep in range(20):
        for tri, tei in KFold(10, shuffle=True, random_state=rep).split(X):
            Xt = np.column_stack([np.ones(len(tri)), X[tri]])
            b = np.linalg.lstsq(Xt * np.sqrt(wt[tri])[:, None], (y - base)[tri] * np.sqrt(wt[tri]), rcond=None)[0]
            pr[tei] += (base[tei] + np.column_stack([np.ones(len(tei)), X[tei]]) @ b) / 20; co.append(b[1:])
    return float(np.sqrt(np.average((y - pr) ** 2, weights=wt))), np.mean(co, axis=0)


def partC(J, q, s, n0):
    print("\nC. NEXT-YEAR PREDICTION, 2025 -> 2026, beyond the shipped model")
    T = J[J["t"] == 2025].copy()
    a = s[(s["year"] == 2025) & (s["n"] >= MIN_BBE)].set_index("name_clean")
    T = T.join(a[["n"] + list(VARS)].rename(columns={"n": "bbe25"}), on="name_clean").dropna(subset=["pull_air"])
    nx = q[q["year"] == 2026].set_index("name_clean")
    T["wcon_next"] = T["name_clean"].map(nx["wcon_n"]); T["hrr_next"] = T["name_clean"].map(nx["hrr"])
    xw = q[q["year"] == 2025].set_index("name_clean")["xwoba"]; pa25 = q[q["year"] == 2025].set_index("name_clean")["pa"]
    T["xw25"] = T["name_clean"].map(xw); T["pa25"] = T["name_clean"].map(pa25)
    T = T[T["pa25"] >= 50].dropna(subset=["wcon_next", "hrr_next", "xw25"])
    lgxw = np.average(T["xw25"], weights=T["pa25"])
    xw_s = ((T["pa25"] * T["xw25"] + 200 * lgxw) / (T["pa25"] + 200)).to_numpy()
    shipped = T["shipped_blend"].to_numpy()
    # shipped = blend + gamma (xwOBA - blend), gamma ~ .47 as fit in the earlier test
    shipped_x = shipped + 0.47 * (xw_s - shipped)
    wt_w = T["w"].to_numpy()
    targets = {"wOBA": (T["x_next"].to_numpy(), shipped_x, "shipped model (with xwOBA nudge)"), "wOBAcon": (T["wcon_next"].to_numpy(), T["wcon_pred"].to_numpy(), "regressed wOBAcon"),
               "HR/PA": (T["hrr_next"].to_numpy(), T["hrr_pred"].to_numpy(), "regressed HR/PA")}
    # z-scores, shrunk by the variable's own reliability (rate variables) -- n / (n + n0)
    Z = {}
    for v in VARS:
        x = T[v].to_numpy(dtype=float); mu = np.average(x, weights=T["bbe25"])
        sh = T["bbe25"].to_numpy() / (T["bbe25"].to_numpy() + n0[v]) if v in RATE_VARS else np.ones(len(T))
        z = (x - mu) * sh; Z[v] = z / z.std()
    print(f"   n = {len(T)} hitters (PA >= 50 in 2025, PA >= 100 in 2026); rate variables are shrunk by n/(n + n0) with n0 from their split-half reliability: "
          + ", ".join(f"{VARS[v]} {n0[v]:.0f}" for v in RATE_VARS))
    for tname, (y, base, blab) in targets.items():
        if tname == "wOBA":
            wt = wt_w
        else:                                           # each target gets its own frozen variance-law weight, from the base model's residuals
            nn = (T["n_next"] if tname == "HR/PA" else q[q["year"] == 2026].set_index("name_clean")["contact"].reindex(T["name_clean"]).to_numpy()).to_numpy() if tname == "HR/PA" else q[q["year"] == 2026].set_index("name_clean")["contact"].reindex(T["name_clean"]).to_numpy()
            wt, _a, _b = ew.weights_from_base(y, base, nn)
        rm0 = float(np.sqrt(np.average((y - base) ** 2, weights=wt)))
        rows = [{"add to the base": f"(base only: {blab})", "RMSE": rm0, "change": 0.0, "coef (per SD)": np.nan, "corr w/ miss": np.nan}]
        miss = y - base
        for v, lab in VARS.items():
            rm, co = cv_add(y, base, Z[v], wt)
            cc = float(np.corrcoef(Z[v], miss)[0, 1])
            rows.append({"add to the base": lab, "RMSE": rm, "change": rm - rm0, "coef (per SD)": float(co[0]), "corr w/ miss": cc})
        zz = np.column_stack([Z[v] for v in ("pull_air", "oppo_gb", "barrel", "ev90", "maxev")])
        rm, co = cv_add(y, base, zz, wt)
        rows.append({"add to the base": "all five together", "RMSE": rm, "change": rm - rm0, "coef (per SD)": np.nan, "corr w/ miss": np.nan})
        print(f"\n   target: next-year {tname}")
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    # reverse check: 2026 -> 2025 with a simple base (own wOBA regressed to the league, n0 = 444); variables from 2026
    print("\n   REVERSE CHECK (stability): predict 2025 wOBA from 2026 data, base = own 2026 wOBA regressed (n0 444)")
    a26 = s[(s["year"] == 2026) & (s["n"] >= MIS_BBE)].set_index("name_clean") if False else s[(s["year"] == 2026) & (s["n"] >= MIN_BBE)].set_index("name_clean")
    q26 = q[q["year"] == 2026].set_index("name_clean"); q25 = q[q["year"] == 2025].set_index("name_clean")
    R = a26[["n"] + list(VARS)].join(q26[["woba_n", "pa"]].rename(columns={"woba_n": "w26", "pa": "pa26"})).join(q25[["woba_n", "pa"]].rename(columns={"woba_n": "w25", "pa": "pa25"}), how="inner")
    R = R[(R["pa26"] >= 50) & (R["pa25"] >= 100)].dropna()
    lg26 = np.average(R["w26"], weights=R["pa26"]); lg25 = np.average(R["w25"], weights=R["pa25"])
    base = lg25 + (R["pa26"] * (R["w26"] - lg26)) / (R["pa26"] + 444.0)
    y = R["w25"].to_numpy(); base = base.to_numpy(); wt2, _a, _b = ew.weights_from_base(y, base, R["pa25"].to_numpy())
    rm0 = float(np.sqrt(np.average((y - base) ** 2, weights=wt2)))
    rows = [{"add to the base": "(base only)", "RMSE": rm0, "change": 0.0, "coef (per SD)": np.nan}]
    for v, lab in VARS.items():
        x = R[v].to_numpy(dtype=float); mu = np.average(x, weights=R["n"]); sh = R["n"].to_numpy() / (R["n"].to_numpy() + n0[v]) if v in RATE_VARS else np.ones(len(R))
        z = (x - mu) * sh; z = z / z.std(); rm, co = cv_add(y, base, z, wt2)
        rows.append({"add to the base": lab, "RMSE": rm, "change": rm - rm0, "coef (per SD)": float(co[0])})
    print(f"   n = {len(R)}"); print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    d = flags(pd.read_parquet(BBE))
    s = season_vars(d)
    n0, j = partA(d, s)
    with contextlib.redirect_stdout(io.StringIO()):
        J, q, xc, w = wc.main()
    partB(s, q)
    partC(J, q, s, n0)
