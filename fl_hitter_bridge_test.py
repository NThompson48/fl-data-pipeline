"""
fl_hitter_bridge_test.py -- can MiLB hitting predict a hitter's FIRST Frontier League season?  (user 2026-10-09: "need to make a bridge for hitters"; rookies / thin history)
Pairs: FL hitters whose first panel season is 2023-26 (fl_hitter_panel) matched BY NAME to MiLB seasons strictly before the debut year (output/milb_hitting_2018_2026.parquet; names that map to more than one MLBAM id are dropped).
Targets (first FL season, deviation from that year's FL league mean): wOBA (direct), K%, BB%, HR/PA.  Source features: per source row (player, level, season) the deviation from the SAME level-season PA-weighted league mean,
 aggregated with PA x decay^lag weights and shrunk to 0 by n0 PA: u_L = sum w dev / (sum w + n0), one u per level group (so a player's AAA and Rookie lines count differently; a missing level is u = 0).
Models (all fitted with the frozen inverse-variance weights from the debut-class-mean residuals; n = FL PA in the debut year):
  M0 league mean (deviation 0)   M1 debut-class mean (intercept only)   M2 pooled slope on u (levels pooled)   M3 level-group slopes (AAA+AA | A+ + A | ROK)
n0 and decay chosen by an INNER leave-one-year-out on the training years, outer = leave-one-debut-year-out (4 folds).  Report weighted RMSE and MAE with a player-cluster bootstrap (fl_eval_report).
"""
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew

warnings.filterwarnings("ignore")
STATS = {"woba": "wOBA", "k": "K%", "bbr": "BB%", "hrr": "HR/PA"}
GROUPS = {"AAA": 0, "AA": 0, "A+": 1, "A": 1, "ROK": 2}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def milb_table():
    m = pd.read_parquet("output/milb_hitting_2018_2026.parquet")
    m = m[m["plateAppearances"] > 0].copy()
    m["nm"] = m["player_name"].map(norm)
    amb = m.groupby("nm")["batter"].nunique()
    m = m[~m["nm"].isin(amb[amb > 1].index)].copy()
    bb = m["baseOnBalls"] - m["intentionalWalks"]
    h1 = m["hits"] - m["doubles"] - m["triples"] - m["homeRuns"]
    den = m["atBats"] + bb + m["sacFlies"] + m["hitByPitch"]
    m["woba"] = (0.69 * bb + 0.72 * m["hitByPitch"] + 0.88 * h1 + 1.25 * m["doubles"] + 1.58 * m["triples"] + 2.03 * m["homeRuns"]) / den
    m["k"] = m["strikeOuts"] / m["plateAppearances"]
    m["bbr"] = m["baseOnBalls"] / m["plateAppearances"]
    m["hrr"] = m["homeRuns"] / m["plateAppearances"]
    m["pa"] = m["plateAppearances"]
    m["grp"] = m["level"].map(GROUPS)
    for s in STATS:
        ok = m[s].notna()
        lg = m[ok].groupby(["level", "season"]).apply(lambda g: np.average(g[s], weights=g["pa"]))
        m[f"dev_{s}"] = m[s] - pd.Series(list(zip(m["level"], m["season"]))).map(lg).to_numpy()
    return m


def fl_debuts():
    h = pd.read_parquet("output/fl_hitter_panel.parquet")
    h["nm"] = h["name_clean"].map(norm)
    h["k"] = h["so"] / h["pa"]
    h["bbr"] = h["bb"] / h["pa"]
    h["hrr"] = h["hr"] / h["pa"]
    first = h.groupby("nm")["year"].transform("min")
    d = h[(h["year"] == first) & (h["year"] >= 2023) & (h["pa"] >= 20)].copy()
    for s in STATS:
        ok = h[s].notna()
        mean = h[ok].groupby("year").apply(lambda g: np.average(g[s], weights=g["pa"]))
        d[f"y_{s}"] = d[s] - d["year"].map(mean)
    return d.drop_duplicates("nm")


def features(d, m, n0, decay, stat):
    """u_g per level group for every debut hitter (columns = level groups), and a has-history flag."""
    by = {k: g for k, g in m.groupby("nm")}
    U = np.zeros((len(d), 3))
    has = np.zeros(len(d), bool)
    for i, (nm, yr) in enumerate(zip(d["nm"], d["year"])):
        g = by.get(nm)
        if g is None:
            continue
        g = g[g["season"] < yr]
        if g.empty:
            continue
        has[i] = True
        w = g["pa"] * decay ** (yr - g["season"] - 1)
        for k in range(3):
            s = (g["grp"] == k).to_numpy()
            if s.any():
                U[i, k] = (w[s] * g.loc[s, f"dev_{stat}"]).sum() / (w[s].sum() + n0)
    return U, has


def fit_pred(Xtr, ytr, wtr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    b = np.linalg.solve(A.T @ (A * wtr[:, None]) + 1e-9 * np.eye(A.shape[1]), A.T @ (wtr * ytr))
    return np.column_stack([np.ones(len(Xte)), Xte]) @ b, b


def design(U, kind):
    return U.sum(axis=1, keepdims=True) if kind == "pooled" else U


def main():
    m = milb_table()
    d = fl_debuts()
    print(f"FL debut hitters 2023-26 with >= 20 PA: {len(d)}")
    RES = {}
    GRID_N0 = {"woba": (100, 200, 400, 800), "k": (50, 100, 200, 400), "bbr": (50, 100, 200, 400), "hrr": (100, 200, 400, 800)}
    for s, lab in STATS.items():
        y = d[f"y_{s}"].to_numpy()
        n = d["pa"].to_numpy(float)
        yr = d["year"].to_numpy()
        _, has = features(d, m, 200, 0.5, s)
        sel = np.isfinite(y) & has
        dd = d[sel].reset_index(drop=True)
        y, n, yr = y[sel], n[sel], yr[sel]
        base = np.array([np.average(y[yr != t], weights=n[yr != t]) for t in yr])
        a, b = ew.fit_variance_law(y - base, n)
        w = ew.inv_var_weights(n, a, b)
        print(f"\n=== {lab}: n = {len(y)} debut hitters with MiLB history; variance law a={a:.5f} b={b:.3f} (n0_eval {b / a:.0f} PA)")
        cache = {(n0, dc): features(dd, m, n0, dc, s)[0] for n0 in GRID_N0[s] for dc in (1.0, 0.5)}
        P = {"M0 league mean": np.zeros(len(y)), "M1 debut-class mean": base.copy(), "M2 pooled slope": np.zeros(len(y)), "M3 level-group slopes": np.zeros(len(y))}
        chosen = {}
        for kind, key in (("pooled", "M2 pooled slope"), ("groups", "M3 level-group slopes")):
            for t in sorted(set(yr)):
                te, tr = yr == t, yr != t
                best = None
                for (n0, dc), U in cache.items():
                    X = design(U, kind)
                    err = 0.0
                    for t2 in sorted(set(yr[tr])):
                        i_tr, i_te = tr & (yr != t2), tr & (yr == t2)
                        p, _ = fit_pred(X[i_tr], y[i_tr], w[i_tr], X[i_te])
                        err += np.sum(w[i_te] * (y[i_te] - p) ** 2)
                    if best is None or err < best[0]:
                        best = (err, n0, dc)
                _, n0, dc = best
                X = design(cache[(n0, dc)], kind)
                P[key][te], _ = fit_pred(X[tr], y[tr], w[tr], X[te])
                chosen[(key, t)] = (n0, dc)
        rows = []
        for k, v in P.items():
            r, ma = er.wmetrics(y, v, w)
            row = {"model": k, "RMSE": r, "MAE": ma}
            if k not in ("M0 league mean", "M1 debut-class mean"):
                rep = er.compare(y, P["M1 debut-class mean"], v, w, dd["nm"].to_numpy(), n=800)
                row.update({"dRMSE vs M1": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE vs M1": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
            rows.append(row)
        print(er.table(rows, 5))
        ch = [chosen[("M2 pooled slope", t)] for t in sorted(set(yr))]
        print("  chosen (n0, decay) by held-out year, pooled:", ch)
        n0, dc = max(set(ch), key=ch.count)
        U = cache[(n0, dc)]
        Xp = design(U, "pooled")
        _, bco = fit_pred(Xp, y, w, Xp)
        print(f"  full-sample pooled fit (n0 {n0}, decay {dc}): intercept {bco[0]:+.4f}, slope {bco[1]:.3f}; sd of u {Xp.std():.4f}")
        _, bg = fit_pred(U, y, w, U)
        print(f"  group slopes AAA+AA {bg[1]:.3f} | A+ + A {bg[2]:.3f} | ROK {bg[3]:.3f}")
        RES[s] = {"dd": dd, "y": y, "n": n, "yr": yr, "w": w, "P": P}
    return RES


if __name__ == "__main__":
    main()
