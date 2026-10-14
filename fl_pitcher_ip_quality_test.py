"""
fl_pitcher_ip_quality_test.py -- do better pitchers get more innings than the IP model gives them (the pitcher analogue of the hitter PA finding)?
Backtest evidence: top 15 pitchers by projected WAR 2025 -> 2026: projected 1.63 vs realised 2.22 WAR, projected IP about 85 vs realised about 100+; slope of realised on projected WAR 1.24 (projections too compressed).
Returning pitchers (any IP next year), transitions 2022->23 ... 2025->26, leave-one-transition-out, equal weights (innings are not a binomial rate), pitcher-cluster bootstrap, RMSE and MAE (reporting standard):
  A  current production: IP_next = 19.31 + 0.628 IP_t (season length ignored)
  B  per team game: IP_next / G_next = a + b IP_t / G_t, projected at G_next (regular-season games 96 in 2022-25, 102 in 2026-27)
  C  B + quality q = shrunk K% - BB% of season t relative to the league (n0 = 150 BF)
  D  C + innings two seasons ago per game
  E  C + starter share (GS / G of season t): NOT adopted by the user earlier ("projecting innings alone will suffice"), shown for scale only.
Also: bias by quality decile and for the top decile / top quintile of IP_t.
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_innings_projection as ip

warnings.filterwarnings("ignore")
G = {2022: 96, 2023: 96, 2024: 96, 2025: 96, 2026: 102, 2027: 102}; N0 = 150.0


def main():
    p = ip.build_panel().dropna(subset=["ip", "bf"]); p = p[p["ip"] > 0].copy(); p["gs"] = p["gs"].fillna(0); p["g"] = p["g"].replace(0, np.nan)
    lg = {y: (g["so"].sum() - g["bb"].sum()) / g["bf"].sum() for y, g in p.groupby("year")}
    p["q"] = [((so - bb) / bf - lg[y]) * bf / (bf + N0) for so, bb, bf, y in zip(p["so"], p["bb"], p["bf"], p["year"])]; p["sp"] = (p["gs"] / p["g"]).fillna(0)
    rows = []
    for t in (2022, 2023, 2024, 2025):
        a = p[p["year"] == t].set_index("name_clean"); b = p[p["year"] == t + 1].set_index("name_clean"); pr = p[p["year"] == t - 1].set_index("name_clean")
        d = pd.DataFrame({"ip_t": a["ip"], "q": a["q"], "sp": a["sp"], "ip_next": b["ip"].reindex(a.index)}); d["ip_prev"] = pr["ip"].reindex(d.index).fillna(0); d["t"] = t; rows.append(d.dropna(subset=["ip_next"]).reset_index())
    r = pd.concat(rows, ignore_index=True); r["x"] = r["ip_t"] / r["t"].map(G); r["y"] = r["ip_next"] / r["t"].map(lambda t: G[t + 1]); r["xp"] = r["ip_prev"] / r["t"].map(lambda t: G.get(t - 1, 96)); r["g_next"] = r["t"].map(lambda t: G[t + 1])
    print(f"{len(r)} returning pitcher-transitions; IP next mean {r['ip_next'].mean():.1f} SD {r['ip_next'].std():.1f}; corr(q, IP/G next) {r['q'].corr(r['y']):+.2f}, corr(IP_t/G, IP_next/G) {r['x'].corr(r['y']):+.2f}")
    specs = {"B per game": ["x"], "C + quality": ["x", "q"], "D + prior-prior IP": ["x", "q", "xp"], "E + starter share (not adopted)": ["x", "q", "sp"]}
    P = {k: [] for k in ["A current"] + list(specs)}; Y, I, T, Q, XT = [], [], [], [], []
    for t in (2022, 2023, 2024, 2025):
        tr, te = r[r["t"] != t], r[r["t"] == t]; ba = np.polyfit(tr["ip_t"], tr["ip_next"], 1); P["A current"].append(np.maximum(ba[1] + ba[0] * te["ip_t"].to_numpy(), 1.0))
        for k, cols in specs.items():
            X = np.column_stack([np.ones(len(tr))] + [tr[c] for c in cols]); bb = np.linalg.lstsq(X, tr["y"].to_numpy(), rcond=None)[0]; P[k].append(np.maximum(te["g_next"].to_numpy() * (np.column_stack([np.ones(len(te))] + [te[c] for c in cols]) @ bb), 1.0))
        Y.append(te["ip_next"].to_numpy()); I.append(te["name_clean"].to_numpy()); T.append(np.full(len(te), t)); Q.append(te["q"].to_numpy()); XT.append(te["ip_t"].to_numpy())
    y, ids, tt, qq, xt = map(np.concatenate, (Y, I, T, Q, XT)); PP = {k: np.concatenate(v) for k, v in P.items()}; w = np.ones(len(y)); rows = []
    for k, v in PP.items():
        rm, ma = er.wmetrics(y, v, w); row = {"model": k, "RMSE": rm, "MAE": ma, "R^2": 1 - ((y - v) ** 2).sum() / ((y - y.mean()) ** 2).sum()}
        if k != "A current":
            rep = er.compare(y, PP["A current"], v, w, ids, n=600); row.update({"dRMSE vs A": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE vs A": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
        rows.append(row)
    print("IP next year, leave-one-transition-out, returners\n" + er.table(rows, 3))
    for lab, m in (("top decile of quality q", qq >= np.quantile(qq, .9)), ("top quintile of prior-season IP", xt >= np.quantile(xt, .8)), ("pitchers with >= 80 IP last season", xt >= 80)):
        print(f"  {lab} (n = {m.sum()}): actual mean IP {y[m].mean():.0f}; A projects {PP['A current'][m].mean():.0f} (bias {y[m].mean() - PP['A current'][m].mean():+.0f}), C projects {PP['C + quality'][m].mean():.0f} ({y[m].mean() - PP['C + quality'][m].mean():+.0f}), E {PP['E + starter share (not adopted)'][m].mean():.0f} ({y[m].mean() - PP['E + starter share (not adopted)'][m].mean():+.0f})")
    X = np.column_stack([np.ones(len(r)), r["x"], r["q"]]); b = np.linalg.lstsq(X, r["y"].to_numpy(), rcond=None)[0]; res = r["y"] - X @ b
    print(f"\nfinal fit C: IP/game_next = {b[0]:.3f} + {b[1]:.3f} IP_t/G_t + {b[2]:.1f} q (q in K-BB points: +.05 = +{b[2] * .05 * 102:.1f} IP at 102 games); resid SD {res.std(ddof=3):.3f} per game = {res.std(ddof=3) * 102:.1f} IP")
    for ipt in (30, 60, 100, 130):
        print(f"  {ipt} IP in a 102-game season: A {19.31 + 0.6278 * ipt:.0f} | C at q = -.03 / 0 / +.05 / +.10: " + " / ".join(f"{102 * (b[0] + b[1] * ipt / 102 + b[2] * qv):.0f}" for qv in (-.03, 0, .05, .10)))


if __name__ == "__main__":
    main()
