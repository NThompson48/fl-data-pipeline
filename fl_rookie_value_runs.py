"""
fl_rookie_value_runs.py -- WHAT ARE THE ROOKIE wOBA AND RA9 PROJECTIONS WORTH IN RUNS?  (user 2026-10-28: "how about for just rookies?"; follows fl_projection_value_runs.py)
Rookies = FL DEBUT players with other-league history (MiLB / college / Pioneer), the players the other-league bridges project.  Out-of-sample (hitters: nested leave-one-debut-class-out, 200 debuts 2023-26 as in fl_rookie_value_test.py; pitchers: leave-one-out
weighted regression on the unified-bridge debut pairs).  Rates at a FIXED workload: hitters 400 PA (runs = wOBA / 1.0175 x 400), pitchers 100 IP (runs = RA9 x 100 / 9).
Projections: BASE = no information (the debut-class mean / the league-mean guess) | RAW = the other-league line at face value (hitters: the latest MiLB wOBA proxy, pitchers: the latest-season K% and multi-year BB% pushed through the RA9 baseline slopes) | MODEL = the bridge.
Outcomes: hitters = first-FL-season park-neutral wOBA (deviation from the FL league); pitchers = (a) the first FL season's ACTUAL RA9 (FL debuts of 2025-26 with >= 20 IP, matched to war_pitchers), and (b) the RA9 implied by his first-season K% and BB% (skill only: BABIP / HR luck removed; all debuts, as fl_rookie_value_test.py).
ACCURACY: level-free, frozen inverse-variance weights (variance law of the BASE residuals), weighted RMSE / MAE in rate and runs, paired player bootstrap (1,000).  SELECTION: K best by each projection from random candidate pools of M rookies, realised runs of the picks at the fixed workload
relative to picking at random from the same pool (so random = 0); runs per season, 1 win ~ 12 runs.   Output: output/rookie_value_runs_log.txt, output/rookie_value_runs.json
"""
import contextlib
import io
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_pitcher_features as fp
import fl_rookie_value_test as rv
import fl_unified_bridge as ub
from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
SC = 1.0175; RPW = 12.0


def hitters():
    D, _ = rv.hitters(); return D.rename(columns={"real": "y"}), 400.0 / SC


def pitchers():
    raw = fp.load_fl_pitches_cached()
    with contextlib.redirect_stdout(io.StringIO()):
        mdl = ub.fit(raw, verbose=False)
    p = mdl.pairs.copy().reset_index(drop=True); w = p["pa"].to_numpy(float); leagues = sorted(p["league"].unique()); I = np.column_stack([(p["league"] == lg).astype(float) for lg in leagues])

    def loo(X, y):
        XtWX = X.T @ (X * w[:, None]); beta = np.linalg.solve(XtWX, X.T @ (w * y)); e = y - X @ beta; h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(XtWX), X); return y - e / (1 - h), (np.sum(w * y) - w * y) / (np.sum(w) - w)
    pk, bk = loo(np.column_stack([I, p["u_k"].to_numpy()]), p["k_rate"].to_numpy()); pb, bb = loo(np.column_stack([np.ones(len(p)), p["u_bb"].to_numpy()]), p["bb_rate"].to_numpy()); ra = lambda k, b: -9.238 * k + 11.53 * b
    D = pd.DataFrame({"pid": p["pitcher_id"].to_numpy(), "fl_year": p["fl_year"].to_numpy(), "n": w, "y_kb": np.asarray(ra(p["k_rate"], p["bb_rate"])), "MODEL": np.asarray(ra(pk, pb)), "BASE": np.asarray(ra(bk, bb)), "RAW": np.asarray(ra(p["k_raw_latest"], p["bb_raw_multi"]))})
    D = D.replace([np.inf, -np.inf], np.nan).dropna().sort_values("n", ascending=False).drop_duplicates("pid").reset_index(drop=True)
    war = read_sql_df("select player, year, ip, ra9 from war_pitchers where game_type='reg_season'"); names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id"); names["pid"] = names["pitcher_id"].astype(str)
    D["pid"] = D["pid"].astype(str); D = D.merge(names[["pid", "pitcher"]], on="pid", how="left").merge(war.rename(columns={"player": "pitcher", "year": "fl_year", "ip": "ip_fl", "ra9": "ra9_fl"}), on=["pitcher", "fl_year"], how="left")
    D["name"] = D["pid"]; print(f"rookie pitchers: {len(D)} debuts; with actual RA9 and >= 20 IP in the debut season: {int((D['ip_fl'] >= 20).sum())}", flush=True); return D, 100.0 / 9.0


def accuracy(D, ycol, label, unit_name, scale, nname, res):
    d = D.dropna(subset=[ycol]); y = d[ycol].to_numpy(float); n = d[nname].to_numpy(float); P = {"no information": d["BASE"].to_numpy(), "other-league line at face value": d["RAW"].to_numpy(), "MODEL (bridge)": d["MODEL"].to_numpy()}
    a, b = ew.fit_variance_law(y - P["no information"], n); w = ew.inv_var_weights(n, a, b); cen = lambda q: q + np.average(y - q, weights=w); rows = []
    for m, q in P.items():
        rm, ma = er.wmetrics(y, cen(q), w); rows.append({"projection": m, "RMSE": rm, "MAE": ma, "RMSE runs": rm * scale, "MAE runs": ma * scale, "corr": np.corrcoef(q, y)[0, 1], "Spearman": pd.Series(q).corr(pd.Series(y), method="spearman")})
    T = pd.DataFrame(rows); ids = d["name"].to_numpy(); out = {}
    print(f"\n{label}: ACCURACY (n = {len(d)}; level-free, frozen weights; runs per {'400 PA' if scale > 20 and unit_name == 'wOBA' else '100 IP'} per player)\n" + T.round(4).to_string(index=False))
    for x, z, lab in (("no information", "MODEL (bridge)", "MODEL vs no information"), ("other-league line at face value", "MODEL (bridge)", "MODEL vs face value"), ("no information", "other-league line at face value", "face value vs no information")):
        c = er.compare(y, cen(P[x]), cen(P[z]), w, ids, n=1000); out[lab] = {"dRMSE": c["dRMSE"], "dRMSE runs": c["dRMSE"] * scale, "90%": c["RMSE 90%"], "P": c["P(RMSE better)"], "dMAE runs": c["dMAE"] * scale, "P(MAE)": c["P(MAE better)"]}
        print(f"   {lab}: dRMSE {c['dRMSE']:+.4f} {unit_name} = {c['dRMSE'] * scale:+.2f} runs {c['RMSE 90%']} P {c['P(RMSE better)']:.2f};  dMAE {c['dMAE'] * scale:+.2f} runs (P {c['P(MAE better)']:.2f})")
    res[label + " accuracy"] = {"table": T.to_dict("records"), "compare": out, "n": len(d)}


def sel(D, ycol, sign, scale, K, M, rng, reps):
    d = D.dropna(subset=[ycol]).reset_index(drop=True); v = sign * d[ycol].to_numpy(float) * scale; P = {m: sign * d[m].to_numpy(float) for m in ("MODEL", "RAW")}; n = len(d); M = min(M, n); out = np.zeros((reps, 2))
    for r in range(reps):
        i = rng.choice(n, M, replace=False); base = v[i].mean(); out[r] = [v[i[np.argsort(-P[m][i])[:K]]].sum() - K * base for m in ("MODEL", "RAW")]
    return out.mean(axis=0)


def selection(H, P, ycol_p, res):
    rng = np.random.default_rng(31); rows = []; store = {}
    for lab, D, yc, sign, scale, K, M in (("hitters 3 of 20", H, "y", 1, 400 / SC, 3, 20), ("hitters 3 of 40", H, "y", 1, 400 / SC, 3, 40), ("hitters 5 of 40", H, "y", 1, 400 / SC, 5, 40),
                                          ("pitchers 3 of 20", P, ycol_p, -1, 100 / 9, 3, 20), ("pitchers 3 of 40", P, ycol_p, -1, 100 / 9, 3, 40), ("pitchers 5 of 40", P, ycol_p, -1, 100 / 9, 5, 40)):
        est = sel(D, yc, sign, scale, K, M, rng, 3000); bs = np.array([sel(D.iloc[rng.integers(0, len(D), len(D))].reset_index(drop=True), yc, sign, scale, K, M, rng, 250) for _ in range(200)]); store[lab] = (est, bs); d = bs[:, 0] - bs[:, 1]
        rows.append({"scenario": lab, "MODEL vs random (runs)": est[0], "90%": f"[{np.percentile(bs[:, 0], 5):+.1f}, {np.percentile(bs[:, 0], 95):+.1f}]", "RAW vs random": est[1], "MODEL - RAW": est[0] - est[1], "90% ": f"[{np.percentile(d, 5):+.1f}, {np.percentile(d, 95):+.1f}]", "P(>0)": float(np.mean(d > 0))})
    T = pd.DataFrame(rows); print(f"\nSELECTION in runs per season (hitters 400 PA each, pitchers 100 IP each; pitcher outcome = {ycol_p}; vs random picks from the same pool)\n" + T.round(1).to_string(index=False)); res["selection " + ycol_p] = T.to_dict("records")
    for a, b_, lab in (("hitters 3 of 40", "pitchers 3 of 40", "COMBINED: 3 rookie hitters from 40 + 3 rookie pitchers from 40"), ("hitters 3 of 20", "pitchers 3 of 20", "COMBINED SMALLER POOLS: 3 + 3 from 20 each")):
        ea, ba = store[a]; eb, bb = store[b_]; e = ea + eb; b = ba + bb; d = b[:, 0] - b[:, 1]
        print(f"{lab}: MODEL vs random {e[0]:+.1f} runs [{np.percentile(b[:, 0], 5):+.1f}, {np.percentile(b[:, 0], 95):+.1f}] (~{e[0] / RPW:+.1f} wins); RAW vs random {e[1]:+.1f}; MODEL - RAW {e[0] - e[1]:+.1f} [{np.percentile(d, 5):+.1f}, {np.percentile(d, 95):+.1f}], P(>0) {np.mean(d > 0):.2f}")
        res[lab + " " + ycol_p] = {"model_vs_random": float(e[0]), "raw_vs_random": float(e[1]), "model_minus_raw": float(e[0] - e[1])}


def main():
    H, sh = hitters(); P, sp = pitchers(); res = {}
    accuracy(H, "y", "ROOKIE HITTERS wOBA", "wOBA", sh, "n", res)
    accuracy(P, "ra9_fl", "ROOKIE PITCHERS, first-season ACTUAL RA9", "RA9", sp, "ip_fl", res) if False else None
    P1 = P[P["ip_fl"] >= 20].reset_index(drop=True); accuracy(P1, "ra9_fl", "ROOKIE PITCHERS, first-season ACTUAL RA9 (>= 20 IP)", "RA9", sp, "ip_fl", res); accuracy(P, "y_kb", "ROOKIE PITCHERS, RA9 implied by first-season K% and BB%", "RA9", sp, "n", res)
    selection(H, P1, "ra9_fl", res); selection(H, P, "y_kb", res); json.dump(res, open("output/rookie_value_runs.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
