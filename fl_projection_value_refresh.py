"""
fl_projection_value_refresh.py -- HOW MANY WINS IS THE MODEL WORTH vs (a) JUST LAST YEAR'S STATS and (b) A NO-INFORMATION BASELINE (the league average / random picks)?  (user 2026-10-28)
A refresh of fl_projection_value_test.py (2026-10-10) on the CURRENT code and data: the hitter 2025 -> 2026 calibration frame is regenerated (fl_hitter_war_calibration.main: production preseason path on <= 2025 data, regular-season-only data) and the pitcher frame uses the production
network stuff estimator (swstr_ct).  Same design as the original: out-of-sample 2025 -> 2026, WAR on the projection's own definition, pools = players who played in 2026 (the same survivor filter for every method); 1 WAR ~ 1 win.
  MODEL    = production projection;   NAIVE-1 = last year's stats as-is (rates x last year's playing time, through the same WAR formula);   RANDOM = no information: K players drawn at random from the candidate pool = K x the pool's mean realised WAR (equivalently, every player projected at the league average).
SELECTION: wins per season = realised WAR of the K best by each projection (whole pool; random candidate pools of M players = a team choosing among M candidates), bootstrap over players (B = 300) for 90% intervals.
Output: output/projection_value_refresh_log.txt, output/projection_value_refresh.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_war_calibration as hwc
import fl_pitcher_features as fp
import fl_projection_value_test as pv

warnings.filterwarnings("ignore")


def sel_stats(D, K, M, rng, reps):
    """returns [MODEL - NAIVE-1, MODEL - RANDOM, NAIVE-1 - RANDOM, MODEL picks, NAIVE-1 picks, RANDOM picks]"""
    n = len(D); real = D["real"].to_numpy(); mo, n1 = D["MODEL"].to_numpy(), D["NAIVE-1"].to_numpy(); mu = real.mean()
    if M is None:
        sel = lambda p: np.argsort(-p)[:K]; a, b = real[sel(mo)].sum(), real[sel(n1)].sum(); return np.array([a - b, a - K * mu, b - K * mu, a, b, K * mu])
    out = np.zeros((reps, 6))
    for r in range(reps):
        i = rng.choice(n, M, replace=False); sel = lambda p: i[np.argsort(-p[i])[:K]]; a, b = real[sel(mo)].sum(), real[sel(n1)].sum(); rnd = K * real[i].mean(); out[r] = [a - b, a - rnd, b - rnd, a, b, rnd]
    return out.mean(axis=0)


def selection(D, label, ks, pools, res):
    rng = np.random.default_rng(10); rows = []
    for K in ks:
        for M in pools:
            est = sel_stats(D, K, M, rng, 2000); bs = np.array([sel_stats(D.iloc[rng.integers(0, len(D), len(D))].reset_index(drop=True), K, M, rng, 300) for _ in range(300)])
            ci = lambda j: f"[{np.percentile(bs[:, j], 5):+.2f}, {np.percentile(bs[:, j], 95):+.2f}]"
            rows.append({"K picked": K, "from pool of": "whole pool (%d)" % len(D) if M is None else M, "MODEL picks (WAR)": est[3], "last-year picks": est[4], "random picks": est[5], "MODEL - last year (wins)": est[0], "90%": ci(0), "MODEL - random (wins)": est[1], "90% ": ci(1), "last year - random (wins)": est[2], "90%  ": ci(2)})
    T = pd.DataFrame(rows); print(f"\n{label}: SELECTION (wins per season for K picks; realised WAR)\n" + T.round(2).to_string(index=False)); res[label] = T.to_dict("records")


def accuracy(D, label, res):
    y = D["real"].to_numpy(); rows = []
    for m, p in (("MODEL", D["MODEL"].to_numpy()), ("last year's stats", D["NAIVE-1"].to_numpy()), ("league average (no information)", np.full(len(y), y.mean()))):
        rm, ma = er.wmetrics(y, p, np.ones(len(y))); rows.append({"projection": m, "RMSE (WAR)": rm, "MAE (WAR)": ma, "corr": np.corrcoef(p, y)[0, 1] if np.std(p) > 0 else np.nan})
    T = pd.DataFrame(rows); r1 = er.compare(y, D["NAIVE-1"].to_numpy(), D["MODEL"].to_numpy(), np.ones(len(y)), D["name"].to_numpy(), n=1000); r0 = er.compare(y, np.full(len(y), y.mean()), D["MODEL"].to_numpy(), np.ones(len(y)), D["name"].to_numpy(), n=1000)
    print(f"\n{label}: ACCURACY of projected WAR vs realised 2026 WAR (n = {len(D)}; mean realised {y.mean():.2f}, SD {y.std():.2f})\n" + T.round(3).to_string(index=False) + f"\n   MODEL vs last year: dRMSE {r1['dRMSE']:+.3f} {r1['RMSE 90%']} P {r1['P(RMSE better)']:.2f};   MODEL vs league average: dRMSE {r0['dRMSE']:+.3f} {r0['RMSE 90%']} P {r0['P(RMSE better)']:.2f}")
    res[label + " accuracy"] = {"table": T.to_dict("records"), "vs last year": {k: r1[k] for k in ("dRMSE", "RMSE 90%", "P(RMSE better)")}, "vs league": {k: r0[k] for k in ("dRMSE", "RMSE 90%", "P(RMSE better)")}}


def main():
    hwc.main(); fp.WH_SVA_ESTIMATOR = "swstr_ct"; H = pv.hitters(); P = pv.pitchers(); nh, npi = len(H), len(P)
    H = H.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True); P = P.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True); print(f"\nhitters pool {len(H)} (of {nh}), pitchers pool {len(P)} (of {npi})"); res = {}
    accuracy(H, "HITTERS", res); accuracy(P, "PITCHERS", res); selection(H, "HITTERS", (5, 9), (None, 40, 25), res); selection(P, "PITCHERS", (3, 5), (None, 40, 25), res)
    rng = np.random.default_rng(21); b = []
    for _ in range(300):
        h = sel_stats(H.iloc[rng.integers(0, len(H), len(H))].reset_index(drop=True), 9, 40, rng, 300); p = sel_stats(P.iloc[rng.integers(0, len(P), len(P))].reset_index(drop=True), 5, 30, rng, 300); b.append(h + p)
    b = np.array(b); eh, ep = sel_stats(H, 9, 40, rng, 2000), sel_stats(P, 5, 30, rng, 2000); e = eh + ep; ci = lambda j: f"[{np.percentile(b[:, j], 5):+.2f}, {np.percentile(b[:, j], 95):+.2f}]"
    print(f"\nCOMBINED REALISTIC SCENARIO (9 hitters chosen from 40 candidates + 5 pitchers from 30): MODEL - last year = {e[0]:+.2f} wins per season {ci(0)} (hitters {eh[0]:+.2f}, pitchers {ep[0]:+.2f}), P(>0) {np.mean(b[:, 0] > 0):.2f};  MODEL - random = {e[1]:+.2f} {ci(1)} (hitters {eh[1]:+.2f}, pitchers {ep[1]:+.2f});  last year - random = {e[2]:+.2f} {ci(2)}")
    res["combined"] = {"model_vs_lastyear": float(e[0]), "model_vs_random": float(e[1]), "lastyear_vs_random": float(e[2]), "ci_model_vs_lastyear": ci(0), "ci_model_vs_random": ci(1), "ci_lastyear_vs_random": ci(2), "hitters": eh.tolist(), "pitchers": ep.tolist()}
    json.dump(res, open("output/projection_value_refresh.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
