"""
fl_projection_value_runs.py -- WHAT ARE THE wOBA AND RA9 PROJECTIONS WORTH IN RUNS?  (user 2026-10-28: "don't judge by projected vs actual WAR; focus on wOBA and RA9; how many runs are those projections worth")
Out-of-sample 2025 -> 2026 backtest on the RATES the model projects (park-neutral wOBA for hitters, RA9 for pitchers), converted to runs at a FIXED workload so playing time, fielding, position and WAR constants play no part:
  hitters 400 PA (a full-time FL season), runs = wOBA difference / 1.0175 x 400;   pitchers 100 IP, runs = RA9 difference x 100 / 9.
Projections: league average (no information) | last year at face value | last year SHRUNK toward the league (hitters: wOBA x PA / (PA + 434), the n0 of the 2026-10-09 history study; pitchers: RA9 x IP / (IP + n0), n0 from {20, 40, 60, 100, 150} IP chosen on this same pool,
  one fitted parameter, mildly optimistic for the shrunk comparator) | MODEL (hitters: production preseason proj_woba on <= 2025 data; pitchers: the RA9 baseline fed by the production K% / BB% path, out-of-fold).
(1) ACCURACY: level-free (the sample's weighted mean error removed, as production recentres to the league), frozen standard inverse-variance weights, RMSE and MAE in wOBA / RA9 and in runs at the fixed workload; paired player bootstrap (1,000).
(2) SELECTION in runs: K best by each projection from random candidate pools; realised value of a pick = his realised rate relative to the average player of the pool, at the fixed workload (hitters: (wOBA_2026 - pool mean) / 1.0175 x 400; pitchers: (pool mean RA9 - RA9_2026) x 100 / 9),
  so "random picks" = 0 by construction.  Pools = players who played enough in 2026 (hitters >= 100 PA, pitchers >= 30 IP; the same survivor filter for every method).  Scenarios: 9 hitters from 40 + 5 pitchers from 30 (a team choosing among candidates), smaller pools, and the known
  stars removed (top 20% by last year's stats unavailable).  Runs per season; 1 win ~ 12 runs.   Output: output/projection_value_runs_log.txt, output/projection_value_runs.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_pitcher_features as fp

warnings.filterwarnings("ignore")
SC = 1.0175; PA_W = 400.0; IP_W = 100.0; RPW = 12.0; N0_H = 434.0


def hitters():
    h = pd.read_parquet("output/hitter_war_calibration_2026.parquet"); h = h[(h["pa_t"] >= 20) & (h["pa26"] >= 100)].dropna(subset=["proj_woba", "woba_n", "woba26"]).copy(); lg = float(h["lg_woba"].iloc[0])
    return pd.DataFrame({"name": h.index, "y": h["woba26"].to_numpy(), "n": h["pa26"].to_numpy(), "MODEL": h["proj_woba"].to_numpy(), "FACE": h["woba_n"].to_numpy(), "SHRUNK": (lg + (h["woba_n"] - lg) * h["pa_t"] / (h["pa_t"] + N0_H)).to_numpy(), "LEAGUE": lg})


def pitchers():
    import fl_pwar_calibration as cal
    cal.RETURN_ALL = True; b, war = cal.main(); w25 = war[war["year"] == 2025].set_index("player"); b = b[b["ip_25"].notna() & (b["ip_25"] >= 5) & (b["ip"] >= 30)].copy(); b["ra9_25"] = b["player"].map(w25["ra9"]); b = b.dropna(subset=["ra9_25", "pred", "target"])
    lg = float(np.average(w25["ra9"].dropna(), weights=w25.loc[w25["ra9"].notna(), "ip"].clip(lower=1))); best = None
    for n0 in (20, 40, 60, 100, 150):
        sh = lg + (b["ra9_25"] - lg) * b["ip_25"] / (b["ip_25"] + n0); e = float(np.sqrt(np.average((b["target"] - sh) ** 2, weights=b["ip"])))
        if best is None or e < best[0]:
            best = (e, n0)
    n0 = best[1]; print(f"pitcher shrinkage n0 = {n0} IP, league RA9 {lg:.2f}; pool {len(b)}", flush=True)
    return pd.DataFrame({"name": b["player"].to_numpy(), "y": b["target"].to_numpy(), "n": b["ip"].to_numpy(), "MODEL": b["pred"].to_numpy(), "FACE": b["ra9_25"].to_numpy(), "SHRUNK": (lg + (b["ra9_25"] - lg) * b["ip_25"] / (b["ip_25"] + n0)).to_numpy(), "LEAGUE": lg}), n0


def runs_scale(kind):
    return PA_W / SC if kind == "h" else IP_W / 9.0


def accuracy(D, kind, label, res):
    y = D["y"].to_numpy(); n = D["n"].to_numpy(float); w = ew.weights_from_base(y, D["MODEL"].to_numpy(), n)[0]; cen = lambda p: p + np.average(y - p, weights=w); sc = runs_scale(kind); rows = []
    P = {"league average (no information)": np.full(len(y), np.average(y, weights=w)), "last year at face value": D["FACE"].to_numpy(), "last year SHRUNK": D["SHRUNK"].to_numpy(), "MODEL": D["MODEL"].to_numpy()}
    for m, p in P.items():
        rm, ma = er.wmetrics(y, cen(p), w); rows.append({"projection": m, "RMSE": rm, "MAE": ma, "RMSE runs": rm * sc, "MAE runs": ma * sc})
    T = pd.DataFrame(rows); ids = D["name"].to_numpy(); cmp = {}
    for a, b, lab in (("last year at face value", "MODEL", "MODEL vs face value"), ("last year SHRUNK", "MODEL", "MODEL vs shrunk"), ("last year at face value", "last year SHRUNK", "shrunk vs face value"),
                      ("league average (no information)", "MODEL", "MODEL vs league average"), ("league average (no information)", "last year SHRUNK", "shrunk vs league average")):
        c = er.compare(y, cen(P[a]), cen(P[b]), w, ids, n=1000); cmp[lab] = {"dRMSE": c["dRMSE"], "90%": c["RMSE 90%"], "P": c["P(RMSE better)"], "dMAE": c["dMAE"], "P(MAE)": c["P(MAE better)"], "dRMSE runs": c["dRMSE"] * sc, "dMAE runs": c["dMAE"] * sc}
    unit = "wOBA" if kind == "h" else "RA9"; wk = "per 400 PA" if kind == "h" else "per 100 IP"
    print(f"\n{label}: ACCURACY (n = {len(D)}; level-free, frozen standard weights; runs {wk} per player)\n" + T.round(4).to_string(index=False))
    for lab, c in cmp.items():
        print(f"   {lab}: dRMSE {c['dRMSE']:+.4f} {unit} = {c['dRMSE runs']:+.2f} runs {c['90%']} P {c['P']:.2f};  dMAE {c['dMAE']:+.4f} = {c['dMAE runs']:+.2f} runs (P {c['P(MAE)']:.2f})")
    g = T.set_index("projection")["MAE runs"]; tot = g["last year at face value"] - g["MODEL"]; sh = g["last year at face value"] - g["last year SHRUNK"]; print(f"   share of the face-value -> MODEL improvement (MAE) delivered by shrinkage alone: {100 * sh / tot:.0f}%")
    res[label + " accuracy"] = {"table": T.to_dict("records"), "compare": cmp, "share_shrinkage_MAE": float(100 * sh / tot)}


def sel(D, kind, K, M, rng, reps, drop_top=0.0):
    """mean over random pools of the realised runs (fixed workload) of the K picks of each method relative to random picks from the same pool: [MODEL, SHRUNK, FACE]"""
    if drop_top > 0:
        D = D[D["FACE"].rank(pct=True, ascending=(kind == "p")) > drop_top].reset_index(drop=True)
    sgn = 1 if kind == "h" else -1; v = sgn * D["y"].to_numpy() * runs_scale(kind); P = {m: sgn * D[m].to_numpy() for m in ("MODEL", "FACE", "SHRUNK")}; n = len(D); M = min(M, n); out = np.zeros((reps, 3))
    for r in range(reps):
        i = rng.choice(n, M, replace=False); base = v[i].mean(); g = {m: v[i[np.argsort(-P[m][i])[:K]]].sum() - K * base for m in P}; out[r] = [g["MODEL"], g["SHRUNK"], g["FACE"]]
    return out.mean(axis=0)


def selection(H, P, res):
    rng = np.random.default_rng(11); rows = []; store = {}
    scen = (("hitters 9 of 40", H, "h", 9, 40, 0.0), ("hitters 9 of 25", H, "h", 9, 25, 0.0), ("hitters 9 of 18", H, "h", 9, 18, 0.0), ("hitters 9 of 40, top 20% by last year unavailable", H, "h", 9, 40, 0.2),
            ("pitchers 5 of 30", P, "p", 5, 30, 0.0), ("pitchers 5 of 15", P, "p", 5, 15, 0.0), ("pitchers 5 of 10", P, "p", 5, 10, 0.0), ("pitchers 5 of 30, top 20% by last year unavailable", P, "p", 5, 30, 0.2))
    for lab, D, kind, K, M, drop in scen:
        est = sel(D, kind, K, M, rng, 3000, drop); bs = np.array([sel(D.iloc[rng.integers(0, len(D), len(D))].reset_index(drop=True), kind, K, M, rng, 250, drop) for _ in range(200)]); store[lab] = (est, bs)
        ci = lambda j: f"[{np.percentile(bs[:, j], 5):+.1f}, {np.percentile(bs[:, j], 95):+.1f}]"; d = bs[:, 0] - bs[:, 2]; ds = bs[:, 0] - bs[:, 1]
        rows.append({"scenario": lab, "MODEL vs random (runs)": est[0], "90% ": ci(0), "SHRUNK vs random": est[1], "FACE vs random": est[2], "MODEL - FACE": est[0] - est[2], "90%  ": f"[{np.percentile(d, 5):+.1f}, {np.percentile(d, 95):+.1f}]", "P(>0)": float(np.mean(d > 0)),
                     "MODEL - SHRUNK": est[0] - est[1], "90%   ": f"[{np.percentile(ds, 5):+.1f}, {np.percentile(ds, 95):+.1f}]"})
    T = pd.DataFrame(rows); print("\nSELECTION in runs per season (hitters at 400 PA each, pitchers at 100 IP each; vs picks at random from the same pool)\n" + T.round(1).to_string(index=False)); res["selection"] = T.to_dict("records")
    for a, b_, lab in (("hitters 9 of 40", "pitchers 5 of 30", "COMBINED REALISTIC: 9 hitters from 40 + 5 pitchers from 30"), ("hitters 9 of 18", "pitchers 5 of 10", "COMBINED TIGHT: 9 hitters from 18 + 5 pitchers from 10"),
                       ("hitters 9 of 40, top 20% by last year unavailable", "pitchers 5 of 30, top 20% by last year unavailable", "COMBINED, KNOWN STARS UNAVAILABLE")):
        ea, ba = store[a]; eb, bb = store[b_]; e = ea + eb; b = ba + bb; d = b[:, 0] - b[:, 2]; ds = b[:, 0] - b[:, 1]
        print(f"\n{lab}: MODEL vs random {e[0]:+.1f} runs [{np.percentile(b[:, 0], 5):+.1f}, {np.percentile(b[:, 0], 95):+.1f}] (~{e[0] / RPW:+.1f} wins); SHRUNK vs random {e[1]:+.1f}; FACE-VALUE vs random {e[2]:+.1f};  "
              f"MODEL - FACE {e[0] - e[2]:+.1f} [{np.percentile(d, 5):+.1f}, {np.percentile(d, 95):+.1f}] (~{(e[0] - e[2]) / RPW:+.1f} wins), P(>0) {np.mean(d > 0):.2f};  MODEL - SHRUNK {e[0] - e[1]:+.1f} [{np.percentile(ds, 5):+.1f}, {np.percentile(ds, 95):+.1f}]")
        res[lab] = {"model_vs_random": float(e[0]), "shrunk_vs_random": float(e[1]), "face_vs_random": float(e[2]), "model_minus_face": float(e[0] - e[2]), "model_minus_shrunk": float(e[0] - e[1])}


def main():
    fp.WH_SVA_ESTIMATOR = "swstr_ct"; H = hitters(); P, n0p = pitchers(); print(f"hitters pool {len(H)} (>= 100 PA in 2026), pitchers pool {len(P)} (>= 30 IP in 2026)"); res = {"pitcher_n0": n0p}
    accuracy(H, "h", "HITTERS wOBA", res); accuracy(P, "p", "PITCHERS RA9", res); selection(H, P, res); json.dump(res, open("output/projection_value_runs.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
