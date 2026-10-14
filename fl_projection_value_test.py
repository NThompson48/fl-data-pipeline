"""
fl_projection_value_test.py -- HOW MANY WINS DO THE PROJECTION MODELS ADD OVER "JUST LAST YEAR'S STATS"?  (user 2026-10-10)  Out-of-sample 2025 -> 2026 backtest, WAR on the projection's own definition.
Pools: hitters with a 2025-and-earlier projection and >= 20 PA in 2026 (fl_hitter_war_calibration.py output, n 146); pitchers with a 2025 record, a projection and >= 10 IP in 2026 (fl_pwar_calibration.py frame).  Realised 2026 WAR on the same definition as the projection.
Projections compared (all in WAR for 2026):
  MODEL   = production projection (hitters: proj PA x (bat + fielding + position + replacement)/rpw; pitchers: (replacement - projected RA9) x projected IP / 9 / 10.706)
  NAIVE-1 = last year as-is: last year's park-neutral wOBA / fielding / position per PA (hitters) or last year's RA9 (pitchers), times LAST YEAR'S playing time scaled to 102 games (x 102/96)
  NAIVE-2 = last year's RATES x the MODEL's playing time      (isolates the value of the rate model: regression to the mean, xwOBA, K/BB components, ...)
  MODEL-R x NAIVE-PA = the model's rates x last year's playing time  (isolates the value of the playing-time model)
Tests: (1) accuracy of projected WAR vs realised (RMSE, MAE, correlation, slope); (2) SELECTION: choose the K best by each projection from (a) the whole pool, (b) random candidate pools of M players (a team picking among M available): realised WAR of the picks, difference
(= wins per season, 1 WAR ~ 1 win), bootstrap over players (B = 300; inner 400 random pools) for 90% intervals.  Caveat: pools only contain players who actually played in 2026 (the same survivor filter for every method).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er

warnings.filterwarnings("ignore")
RPW_P = 10.706; G = 102 / 96


def hitters():
    h = pd.read_parquet("output/hitter_war_calibration_2026.parquet"); h = h[h["pa_t"] >= 20].copy(); rr = h["rar_rate"]; repl = h["repl_rate"]
    naive_rate = (h["woba_n"] - h["lg_woba"]) / 1.0175 + h["fielding_runs"] / h["pa_t"] + h["pos_adj_runs"] / h["pa_t"] + repl
    return pd.DataFrame({"name": h.index, "real": h["real_war"].to_numpy(), "MODEL": (h["proj_pa"] * rr / h["rpw"]).to_numpy(), "NAIVE-1": (h["pa_t"] * G * naive_rate / h["rpw"]).to_numpy(), "NAIVE-2": (h["proj_pa"] * naive_rate / h["rpw"]).to_numpy(),
                         "MODEL-R x NAIVE-PA": (h["pa_t"] * G * rr / h["rpw"]).to_numpy()})


def pitchers():
    import fl_pwar_calibration as cal
    import fl_ra9_trustmark_projection as ra9
    cal.RETURN_ALL = True; b, war = cal.main(); w25 = war[war["year"] == 2025].set_index("player"); b = b[b["ip_25"].notna() & (b["ip_25"] >= 5)].copy(); b["ra9_25"] = b["player"].map(w25["ra9"]); b = b.dropna(subset=["ra9_25"]); repl = ra9.LEAGUE_RA9_REPLACEMENT
    w = lambda r, ip: (repl - r) * ip / 9 / RPW_P
    return pd.DataFrame({"name": b["player"].to_numpy(), "real": w(b["target"], b["ip"]).to_numpy(), "MODEL": w(b["pred"], b["ip_proj"]).to_numpy(), "NAIVE-1": w(b["ra9_25"], b["ip_25"] * G).to_numpy(), "NAIVE-2": w(b["ra9_25"], b["ip_proj"]).to_numpy(),
                         "MODEL-R x NAIVE-PA": w(b["pred"], b["ip_25"] * G).to_numpy()})


def accuracy(D, label):
    rows = []; y = D["real"].to_numpy()
    for m in ("MODEL", "NAIVE-1", "NAIVE-2", "MODEL-R x NAIVE-PA"):
        p = D[m].to_numpy(); rm, ma = er.wmetrics(y, p, np.ones(len(y))); sl = np.polyfit(p, y, 1)[0]; rows.append({"projection": m, "RMSE": rm, "MAE": ma, "corr": np.corrcoef(p, y)[0, 1], "Spearman": pd.Series(p).corr(pd.Series(y), method="spearman"), "slope of realised on projected": sl, "mean projected": p.mean()})
    t = pd.DataFrame(rows); rep = er.compare(y, D["NAIVE-1"].to_numpy(), D["MODEL"].to_numpy(), np.ones(len(y)), D["name"].to_numpy(), n=1000)
    print(f"\n{label}: accuracy of projected WAR vs realised 2026 WAR (n = {len(D)}; mean realised {y.mean():.2f}, SD {y.std():.2f})\n" + t.round(3).to_string(index=False) + f"\n   MODEL vs NAIVE-1: dRMSE {rep['dRMSE']:+.3f} {rep['RMSE 90%']} P(better) {rep['P(RMSE better)']:.2f}; dMAE {rep['dMAE']:+.3f} {rep['MAE 90%']} P {rep['P(MAE better)']:.2f}")


def picks(D, K, M, rng, reps):
    """mean over random pools of size M (M = None: the whole pool) of realised WAR of the K best by MODEL minus by NAIVE-1 (and NAIVE-2)"""
    n = len(D); real = D["real"].to_numpy(); mo, n1, n2 = D["MODEL"].to_numpy(), D["NAIVE-1"].to_numpy(), D["NAIVE-2"].to_numpy()
    if M is None:
        sel = lambda p: np.argsort(-p)[:K]; return np.array([real[sel(mo)].sum() - real[sel(n1)].sum(), real[sel(n2)].sum() - real[sel(n1)].sum(), real[sel(mo)].sum() - real[sel(n2)].sum(), real[sel(mo)].sum(), real[sel(n1)].sum()])
    out = np.zeros((reps, 5))
    for r in range(reps):
        i = rng.choice(n, M, replace=False); sel = lambda p: i[np.argsort(-p[i])[:K]]; a, b, c = real[sel(mo)].sum(), real[sel(n1)].sum(), real[sel(n2)].sum(); out[r] = [a - b, c - b, a - c, a, b]
    return out.mean(axis=0)


def selection(D, label, ks, pools):
    rng = np.random.default_rng(10); rows = []
    for K in ks:
        for M in pools:
            est = picks(D, K, M, rng, 2000); bs = []
            for _ in range(300):
                Db = D.iloc[rng.integers(0, len(D), len(D))].reset_index(drop=True); bs.append(picks(Db, K, M, rng, 300))
            bs = np.array(bs); lo, hi = np.percentile(bs[:, 0], [5, 95]); lo2, hi2 = np.percentile(bs[:, 2], [5, 95])
            rows.append({"K picked": K, "from pool of": "whole pool (%d)" % len(D) if M is None else M, "realised WAR: MODEL picks": est[3], "NAIVE-1 picks": est[4], "MODEL minus NAIVE-1 (wins)": est[0], "90% interval": f"[{lo:+.2f}, {hi:+.2f}]",
                         "of which PLAYING-TIME model (NAIVE-2 minus NAIVE-1)": est[1], "of which RATE model (MODEL minus NAIVE-2)": est[2], "rate-model 90%": f"[{lo2:+.2f}, {hi2:+.2f}]"})
    print(f"\n{label}: SELECTION GAINS (wins per season from choosing the K best by the model instead of by last year's stats)\n" + pd.DataFrame(rows).round(2).to_string(index=False))


def main():
    H = hitters(); P = pitchers(); nh, npi = len(H), len(P); H = H.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True); P = P.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True); print(f"hitters pool {len(H)} (of {nh}), pitchers pool {len(P)} (of {npi}); rows with missing last-year inputs dropped for every method")
    accuracy(H, "HITTERS"); accuracy(P, "PITCHERS")
    selection(H, "HITTERS", (5, 9), (None, 40, 25)); selection(P, "PITCHERS", (3, 5), (None, 40, 25))
    rng = np.random.default_rng(21); b_h = np.array([picks(H.iloc[rng.integers(0, len(H), len(H))].reset_index(drop=True), 9, 40, rng, 300) for _ in range(300)]); b_p = np.array([picks(P.iloc[rng.integers(0, len(P), len(P))].reset_index(drop=True), 5, 30, rng, 300) for _ in range(300)])
    est_h, est_p = picks(H, 9, 40, rng, 2000), picks(P, 5, 30, rng, 2000); tot = b_h[:, 0] + b_p[:, 0]; print(f"\nCOMBINED REALISTIC SCENARIO: 9 hitters chosen from 40 candidates + 5 pitchers chosen from 30 candidates: MODEL minus NAIVE-1 = {est_h[0] + est_p[0]:+.2f} wins per season (hitters {est_h[0]:+.2f}, pitchers {est_p[0]:+.2f}); 90% interval [{np.percentile(tot, 5):+.2f}, {np.percentile(tot, 95):+.2f}]; P(> 0) {(tot > 0).mean():.2f}")


if __name__ == "__main__":
    main()
