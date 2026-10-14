"""
fl_informative_pt_test.py -- INFORMATIVE PLAYING TIME in the WAR / P(WAR>0) simulations (user 2026-10-12, items 4 + 5: "the prior is probably too generous to
thin-history hitters"; "for pitchers this also matters").

Finding behind it: the rate projections are calibrated on the PLAYING-TIME-WEIGHTED target (what WAR = rate x playing time needs), but per player the thin
group does much worse (hitters: PA-weighted +.011 vs unweighted -.033 wOBA for < 50 prior PA; pitchers: P(WAR>0) ~7 points optimistic for arms that end with
few innings), because players who perform badly lose playing time.  The production simulations draw playing time INDEPENDENTLY of performance.

Joint model (one parameter per side): z ~ N(0, 1) = the player's standardised season performance error; playing time residual = rho x z + sqrt(1 - rho^2) x e;
realised rate = projected rate + z x SD(rate at the REALISED playing time).  Marginals of z and of playing time are unchanged; only their dependence is new.
  hitters:  PA = proj_PA + SD_PA x resid (production: normal, SD from the PA model, clipped [10, 700]); wOBA SD = sqrt(LUCK_A + LUCK_B / PA)
  pitchers: log(IP / proj_IP) = mu + s x resid (lognormal: the normal with SD ~30 IP is meaningless at a 24-IP projection); RA9 SD = sqrt(185.16 / IP)
rho = corr(z at realised playing time, playing-time residual) in the 2025 -> 2026 backtest (players who played: hitters >= 1 PA, pitchers >= 10 IP, the data
build's floor), 2-fold cross-fitted by player (rho fit on one half, scored on the other, 20 repeats).
Scored per player (unweighted -- the per-player view is what P(WAR>0) claims): Brier and log loss of P(WAR>0), reliability by history / workload group, coverage of
the 80% range; plus weighted RMSE / MAE of the projected (mean) WAR against realised WAR (standard: playing-time weights) to show the point projection is not hurt.
Hitter WAR here = batting + position + replacement (fielding dropped 2026-10-12, item 2; baserunning has no pre-2026 data).
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
N = 4000
RNG = np.random.default_rng(31)


def hitters_frame():
    import fl_hitter_baseline_backtest as bt
    import fl_hitter_model_dev as dev
    import fl_hitter_pre2026 as pre
    import fl_hitter_projection as hp
    d, c, fit = pre.build_shifted()
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; lg26 = float(bt.league(q, "woba")[2026])
    act = p[p["year"] == 2026].drop_duplicates("name_clean").set_index("name_clean")
    j = d.set_index("name_clean").join(act[["pa", "woba_n", "pos_adj_runs"]].rename(columns={"pa": "pa26", "woba_n": "woba26", "pos_adj_runs": "pos26"}), how="inner")
    j = j[(j["pa26"] >= 1) & j["proj_woba"].notna() & j["woba26"].notna()].copy()
    sdw = lambda n: np.sqrt(hp.LUCK_A + hp.LUCK_B / n)
    j["z"] = ((j["woba26"] - lg26) - (j["proj_woba"] - c["lg"])) / sdw(j["pa26"])
    j["r"] = (j["pa26"] - j["proj_pa"]) / fit["pa"]["resid_sd"]
    pos26 = (j["pos26"] / j["pa26"]).fillna(0.0)
    j["real_war"] = j["pa26"] * ((j["woba26"] - lg26) / hp.SC + pos26 + c["repl"]) / c["rpw"]
    j["rate_hat"] = (j["proj_woba"] - c["lg"]) / hp.SC + j["pos_rate"] + c["repl"]
    j["grp"] = pd.cut(j["pa_t"].where(~j["gap"], 0), [-1, 49, 149, 1e4], labels=["< 50 prior PA", "50-149", "150+"])
    return j, dict(sd_pa=fit["pa"]["resid_sd"], sc=hp.SC, rpw=c["rpw"], la=hp.LUCK_A, lb=hp.LUCK_B)


def sim_hitters(j, k, rho):
    n = len(j); z = RNG.standard_normal((n, N)); e = RNG.standard_normal((n, N))
    pa = np.clip(j["proj_pa"].to_numpy()[:, None] + k["sd_pa"] * (rho * z + np.sqrt(1 - rho ** 2) * e), 10, 700)
    rate = j["rate_hat"].to_numpy()[:, None] + z * np.sqrt(k["la"] + k["lb"] / pa) / k["sc"]
    return pa * rate / k["rpw"]


def pitchers_frame():
    import fl_pwar_calibration as cal
    import fl_ra9_trustmark_projection as ra9
    cal.RETURN_ALL = True
    b, _ = cal.main()
    b = b[(b["ip"] >= 10) & b["pred"].notna() & b["target"].notna()].copy()
    b["z"] = (b["target"] - b["pred"]) / np.sqrt(ra9.RA9_LUCK_VAR_COEF / b["ip"])
    lr = np.log(b["ip"] / b["ip_proj"]); b["lr"] = lr
    b["repl"] = ra9.LEAGUE_RA9_REPLACEMENT
    b["real_war"] = (b["repl"] - b["target"]) * b["ip"] / 9 / 10.706
    b["grp"] = pd.cut(b["ip_proj"], [0, 22, 45, 1e4], labels=["< 22 proj IP", "22-45", "45+"], right=False)
    return b, dict(luck=ra9.RA9_LUCK_VAR_COEF)


def sim_pitchers(b, k, rho, mu, s):
    n = len(b); z = RNG.standard_normal((n, N)); e = RNG.standard_normal((n, N))
    ip = np.clip(b["ip_proj"].to_numpy()[:, None] * np.exp(mu + s * (rho * z + np.sqrt(1 - rho ** 2) * e)), 1, 200)
    ra = b["pred"].to_numpy()[:, None] + z * np.sqrt(k["luck"] / ip)
    return (b["repl"].to_numpy()[:, None] - ra) * ip / 9 / 10.706


def score(df, war, label, wcol):
    P = (war > 0).mean(1); y = (df["real_war"].to_numpy() > 0).astype(float); Pc = np.clip(P, 1e-3, 1 - 1e-3)
    mean = war.mean(1); p10, p90 = np.percentile(war, 10, axis=1), np.percentile(war, 90, axis=1); rw = df["real_war"].to_numpy(); w = df[wcol].to_numpy()
    out = {"model": label, "Brier": np.mean((P - y) ** 2), "log loss": -np.mean(y * np.log(Pc) + (1 - y) * np.log(1 - Pc)),
           "mean P": P.mean(), "observed": y.mean(), "80% cover": np.mean((rw >= p10) & (rw <= p90)),
           "WAR RMSE (w)": np.sqrt(np.average((mean - rw) ** 2, weights=w)), "WAR MAE (w)": np.average(np.abs(mean - rw), weights=w)}
    by = pd.DataFrame({"grp": df["grp"].to_numpy(), "P": P, "y": y}).groupby("grp", observed=True).agg(n=("y", "size"), pred=("P", "mean"), obs=("y", "mean"))
    return out, by, P


def boot_diff(y, Pa, Pb, B=2000):
    rng = np.random.default_rng(3); n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n); d.append(np.mean((Pb[i] - y[i]) ** 2) - np.mean((Pa[i] - y[i]) ** 2))
    d = np.array(d); return np.percentile(d, 5), np.percentile(d, 95), np.mean(d < 0)


def crossfit_rho(df, a, b_, reps=20):
    rng = np.random.default_rng(8); rhos = np.full((reps, len(df)), np.nan)
    for r in range(reps):
        half = rng.random(len(df)) < .5
        for m in (half, ~half):
            rhos[r, (~m).nonzero()[0]] = np.corrcoef(df.loc[m, a], df.loc[m, b_])[0, 1]
    return float(np.nanmean(rhos)), float(np.nanstd(rhos))


def main():
    print("=" * 100 + "\nHITTERS (2025-and-earlier data -> realised 2026, hitters with >= 1 PA in 2026)\n" + "=" * 100)
    j, k = hitters_frame()
    rho_h = np.corrcoef(j["z"], j["r"])[0, 1]; rho_cf, rho_sd = crossfit_rho(j, "z", "r")
    print(f"n = {len(j)}; corr(standardised wOBA error, PA residual) rho = {rho_h:+.3f} (2-fold cross-fit mean {rho_cf:+.3f}, SD across folds {rho_sd:.3f})")
    print("per group: unweighted mean z / PA-weighted mean z:"); print(j.groupby("grp", observed=True).apply(lambda g: pd.Series({"n": len(g), "mean z": g["z"].mean(), "PA-weighted z": np.average(g["z"], weights=g["pa26"]), "rho": np.corrcoef(g["z"], g["r"])[0, 1]})).round(3).to_string())
    rows, bys, Ps = [], {}, {}
    for lab, rho in (("independent (production)", 0.0), ("informative PT", rho_cf)):
        war = sim_hitters(j, k, rho); o, by, P = score(j, war, lab, "pa26"); rows.append(o); bys[lab] = by; Ps[lab] = P
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    for lab, by in bys.items():
        print(f"  reliability by history, {lab}:"); print(by.round(3).to_string())
    y = (j["real_war"].to_numpy() > 0).astype(float); lo, hi, pb = boot_diff(y, Ps["independent (production)"], Ps["informative PT"])
    print(f"  Brier change (new - old) 90% [{lo:+.4f}, {hi:+.4f}], P(better) {pb:.2f}")

    print("\n" + "=" * 100 + "\nPITCHERS (2025 -> 2026 backtest, arms with >= 10 IP in 2026)\n" + "=" * 100)
    b, kp = pitchers_frame()
    mu, s = float(b["lr"].mean()), float(b["lr"].std())
    b["r"] = (b["lr"] - mu) / s
    rho_p = np.corrcoef(b["z"], b["r"])[0, 1]; rho_pcf, rho_psd = crossfit_rho(b, "z", "r")
    print(f"n = {len(b)}; log(IP / proj IP) mean {mu:+.3f} SD {s:.3f}; rho = {rho_p:+.3f} (cross-fit {rho_pcf:+.3f}, SD {rho_psd:.3f})")
    from scipy.stats import norm
    P_an = np.asarray(norm.cdf((b["repl"] - b["pred"]) / np.sqrt(kp["luck"] / b["ip_proj"])))
    yb = (b["real_war"].to_numpy() > 0).astype(float)
    rows = [{"model": "analytic at projected IP (production)", "Brier": np.mean((P_an - yb) ** 2), "mean P": P_an.mean(), "observed": yb.mean()}]
    bys = {"analytic at projected IP (production)": pd.DataFrame({"grp": b["grp"].to_numpy(), "P": P_an, "y": yb}).groupby("grp", observed=True).agg(n=("y", "size"), pred=("P", "mean"), obs=("y", "mean"))}
    Ps = {"analytic at projected IP (production)": P_an}
    for lab, rho in (("simulated IP, independent", 0.0), ("informative PT", rho_pcf)):
        war = sim_pitchers(b, kp, rho, mu, s); o, by, P = score(b, war, lab, "ip"); rows.append(o); bys[lab] = by; Ps[lab] = P
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    for lab, by in bys.items():
        print(f"  reliability by projected IP, {lab}:"); print(by.round(3).to_string())
    lo, hi, pb = boot_diff(yb, Ps["analytic at projected IP (production)"], Ps["informative PT"])
    print(f"  Brier change (informative - production) 90% [{lo:+.4f}, {hi:+.4f}], P(better) {pb:.2f}")
    import json
    json.dump({"rho_hitters": rho_cf, "rho_pitchers": rho_pcf, "pitcher_log_ip_mu": mu, "pitcher_log_ip_sd": s}, open("output/informative_pt_params.json", "w"), indent=1)


if __name__ == "__main__":
    main()
