"""
fl_hitter_war_calibration.py -- is the hitter WAR distribution (range and P(WAR>0)) calibrated, by playing time?  Out of sample 2025 -> 2026.

Projection: the production path run on 2025-and-earlier data only (fl_hitter_pre2026.build_shifted: constants, PA model, pitch-level terms from 2025 pitches), then the production WAR simulation
(PA ~ N(proj_PA, resid SD) clipped [10, 700]; wOBA luck a + b/PA; fielding and position residuals), 3,000 draws per hitter, conditional on playing (the return probability is a separate column).
Outcome: the hitter's REALISED 2026 WAR on the projection's own definition: realised PA x [(park-neutral wOBA_2026 - league_2026)/1.0175 + realised fielding runs/PA + realised position runs/PA + replacement] / runs-per-win,
with the 2025 replacement and runs-per-win constants the projection used (the league level is the realised one, because WAR is measured against the league of the season played).
Checks: PIT of the realised WAR inside each hitter's simulated distribution (coverage of the central 50 / 80 / 90 % intervals and the 10-90 range), reliability of P(WAR>0), by projected-PA bucket and by realised PA;
then a decomposition of the error (batting vs fielding vs position vs playing time) to see which layer drives any miscalibration.
"""
import warnings

import numpy as np
import pandas as pd

import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev
import fl_hitter_pre2026 as pre
import fl_hitter_projection as hp

warnings.filterwarnings("ignore")
N = 3000


def wilson(k, n, z=1.645):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    d, c, fit = pre.build_shifted()
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]
    lg26 = float(bt.league(q, "woba")[2026])
    act = p[p["year"] == 2026].drop_duplicates("name_clean").set_index("name_clean")
    d = d.set_index("name_clean"); j = d.join(act[["pa", "woba_n", "fielding_runs", "pos_adj_runs"]].rename(columns={"pa": "pa26", "woba_n": "woba26", "fielding_runs": "fld26", "pos_adj_runs": "pos26"}), how="inner")
    j = j[(j["pa26"] >= 20) & j["proj_woba"].notna()].copy(); print(f"{len(j)} hitters with a projection and >= 20 PA in 2026; 2025 league wOBA {c['lg']:.4f} (projection baseline) vs 2026 {lg26:.4f} (realised); rpw {c['rpw']:.2f}, repl {c['repl']:.4f}/PA")
    rng = np.random.default_rng(11); n = len(j)
    pa = np.clip(j["proj_pa"].to_numpy()[:, None] + fit["pa"]["resid_sd"] * rng.standard_normal((n, N)), 10, 700)
    woba = j["proj_woba"].to_numpy()[:, None] + np.sqrt(hp.LUCK_A) * rng.standard_normal((n, N)) + np.sqrt(hp.LUCK_B / pa) * rng.standard_normal((n, N))
    fld = j["fld_rate"].to_numpy()[:, None] + hp.FLD_RESID_PER_PA * rng.standard_normal((n, N)); pos = j["pos_rate"].to_numpy()[:, None] + hp.POS_RESID_PER_PA * rng.standard_normal((n, N))
    war = pa * ((woba - c["lg"]) / hp.SC + fld + pos + c["repl"]) / c["rpw"]
    pa_r = j["pa26"].to_numpy(); real = pa_r * ((j["woba26"].to_numpy() - lg26) / hp.SC + j["fld26"].to_numpy() / pa_r + j["pos26"].to_numpy() / pa_r + c["repl"]) / c["rpw"]
    j["real_war"] = real; j["pit"] = (war < real[:, None]).mean(1) + 0.5 * (war == real[:, None]).mean(1); j["P"] = (war > 0).mean(1); j["y"] = (real > 0).astype(float)
    j["p10"], j["p90"] = np.percentile(war, 10, axis=1), np.percentile(war, 90, axis=1); j["mean"] = war.mean(1)
    j["pab"] = pd.cut(j["proj_pa"], [0, 100, 200, 300, 1000], labels=["<100 proj PA", "100-200", "200-300", "300+"])
    j["pa26b"] = pd.cut(j["pa26"], [0, 60, 150, 300, 1000], labels=["20-60 PA", "60-150", "150-300", "300+"])
    def cover(g):
        r = {"n": len(g)}
        for lev in (.5, .8, .9):
            r[f"{int(lev*100)}%"] = float(((g["pit"] >= .5 - lev / 2) & (g["pit"] <= .5 + lev / 2)).mean())
        r["10-90 range"] = float(((g["real_war"] >= g["p10"]) & (g["real_war"] <= g["p90"])).mean()); r["mean PIT"] = g["pit"].mean(); r["mean real - mean proj WAR"] = (g["real_war"] - g["mean"]).mean(); return pd.Series(r)
    print("\nCOVERAGE of the simulated WAR distribution (nominal 50 / 80 / 90; the 10-90 range is nominally 80%)")
    for lab, key in (("by PROJECTED PA", "pab"), ("by REALISED PA (conditioning on how much he played; informative about the playing-time layer)", "pa26b")):
        print(f"  {lab}"); t = pd.concat([cover(j).rename("all").to_frame().T, j.groupby(key, observed=True).apply(cover)]); print(t.round(3).to_string())
    print("\n  PIT histogram, deciles (uniform = .10):", np.round(np.histogram(j["pit"], bins=np.linspace(0, 1, 11))[0] / n, 3).tolist())
    print("\nRELIABILITY of P(WAR>0), 90% Wilson intervals")
    for lab, g0 in [("all", j)] + [(str(k), g) for k, g in j.groupby("pab", observed=True)]:
        g0 = g0.assign(bin=pd.cut(g0["P"], [-.01, .2, .4, .6, .8, .95, 1.0])); rows = []
        for b, g in g0.groupby("bin", observed=True):
            lo, hi = wilson(g["y"].sum(), len(g)); rows.append({"P bin": str(b), "n": len(g), "mean predicted": g["P"].mean(), "observed": g["y"].mean(), "90% interval": f"[{lo:.2f}, {hi:.2f}]", "inside": bool(lo <= g["P"].mean() <= hi)})
        print(f"  {lab} (n={len(g0)}; mean predicted {g0['P'].mean():.3f}, observed {g0['y'].mean():.3f})"); print(pd.DataFrame(rows).round(3).to_string(index=False))
    # decomposition of the error in realised-minus-expected terms, per realised PA
    j["bat_err"] = ((j["woba26"] - lg26) - (j["proj_woba"] - c["lg"])) / hp.SC; j["fld_err"] = j["fld26"] / j["pa26"] - j["fld_rate"]; j["pos_err"] = j["pos26"] / j["pa26"] - j["pos_rate"]
    sd_bat = np.sqrt(hp.LUCK_A + hp.LUCK_B / j["pa26"]) / hp.SC
    print("\nDECOMPOSITION (realised minus projected, runs per PA), standardised by the model's own SD at realised PA:")
    print(f"  batting:   mean error {j['bat_err'].mean():+.4f} runs/PA ({j['bat_err'].mean() * 600:+.1f} per 600 PA); SD of z = {(j['bat_err'] / sd_bat).std():.2f}; mean z = {(j['bat_err'] / sd_bat).mean():+.2f}")
    print(f"  fielding:  mean error {j['fld_err'].mean() * 600:+.1f} runs per 600 PA; SD of error {j['fld_err'].std() * 600:.1f} vs model residual {hp.FLD_RESID_PER_PA * 600:.1f}")
    print(f"  position:  mean error {j['pos_err'].mean() * 600:+.1f} runs per 600 PA; SD of error {j['pos_err'].std() * 600:.1f} vs model residual {hp.POS_RESID_PER_PA * 600:.1f}")
    print(f"  playing time: realised PA vs projected: mean {(j['pa26'] - j['proj_pa']).mean():+.0f} PA, RMSE {np.sqrt(((j['pa26'] - j['proj_pa']) ** 2).mean()):.0f} vs model SD {fit['pa']['resid_sd']:.0f}")
    j.to_parquet("output/hitter_war_calibration_2026.parquet")


if __name__ == "__main__":
    main()
