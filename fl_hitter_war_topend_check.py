"""
fl_hitter_war_topend_check.py -- top-end calibration of projected hitter WAR after the quality-aware playing-time model (user concern: wOBA / wRC+ look calibrated, WAR looks too low for the best hitters).
Backtest 2025-and-earlier data -> 2026 (production path via fl_hitter_pre2026.build_shifted: constants, PA model (now per game + quality), pitch terms from 2025 pitches; nothing from 2026), realised 2026 WAR on the projection's own
definition (fl_hitter_war_calibration.py).  OLD = the same projection with the previous per-game PA model (no quality term, fit on the same transitions); NEW = production.
Expected WAR = PA x (bat + fielding + position + replacement) / runs-per-win for both (same rates); realised WAR uses realised PA.  Reports: weighted-free RMSE and MAE of WAR (equal weights) with hitter-cluster bootstrap,
top-N projected vs realised, the >= 4 WAR hitters, and the coverage of the NEW simulated distribution (50 / 80 / 90 %, 10-90 range).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev
import fl_hitter_pre2026 as pre
import fl_hitter_projection as hp

warnings.filterwarnings("ignore")
N = 3000


def main():
    d, c, fit = pre.build_shifted()
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; lg26 = float(bt.league(q, "woba")[2026]); act = p[p["year"] == 2026].drop_duplicates("name_clean").set_index("name_clean")
    j = d.set_index("name_clean").join(act[["pa", "woba_n", "fielding_runs", "pos_adj_runs"]].rename(columns={"pa": "pa26", "woba_n": "woba26", "fielding_runs": "fld26", "pos_adj_runs": "pos26"}), how="inner")
    j = j[(j["pa26"] >= 20) & j["proj_woba"].notna()].copy(); n = len(j)
    # OLD playing time: per-game model without the quality term, same shifted panel / transitions
    ps = p[p["year"] <= 2025].copy(); ps["year"] = ps["year"] + 1; r = dev.pa_model(ps).dropna(subset=["pa_next"]).copy(); G = hp.GAMES
    r["x_pg"] = r["pa_t"] / r["t"].map(G); r["y_pg"] = r["pa_next"] / r["t"].map(lambda t: G[t + 1]); b = np.polyfit(r["x_pg"], r["y_pg"], 1)
    g_latest = j["last_season"].map(lambda y: hp.GAMES.get(y + 1, 96)); pa_old = np.where(j["gap"], hp.GAP_PA, np.maximum(hp.TARGET_GAMES * (b[1] + b[0] * j["pa_t"] / g_latest), 20.0))
    rate = (j["proj_woba"] - c["lg"]) / hp.SC + j["fld_rate"] + j["pos_rate"] + c["repl"]
    j["war_old"] = pa_old * rate / c["rpw"]; j["war_new"] = j["proj_pa"] * rate / c["rpw"]
    pa_r = j["pa26"].to_numpy(); real = pa_r * ((j["woba26"].to_numpy() - lg26) / hp.SC + j["fld26"].to_numpy() / pa_r + j["pos26"].to_numpy() / pa_r + c["repl"]) / c["rpw"]; j["real"] = real
    print(f"BACKTEST n = {n} hitters (>= 20 PA in 2026); new PA model: a {fit['pa']['a']:.3f}, b {fit['pa']['b']:.3f}, quality {fit['pa']['c_quality']:.1f}, resid SD {fit['pa']['resid_sd']:.0f} PA")
    ids = j.index.to_numpy(); w = np.ones(n)
    print(er.table([{"expected WAR": "all hitters", **er.compare(real, j["war_old"].to_numpy(), j["war_new"].to_numpy(), w, ids, n=800)},
                    {"expected WAR": "top 40 by NEW projected WAR", **er.compare(real[j["war_new"].rank(ascending=False) <= 40], j["war_old"].to_numpy()[j["war_new"].rank(ascending=False) <= 40], j["war_new"].to_numpy()[j["war_new"].rank(ascending=False) <= 40], np.ones(40), ids[j["war_new"].rank(ascending=False) <= 40], n=800)}], 4))
    for lab, col in (("OLD", "war_old"), ("NEW", "war_new")):
        s = j.sort_values(col, ascending=False)
        print(f"  {lab}: " + " | ".join(f"top {k}: projected {s[col].head(k).mean():.2f} vs realised {s['real'].head(k).mean():.2f}" for k in (10, 20, 30)) + f" | max projected {s[col].max():.2f}")
    print("  hitters with realised >= 4 WAR (n = %d): projected old / new mean %.2f / %.2f, realised mean %.2f" % ((j.real >= 4).sum(), j[j.real >= 4].war_old.mean(), j[j.real >= 4].war_new.mean(), j[j.real >= 4].real.mean()))
    # coverage of the NEW simulated distribution
    rng = np.random.default_rng(11); pa = np.clip(j["proj_pa"].to_numpy()[:, None] + fit["pa"]["resid_sd"] * rng.standard_normal((n, N)), 10, 700)
    woba = j["proj_woba"].to_numpy()[:, None] + np.sqrt(hp.LUCK_A) * rng.standard_normal((n, N)) + np.sqrt(hp.LUCK_B / pa) * rng.standard_normal((n, N))
    war = pa * ((woba - c["lg"]) / hp.SC + j["fld_rate"].to_numpy()[:, None] + hp.FLD_RESID_PER_PA * rng.standard_normal((n, N)) + j["pos_rate"].to_numpy()[:, None] + hp.POS_RESID_PER_PA * rng.standard_normal((n, N)) + c["repl"]) / c["rpw"]
    pit = (war < real[:, None]).mean(1); cov = {int(l * 100): float(((pit >= .5 - l / 2) & (pit <= .5 + l / 2)).mean()) for l in (.5, .8, .9)}; p10, p90 = np.percentile(war, 10, axis=1), np.percentile(war, 90, axis=1)
    print(f"  coverage of the NEW simulated distribution (nominal 50 / 80 / 90): {cov[50]:.3f} / {cov[80]:.3f} / {cov[90]:.3f}; above p90 {np.mean(real > p90):.0%}, below p10 {np.mean(real < p10):.0%} (nominal 10% each)")
    print("  EXPECTED number of hitters at >= 4 WAR under the NEW simulated distributions: %.1f (realised %d); at >= 5: %.1f (realised %d); at >= 3: %.1f (realised %d)" % ((war >= 4).mean(1).sum(), (real >= 4).sum(), (war >= 5).mean(1).sum(), (real >= 5).sum(), (war >= 3).mean(1).sum(), (real >= 3).sum()))
    s = j.assign(pit=pit, P=(war > 0).mean(1)).sort_values("war_new", ascending=False).head(10); print("  top 10 by NEW projected WAR: share above their own p90: %.0f%%; mean PIT %.2f" % (100 * (s["real"] > np.percentile(war, 90, axis=1)[j.index.get_indexer(s.index)]).mean(), s["pit"].mean()))
    print("  PIT of the >= 4 WAR hitters (old model had .84-.99):", j.assign(pit=pit)[j.real >= 4]["pit"].round(2).tolist())


if __name__ == "__main__":
    main()
