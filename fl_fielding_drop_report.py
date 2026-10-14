"""
fl_fielding_drop_report.py -- model-change report for DROPPING the fielding term from projected hitter WAR (user 2026-10-12: "drop fielding data until infield data
exists"; infield fielding is 0 for 90% of SS, ~72% of 2B / 3B and 50% of 1B even among 2026 regulars, so the term only ever credits outfielders and catchers).

Backtest 2025-and-earlier -> 2026 (fl_hitter_pre2026.build_shifted, the production path; same rows for both versions), hitters with >= 20 PA in 2026.  Point WAR =
projected PA x per-PA rate (old: bat + fielding + position + replacement; new: the same without fielding).  Two targets, because the right one is the question:
  (A) realised WAR WITH realised fielding (the full value: does dropping lose real signal?)
  (B) realised WAR WITHOUT fielding (the new definition)
Weighted RMSE / MAE with realised-PA weights, hitter bootstrap (2,000).  Also by position group (OF + C, where fielding is measured, vs IF).
"""
import warnings

import numpy as np
import pandas as pd

import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev
import fl_hitter_pre2026 as pre
import fl_hitter_projection as hp

warnings.filterwarnings("ignore")


def main():
    d, c, fit = pre.build_shifted()
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; lg26 = float(bt.league(q, "woba")[2026])
    act = p[p["year"] == 2026].drop_duplicates("name_clean").set_index("name_clean")
    j = d.set_index("name_clean").join(act[["pa", "woba_n", "fielding_runs", "pos_adj_runs", "primary_pos"]].rename(columns={"pa": "pa26", "woba_n": "woba26", "fielding_runs": "fld26", "pos_adj_runs": "pos26", "primary_pos": "pos_26"}), how="inner", rsuffix="_a")
    j = j[(j["pa26"] >= 20) & j["proj_woba"].notna()].copy()
    base = (j["proj_woba"] - c["lg"]) / hp.SC + j["pos_rate"] + c["repl"]
    old = j["proj_pa"] * (base + j["fld_rate"]) / c["rpw"]; new = j["proj_pa"] * base / c["rpw"]
    rbat = j["pa26"] * ((j["woba26"] - lg26) / hp.SC + j["pos26"].fillna(0) / j["pa26"] + c["repl"]) / c["rpw"]
    tgt = {"(A) realised WAR incl. fielding": rbat + j["fld26"].fillna(0) / c["rpw"], "(B) realised WAR excl. fielding": rbat}
    w = j["pa26"].to_numpy(); grp = np.where(j["pos_26"].isin(["LF", "CF", "RF", "C"]), "OF + C", np.where(j["pos_26"].notna(), "IF", "no 2026 position"))
    rng = np.random.default_rng(2); n = len(j)
    print(f"n = {len(j)} hitters; projected fielding: non-zero for {(j['fld_rate'] != 0).mean():.0%} (mean |fld| {600 * j['fld_rate'].abs().mean():.1f} runs / 600 PA)")
    for lab, y in tgt.items():
        y = y.to_numpy(); o, nw = old.to_numpy(), new.to_numpy()
        f = lambda pr, i: (np.sqrt(np.average((pr[i] - y[i]) ** 2, weights=w[i])), np.average(np.abs(pr[i] - y[i]), weights=w[i]))
        allidx = np.arange(n); (r0, m0), (r1, m1) = f(o, allidx), f(nw, allidx)
        bs = np.array([[a - b for a, b in zip(f(nw, i), f(o, i))] for i in (rng.integers(0, n, n) for _ in range(2000))])
        print(f"\n{lab}: RMSE {r0:.4f} -> {r1:.4f} ({r1 - r0:+.4f}, 90% [{np.percentile(bs[:, 0], 5):+.4f}, {np.percentile(bs[:, 0], 95):+.4f}], P(better) {np.mean(bs[:, 0] < 0):.2f}); "
              f"MAE {m0:.4f} -> {m1:.4f} ({m1 - m0:+.4f}, [{np.percentile(bs[:, 1], 5):+.4f}, {np.percentile(bs[:, 1], 95):+.4f}], P(better) {np.mean(bs[:, 1] < 0):.2f})")
        for g in ("OF + C", "IF"):
            m = grp == g; idx = np.nonzero(m)[0]; (a0, b0), (a1, b1) = f(o, idx), f(nw, idx)
            print(f"   {g:8s} (n={m.sum():3d}): RMSE {a0:.4f} -> {a1:.4f}; MAE {b0:.4f} -> {b1:.4f}; mean error old {np.average(o[m] - y[m], weights=w[m]):+.3f} new {np.average(nw[m] - y[m], weights=w[m]):+.3f}")


if __name__ == "__main__":
    main()
