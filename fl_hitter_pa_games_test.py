"""
fl_hitter_pa_games_test.py -- season-length-aware hitter playing time.
Regular-season length (user: 2025 = 96 games, 2026 = 102, 2027 = 102; 2022-24 taken as 96: the maximum games played by any hitter on a team was 96 in each of those years and the median of the team maxima, 91.5-94, matches
2025's 91.5 in the tracked data).
OLD (production): PA_next = a + b PA_t on hitters who return (any PA next year), season length ignored.
NEW: PA per team game: PA_next / G_next = a' + b' PA_t / G_t, projected PA = G_target (a' + b' PA_t / G_latest).
Leave-one-transition-out, returners only, equal weights (playing time is not a binomial rate, so the inverse-variance weights do not apply), hitter-cluster bootstrap, RMSE and MAE (reporting standard); the 2025 -> 2026
transition (the only one with a longer season) is shown separately.  Residual SD of the new model in PA at 102 games is what the WAR simulation should use.
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_model_dev as dev

warnings.filterwarnings("ignore")
G = {2022: 96, 2023: 96, 2024: 96, 2025: 96, 2026: 102}


def main():
    p = dev.load(); d = dev.pa_model(p); r = d.dropna(subset=["pa_next"]).copy(); r["g_t"] = r["t"].map(G); r["g_next"] = r["t"].map(lambda t: G[t + 1])
    r["x_pg"] = r["pa_t"] / r["g_t"]; r["y_pg"] = r["pa_next"] / r["g_next"]; print(f"{len(r)} returning hitter-transitions (any PA next year); transitions: {r.groupby('t').size().to_dict()}")
    Y, O, N, ids, fold = [], [], [], [], []; coefs = {}
    for T in (2022, 2023, 2024, 2025):
        tr, te = r[r["t"] != T], r[r["t"] == T]
        bo = np.polyfit(tr["pa_t"], tr["pa_next"], 1); bn = np.polyfit(tr["x_pg"], tr["y_pg"], 1); coefs[T] = (bo, bn)
        Y.append(te["pa_next"].to_numpy()); O.append(np.maximum(bo[1] + bo[0] * te["pa_t"].to_numpy(), 20)); N.append(np.maximum(te["g_next"].to_numpy() * (bn[1] + bn[0] * te["x_pg"].to_numpy()), 20)); ids.append(te["name_clean"].to_numpy()); fold.append(np.full(len(te), T))
    y, o, n, i, f = map(np.concatenate, (Y, O, N, ids, fold)); w = np.ones(len(y))
    rows = [{"scope": "all four transitions (LOTO)", **er.compare(y, o, n, w, i, n=800)}]
    m = f == 2025; rows.append({"scope": "2025 -> 2026 (102-game season) only", **er.compare(y[m], o[m], n[m], w[m], i[m], n=800)})
    m = f != 2025; rows.append({"scope": "the three 96 -> 96 transitions", **er.compare(y[m], o[m], n[m], w[m], i[m], n=800)})
    print("PA next year, returners: old = production PA regression, new = per-game model\n" + er.table(rows, 3))
    print("\nbias (actual - projected PA) on 2025 -> 2026:  old %+.1f   new %+.1f" % ((y[f == 2025] - o[f == 2025]).mean(), (y[f == 2025] - n[f == 2025]).mean()))
    bn_all = np.polyfit(r["x_pg"], r["y_pg"], 1); res = r["y_pg"] - (bn_all[1] + bn_all[0] * r["x_pg"]); bo_all = np.polyfit(r["pa_t"], r["pa_next"], 1)
    print(f"final fits: OLD PA_next = {bo_all[1]:.1f} + {bo_all[0]:.3f} PA_t (resid SD {np.std(r['pa_next'] - (bo_all[1] + bo_all[0] * r['pa_t']), ddof=2):.1f}); NEW PA/game_next = {bn_all[1]:.3f} + {bn_all[0]:.3f} PA_t/G_t (resid SD {res.std(ddof=2):.3f} per game = {res.std(ddof=2) * 102:.1f} PA at 102 games)")
    for pa_t in (150, 300, 400, 470):
        print(f"  a hitter with {pa_t} PA in a 102-game season: old projects {bo_all[1] + bo_all[0] * pa_t:.0f}, new projects {102 * (bn_all[1] + bn_all[0] * pa_t / 102):.0f}")


if __name__ == "__main__":
    main()
