"""
fl_posplayer_refit_check.py -- does excluding position players from the frozen pitcher K% / BB% fit help or hurt held out?
Fit A = production fit WITH position players (original), fit B = WITHOUT (flagged pitchers dropped from the pitch data).  Both are scored on fold 2 of Experiment 2
(features 2025 + 1H2026 -> actual 2H2026) on the SAME non-flagged pitchers, weighted RMSE with inverse-variance weights (fl_eval_weights, frozen from the naive model), pitcher bootstrap for the difference.
"""
import warnings
import numpy as np
import pandas as pd
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_production_projection as pp
import mlb_marcel as mm

warnings.filterwarnings("ignore"); rng = np.random.default_rng(5)


def fold2(raw, m_k, m_bb, n0_bridge_bb):
    df, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)], [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)
    pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025), fp.filter_period(raw, bt.START_2026, bt.CUTOFF_2026)], ignore_index=True); t = pool[pool["is_terminal"]]
    lg_bb = (t["k_or_bb"] == "walk").mean()
    df["BB_raw"] = df["naive_BB_rate"] * (df["PA"] + n0_bb) - n0_bb * lg_bb
    df["blended_BB_rate"] = mm.blended_rate(df["BB_raw"], df["PA"], n0_bb, lg_bb, n0_bridge_bb, df["fl_bb_rate_bridge"])
    dk = df.dropna(subset=pp.SPEC["K"] + ["K_rate_actual"]).copy(); db = df.dropna(subset=pp.BB_SPEC_BLENDED + ["BB_rate_actual"]).copy()
    dk["pred"] = m_k.predict(dk); db["pred"] = m_bb.predict(db)
    return dk, db


def main():
    raw = fp.load_fl_pitches_cached(); flagged = set(pd.read_parquet("output/position_player_pitchers.parquet").query("is_position_player")["pitcher_id"])
    rawx = ppf.exclude(raw)
    fits = {"A with position players": pp.fit_frozen_model(raw), "B without": pp.fit_frozen_model(rawx)}
    res = {}
    for name, (m_k, m_bb, n0_k, n0_bb, n0b, rho) in fits.items():
        res[name] = fold2(rawx, m_k, m_bb, n0b)
    for stat, idx, tcol in (("K%", 0, "K_rate_actual"), ("BB%", 1, "BB_rate_actual")):
        a, b = res["A with position players"][idx], res["B without"][idx]
        m = a.merge(b[["pitcher_id", "pred"]], on="pitcher_id", suffixes=("_a", "_b")); y = m[tcol].to_numpy(); n = m["PA_test"].to_numpy()
        w = ew.weights_from_base(y, m["pred_a"].to_numpy(), n)[0]
        def r(p, ii=slice(None), centre=True):                       # centre=True: both fits get the same free level (the production projection is recentred to the league level anyway)
            e = y[ii] - p[ii]; c = np.average(e, weights=w[ii]) if centre else 0.0
            return float(np.sqrt(np.average((e - c) ** 2, weights=w[ii])))
        ids = m["pitcher_id"].to_numpy(); by = pd.Series(np.arange(len(m))).groupby(ids).apply(np.array); ug = by.index.to_numpy(); d = []
        for _ in range(1000):
            ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); d.append(r(m["pred_b"].to_numpy(), ii) - r(m["pred_a"].to_numpy(), ii))
        d = np.array(d); print(f"{stat}: n={len(m)} pitchers (2H2026 targets, non-flagged); (level-free) RMSE A {r(m['pred_a'].to_numpy()):.5f}  [raw {r(m['pred_a'].to_numpy(), centre=False):.5f}]  B {r(m['pred_b'].to_numpy()):.5f}  [raw {r(m['pred_b'].to_numpy(), centre=False):.5f}]  B-A {r(m['pred_b'].to_numpy()) - r(m['pred_a'].to_numpy()):+.5f}  90% [{np.percentile(d, 5):+.5f}, {np.percentile(d, 95):+.5f}]  P(B better) {np.mean(d < 0):.2f}")


if __name__ == "__main__":
    main()
