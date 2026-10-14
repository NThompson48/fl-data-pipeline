"""
fl_pitcher_velo_k_test.py -- MODEL-CHANGE TEST: fastball velocity as a regressor in the production pitcher K% model (follow-up to the error analysis, 2026-10-13:
fb_velo was the one trait correlated with K% errors in every sample (t 4.6 / 3.4 / 3.0), sticky (split-half .96, y/y disattenuated .89) and better held out as a residual term).

Production harness: the frozen K% model is REFIT on the training transition (features full 2025 -> actual 1H2026) with fb_velo added
(K_rate_actual ~ naive_K_rate + fl_k_rate_bridge + wh_sva_level + spin_z_norm + extension + fb_velo_shrunk), then scored on fold 2 (features 2025 + 1H2026 -> actual 2H2026),
position players excluded, level-free, frozen standard inverse-variance weights (from production's residuals), pitcher bootstrap.  fb_velo = mean effective velocity of
four-seam / sinker pitches in the feature window, shrunk toward the population mean by n / (n + 8.4) fastballs (split-half n0); no fastball -> population mean.
Also reported: the low-history (< 150 BF in the window) and rookie (no 2025 appearance) groups, and the same refit with fb_velo REPLACING spin_z_norm + extension.
"""
import warnings

import numpy as np
import pandas as pd

import fl_error_analysis_common as ea
import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_production_projection as pp
import mlb_marcel as mm

warnings.filterwarnings("ignore")
N0_VELO = 8.4


def velo(t, mask):
    w = t[mask]; w = w[w["pitch_type"].isin(ea.FB_TYPES) & w["effective_velo"].notna()]
    g = w.groupby("pitcher_id")["effective_velo"].agg(["mean", "size"]); mu = np.average(g["mean"], weights=g["size"])
    return mu + (g["mean"] - mu) * g["size"] / (g["size"] + N0_VELO), mu


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); t = ea.load_tracking()
    v_tr, mu_tr = velo(t, t["year"] == 2025)
    v_te, mu_te = velo(t, (t["year"] == 2025) | ((t["year"] == 2026) & (t["date"] <= pd.Timestamp(bt.CUTOFF_2026))))
    df_tr, n0_k, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)], [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)
    for d, v, mu in ((df_tr, v_tr, mu_tr), (df_te, v_te, mu_te)):
        d["fb_velo"] = d["pitcher_id"].astype(str).map(v).fillna(mu)
    specs = {"production": pp.SPEC["K"], "+ fb_velo": pp.SPEC["K"] + ["fb_velo"], "fb_velo replaces spin + extension": [c for c in pp.SPEC["K"] if c not in ("spin_z_norm", "extension")] + ["fb_velo"]}
    tr = df_tr.dropna(subset=["K_rate_actual"] + pp.SPEC["K"]); w_tr = mm.reliability_weight(tr["PA_test"], n0_k).values
    te = df_te.dropna(subset=["K_rate_actual"] + pp.SPEC["K"]).copy()
    preds = {}
    for name, sp in specs.items():
        m = bt.fit_weighted(tr, "K_rate_actual", sp, w_tr); preds[name] = m.predict(te).to_numpy()
        print(f"{name}: fit n = {len(tr)}; " + ", ".join(f"{k} {m.params[k]:+.4f} (t {m.tvalues[k]:+.2f})" for k in sp))
    y = te["K_rate_actual"].to_numpy(); w = ew.weights_from_base(y, preds["production"], te["PA_test"].to_numpy())[0]
    c = lambda p: p + np.average(y - p, weights=w)
    hist = te["PA"].to_numpy(); rookie = ~te["pitcher_id"].astype(str).isin(t.loc[t["year"] == 2025, "pitcher_id"].unique())
    ids = te["pitcher_id"].astype(str).to_numpy(); rows = []
    for name in list(specs)[1:]:
        for grp, msk in (("all", np.ones(len(te), bool)), ("low-history (< 150 BF)", hist < 150), ("rookies (no 2025)", rookie.to_numpy())):
            yy, ww = y[msk], w[msk]; cc = lambda p: p + np.average(yy - p, weights=ww)
            r = er.compare(yy, cc(preds["production"][msk]), cc(preds[name][msk]), ww, ids[msk], n=1000)
            rows.append({"model": name, "group": grp, "n": int(msk.sum()), **{k: r[k] for k in ("RMSE old", "RMSE new", "dRMSE", "RMSE 90%", "P(RMSE better)", "MAE old", "MAE new", "dMAE", "MAE 90%", "P(MAE better)")}})
    print(f"\nFOLD 2 (2H2026 targets), n = {len(te)} pitchers, level-free, frozen weights\n" + er.table(rows, 5))
    d = pd.DataFrame({"diff": preds["+ fb_velo"] - preds["production"]}); print(f"\nprojection change (+ fb_velo vs production): mean |change| {d['diff'].abs().mean():.4f} K%, max {d['diff'].abs().max():.4f}")


if __name__ == "__main__":
    main()
