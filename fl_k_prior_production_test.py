"""
fl_k_prior_production_test.py -- standard before/after report for the production change 'network stuff as a fading prior, (c, n0_s) fit jointly' (user 2026-10-24: "Add it and push")
BEFORE = additive XGBoost K% model (what was published); AFTER = fl_k_prior.fit with the 3-seed network stuff.  Also shown: additive network, the earlier grid-fit prior, no-stuff control.
Protocol identical to fl_wh_sva_prior_k_test.py (fit on full 2025 -> 1H2026, fold 2 = 2025+1H2026 -> 2H2026, position players excluded, level-free, frozen standard inverse-variance weights from BEFORE's residuals, pitcher bootstrap 1,000), RMSE AND MAE.
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_k_prior as kp
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_production_projection as pp
import fl_wh_sva_prior_k_test as t
import mlb_marcel as mm

warnings.filterwarnings("ignore")
ADD = pp.SPEC["K"]


def frames(raw, est):
    fp.WH_SVA_ESTIMATOR = est
    tr, n0_k, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    f2, n0_k2, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)], [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)
    return tr, n0_k, f2, n0_k2


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); out = {}; P = {}
    lg_tr = t.league_k(raw, [(bt.START_2025, bt.END_2025)]); lg_f2 = t.league_k(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)])
    for est in ("xgb", "torch3", "swstr_ct"):
        tr, n0_k, f2, n0_k2 = frames(raw, est); d = tr.dropna(subset=["K_rate_actual"] + ADD).copy(); w = mm.reliability_weight(d["PA_test"], n0_k).values
        x = f2.dropna(subset=ADD + ["K_rate_actual"]).copy(); x["lg_k_frame"] = lg_f2; x["n0_k_frame"] = n0_k2
        m_add = bt.fit_weighted(d, "K_rate_actual", ADD, w); m_pr = kp.fit(d, n0_k, lg_tr, w)
        print(f"[{est}] NLS prior: c={m_pr.c:.4f} (OLS slope {m_pr.c0:.4f}, x{m_pr.c / m_pr.c0:.2f}), n0_s={m_pr.n0_s:.1f}; params {m_pr.params.round(4).to_dict()}", flush=True)
        P[est] = pd.DataFrame({"additive": m_add.predict(x).to_numpy(), "prior": m_pr.predict(x).to_numpy()}, index=x["pitcher_id"].to_numpy()); out[est] = {"c": m_pr.c, "n0_s": m_pr.n0_s, "c0": m_pr.c0}
        P[est + "_meta"] = x.set_index("pitcher_id")[["K_rate_actual", "PA_test", "PA"]]
    ids = sorted(set(P["xgb"].index) & set(P["torch3"].index) & set(P["swstr_ct"].index)); meta = P["xgb_meta"].loc[ids]; y = meta["K_rate_actual"].to_numpy(); n_t = meta["PA_test"].to_numpy(); ids_a = np.array(ids)
    preds = {"BEFORE: additive XGBoost (published)": P["xgb"].loc[ids, "additive"].to_numpy(), "additive network": P["torch3"].loc[ids, "additive"].to_numpy(), "prior XGBoost (NLS)": P["xgb"].loc[ids, "prior"].to_numpy(), "AFTER: prior network (NLS)": P["torch3"].loc[ids, "prior"].to_numpy(),
             "additive swstr_ct": P["swstr_ct"].loc[ids, "additive"].to_numpy(), "prior swstr_ct (NLS)": P["swstr_ct"].loc[ids, "prior"].to_numpy()}
    base = preds["BEFORE: additive XGBoost (published)"]; w = ew.weights_from_base(y, base, n_t)[0]; cen = lambda p: p + np.average(y - p, weights=w); rows = []
    for lab, p in preds.items():
        rm, ma = er.wmetrics(y, cen(p), w); r = {"model": lab, "RMSE": rm, "MAE": ma}
        if not lab.startswith("BEFORE"):
            c = er.compare(y, cen(base), cen(p), w, ids_a, n=1000); r.update({"dRMSE": c["dRMSE"], "RMSE 90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "MAE 90%": c["MAE 90%"], "P(MAE better)": c["P(MAE better)"]})
        rows.append(r)
    pd.set_option("display.width", 250); print(f"\nK%: fold 2 (2H2026 targets), n = {len(ids)}, level-free, frozen weights\n" + er.table(rows, 5))
    tier = pd.cut(meta["PA"], [0, 100, 250, 1e9], labels=["thin (<100 PA)", "mid (100-250)", "thick (>250)"]).to_numpy(); rows3 = []
    for tn in ["thin (<100 PA)", "mid (100-250)", "thick (>250)"]:
        m = tier == tn
        if m.sum() >= 8:
            r = {"PA tier": tn, "n": int(m.sum())}
            for lab, p in preds.items():
                r[lab.split(":")[0]] = float(np.sqrt(np.average((y[m] - cen(p)[m]) ** 2, weights=w[m])))
            rows3.append(r)
    print("\nWEIGHTED RMSE BY PA TIER\n" + pd.DataFrame(rows3).round(5).to_string(index=False))
    cmp = er.compare(y, cen(preds["AFTER: prior network (NLS)"]), cen(preds["prior swstr_ct (NLS)"]), w, ids_a, n=1000)
    print("\nprior swstr_ct vs prior network (swing-based): dRMSE %+.5f %s P=%.3f; dMAE %+.5f %s P=%.3f" % (cmp["dRMSE"], cmp["RMSE 90%"], cmp["P(RMSE better)"], cmp["dMAE"], cmp["MAE 90%"], cmp["P(MAE better)"]))
    json.dump({"fit": out, "table": rows, "tiers": rows3}, open("output/k_prior_production_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
