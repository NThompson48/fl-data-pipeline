"""
fl_k_both_test.py -- BOTH stuff measures in the K% model  (user 2026-10-24: "What about prior network and additive swstr_ct? both")
Variants (standard K% harness as fl_k_prior_production_test.py: fit full 2025 -> 1H2026, fold 2 -> 2H2026, level-free, frozen weights from the published model, pitcher bootstrap 1,000; RMSE AND MAE):
  published (additive XGBoost) | prior network | additive swstr_ct | BOTH-A: swing-network stuff as the fading prior (blended_K_stuff) + swstr_ct level as an additive regressor | BOTH-B: both levels additive (swing network + swstr_ct)
Output: output/k_both_test.json
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
import fl_k_prior_production_test as pt
import fl_wh_sva_prior_k_test as t
import mlb_marcel as mm

warnings.filterwarnings("ignore")
ADD = pp.SPEC["K"]


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); lg_tr = t.league_k(raw, [(bt.START_2025, bt.END_2025)]); lg_f2 = t.league_k(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)])
    F = {est: pt.frames(raw, est) for est in ("xgb", "torch3", "swstr_ct")}
    tr, n0_k, f2, n0_k2 = F["torch3"]; sw_tr = F["swstr_ct"][0][["pitcher_id", "wh_sva_level"]].rename(columns={"wh_sva_level": "swstr_level"}); sw_f2 = F["swstr_ct"][2][["pitcher_id", "wh_sva_level"]].rename(columns={"wh_sva_level": "swstr_level"})
    xg_tr, xg_f2 = F["xgb"][0], F["xgb"][2]; sw_spec = [f if f != "wh_sva_level" else "swstr_level" for f in ADD]
    d = tr.merge(sw_tr, on="pitcher_id").dropna(subset=["K_rate_actual", "swstr_level"] + ADD).copy(); w = mm.reliability_weight(d["PA_test"], n0_k).values
    x = f2.merge(sw_f2, on="pitcher_id").dropna(subset=["K_rate_actual", "swstr_level"] + ADD).copy(); x["lg_k_frame"] = lg_f2; x["n0_k_frame"] = n0_k2
    dx = xg_tr.dropna(subset=["K_rate_actual"] + ADD).copy(); wx = mm.reliability_weight(dx["PA_test"], F["xgb"][1]).values; m_pub = bt.fit_weighted(dx, "K_rate_actual", ADD, wx); xx = xg_f2.dropna(subset=ADD + ["K_rate_actual"]).copy()
    m_prior = kp.fit(d, n0_k, lg_tr, w); m_sw = bt.fit_weighted(d, "K_rate_actual", sw_spec, w); m_bothA = kp.fit(d, n0_k, lg_tr, w, extra=["swstr_level"]); m_bothB = bt.fit_weighted(d, "K_rate_actual", ADD + ["swstr_level"], w)
    print("BOTH-A (prior network + additive swstr_ct) params:", m_bothA.params.round(4).to_dict(), f"c={m_bothA.c:.3f} n0_s={m_bothA.n0_s:.1f}\nBOTH-B (both additive) params:", m_bothB.params.round(4).to_dict(), "\nBOTH-B t-values:", m_bothB.tvalues.round(2).to_dict(), flush=True)
    P = {"published": pd.Series(m_pub.predict(xx).to_numpy(), index=xx["pitcher_id"].to_numpy())}
    for lab, m in (("prior network", m_prior), ("additive swstr_ct", m_sw), ("BOTH-A", m_bothA), ("BOTH-B", m_bothB)):
        P[lab] = pd.Series(m.predict(x).to_numpy(), index=x["pitcher_id"].to_numpy())
    ids = sorted(set(P["published"].index) & set(x["pitcher_id"])); meta = x.set_index("pitcher_id").loc[ids]; y = meta["K_rate_actual"].to_numpy(); n_t = meta["PA_test"].to_numpy(); ids_a = np.array(ids)
    preds = {k: v.loc[ids].to_numpy() for k, v in P.items()}; base = preds["published"]; w2 = ew.weights_from_base(y, base, n_t)[0]; cen = lambda p: p + np.average(y - p, weights=w2); rows = []
    for lab, p in preds.items():
        rm, ma = er.wmetrics(y, cen(p), w2); r = {"model": lab, "RMSE": rm, "MAE": ma}
        if lab != "published":
            c = er.compare(y, cen(base), cen(p), w2, ids_a, n=1000); r.update({"dRMSE": c["dRMSE"], "RMSE 90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "MAE 90%": c["MAE 90%"], "P(MAE better)": c["P(MAE better)"]})
        rows.append(r)
    pd.set_option("display.width", 250); print(f"\nK%: fold 2, n = {len(ids)}, level-free, frozen weights\n" + er.table(rows, 5))
    r2 = []
    for a, b in (("additive swstr_ct", "BOTH-A"), ("additive swstr_ct", "BOTH-B"), ("prior network", "BOTH-A"), ("BOTH-B", "BOTH-A")):
        c = er.compare(y, cen(preds[a]), cen(preds[b]), w2, ids_a, n=1000); r2.append({"comparison": f"{a} -> {b}", "dRMSE": c["dRMSE"], "RMSE 90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "P(MAE better)": c["P(MAE better)"]})
    print("\nPAIRED\n" + er.table(r2, 5))
    tier = pd.cut(meta["PA"], [0, 100, 250, 1e9], labels=["thin (<100 PA)", "mid (100-250)", "thick (>250)"]).to_numpy(); r3 = []
    for tn in ["thin (<100 PA)", "mid (100-250)", "thick (>250)"]:
        m = tier == tn
        if m.sum() >= 8:
            r = {"PA tier": tn, "n": int(m.sum())}; r.update({lab: float(np.sqrt(np.average((y[m] - cen(p)[m]) ** 2, weights=w2[m]))) for lab, p in preds.items()}); r3.append(r)
    print("\nWEIGHTED RMSE BY PA TIER\n" + pd.DataFrame(r3).round(5).to_string(index=False))
    json.dump({"table": rows, "paired": r2, "tiers": r3}, open("output/k_both_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
