"""
fl_wh_sva_prior_k_test.py -- stuff as a FADING PRIOR in the pitcher K% model, the neural-network stuff vs XGBoost stuff  (user 2026-10-23: "let's add it as a prior and report the updated results")

Production K% = a WLS regression on [naive_K_rate (own K%, shrunk to the league), fl_k_rate_bridge, wh_sva_level, spin_z_norm, extension]: the stuff term has the SAME weight for a pitcher with 40 PA and one with 800.
PRIOR FORM, following the project's own BB% treatment of the bridge (fl_production_projection.BB_SPEC_BLENDED; mm.blended_rate / mm.fit_n0_bridge): the stuff-implied K% becomes a block of pseudo-plate-appearances,
    P_stuff = lg + c * (wh_sva_level - mean)              (c = K% per unit of stuff, calibrated on the training transition: the OLS slope times a grid multiplier {.5, .75, 1, 1.25, 1.5} chosen by the training-fit SSE)
    blended_K = (K_count + n0 * lg + n0_s * P_stuff) / (PA + n0 + n0_s)      (n0 = the production league n0; n0_s = the stuff prior's pseudo-PA, fit by nonlinear least squares on the training transition, mm.fit_n0_bridge)
    K% model = WLS on [blended_K, fl_k_rate_bridge, spin_z_norm, extension]   (wh_sva_level leaves the regression; its weight n0_s / (PA + n0 + n0_s) fades with PA)
n0_s = 0 would recover the league-shrunk naive rate; a pitcher with no PA gets the pure prior.  Pitchers with no FL pitches at all have no stuff number (stuff needs their own pitches): they stay on the bridge path.
PROTOCOL = the standard K% harness (fl_wh_sva_variants_test.py): fit on the training transition full 2025 -> actual 1H2026; fold 2 = features 2025 + 1H2026 -> targets actual 2H2026; position players excluded; level-free;
standard inverse-variance weights from the PRODUCTION (additive, XGBoost) variant's residuals, frozen; weighted RMSE AND MAE; pitcher bootstrap (1,000).
Models: additive_xgb (= production today) | additive_nn | nostuff (control: stuff removed) | prior_xgb | prior_nn   (nn = the 3-seed PyTorch network through fp.compute_pooled_wh_sva(estimator='torch3'))
Also by feature-window PA tier (thin < 100, mid 100-250, thick > 250).
ADOPTION (pre-stated): the prior form replaces the additive term in production only if it beats the additive form of the SAME estimator on both RMSE and MAE with P(better) >= .90 overall, or does so clearly in the thin tier
without being worse overall; otherwise the additive form stays and the estimator choice is reported on its own.  Output: output/wh_sva_prior_k_test.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_production_projection as pp
import mlb_marcel as mm

warnings.filterwarnings("ignore")
SPEC_ADD = pp.SPEC["K"]; SPEC_NOSTUFF = [f for f in SPEC_ADD if f != "wh_sva_level"]; SPEC_PRIOR = ["blended_K_stuff"] + [f for f in SPEC_ADD if f not in ("naive_K_rate", "wh_sva_level")]
CMULT = (0.5, 0.75, 1.0, 1.25, 1.5)


def league_k(raw, windows):
    t = pd.concat([fp.filter_period(raw, a, b) for a, b in windows], ignore_index=True); t = t[t["is_terminal"]]; return float((t["k_or_bb"] == "strikeout").mean())


def with_blend(df, n0_k, lg, c, n0_s, stuff_mean):
    d = df.copy(); d["K_raw"] = d["naive_K_rate"] * (d["PA"] + n0_k) - n0_k * lg; d["P_stuff"] = lg + c * (d["wh_sva_level"] - stuff_mean)
    d["blended_K_stuff"] = mm.blended_rate(d["K_raw"], d["PA"], n0_k, lg, n0_s, d["P_stuff"]); return d


def build(raw, estimator):
    """training transition + fold 2 frames for one stuff estimator, with the fitted additive / no-stuff / prior K% models."""
    fp.WH_SVA_ESTIMATOR = estimator
    tr, n0_k, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    f2, n0_k2, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)], [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)
    lg_tr = league_k(raw, [(bt.START_2025, bt.END_2025)]); lg_f2 = league_k(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)])
    d = tr.dropna(subset=["K_rate_actual"] + SPEC_ADD).copy(); w = mm.reliability_weight(d["PA_test"], n0_k).values; stuff_mean = float(d["wh_sva_level"].mean())
    m_add = bt.fit_weighted(d, "K_rate_actual", SPEC_ADD, w); m_no = bt.fit_weighted(d, "K_rate_actual", SPEC_NOSTUFF, w)
    ds = (d["wh_sva_level"] - stuff_mean).to_numpy(); y = d["K_rate_actual"].to_numpy(); c0 = float(np.sum(w * ds * (y - np.average(y, weights=w))) / np.sum(w * ds ** 2)); best = None
    for cm in CMULT:
        c = c0 * cm; Kraw = d["naive_K_rate"] * (d["PA"] + n0_k) - n0_k * lg_tr; P = lg_tr + c * ds
        n0_s = mm.fit_n0_bridge(Kraw, d["PA"], n0_k, lg_tr, P, d["K_rate_actual"], w); db = with_blend(d, n0_k, lg_tr, c, n0_s, stuff_mean); m = bt.fit_weighted(db, "K_rate_actual", SPEC_PRIOR, w)
        sse = float(np.sum(w * (y - m.predict(db).to_numpy()) ** 2))
        if best is None or sse < best[0]:
            best = (sse, cm, c, n0_s, m)
    _, cm, c, n0_s, m_pr = best
    x = f2.dropna(subset=SPEC_ADD + ["K_rate_actual"]).copy(); xb = with_blend(x, n0_k2, lg_f2, c, n0_s, stuff_mean)
    info = {"n_train": len(d), "n_fold2": len(x), "c0 (OLS K% per unit stuff)": c0, "c multiplier chosen": cm, "n0_s (pseudo-PA of the stuff prior)": n0_s, "n0_k (league)": n0_k, "stuff coefficient in the additive model": float(m_add.params["wh_sva_level"]),
            "stuff t in the additive model": float(m_add.tvalues["wh_sva_level"]), "prior-model params": {k: float(v) for k, v in m_pr.params.items()}}
    pred = pd.DataFrame({"additive": m_add.predict(xb).to_numpy(), "nostuff": m_no.predict(xb).to_numpy(), "prior": m_pr.predict(xb).to_numpy()}, index=xb["pitcher_id"].to_numpy())
    meta = xb.set_index("pitcher_id")[["K_rate_actual", "PA_test", "PA"]]; return pred, meta, info


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); P, M, I = {}, {}, {}
    for est in ("xgb", "torch3"):
        P[est], M[est], I[est] = build(raw, est); print(f"\n[{est}] " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in I[est].items() if k != "prior-model params"}), flush=True)
        print(f"[{est}] prior-model parameters: {I[est]['prior-model params']}", flush=True)
    ids = sorted(set(P["xgb"].index) & set(P["torch3"].index)); meta = M["xgb"].loc[ids]; y = meta["K_rate_actual"].to_numpy(); n_t = meta["PA_test"].to_numpy()
    preds = {"additive_xgb (production today)": P["xgb"].loc[ids, "additive"].to_numpy(), "nostuff (control)": P["xgb"].loc[ids, "nostuff"].to_numpy(), "prior_xgb": P["xgb"].loc[ids, "prior"].to_numpy(),
             "additive_nn": P["torch3"].loc[ids, "additive"].to_numpy(), "prior_nn": P["torch3"].loc[ids, "prior"].to_numpy()}
    base = preds["additive_xgb (production today)"]; w = ew.weights_from_base(y, base, n_t)[0]; cen = lambda p: p + np.average(y - p, weights=w); ids_a = np.array(ids)
    rows = []
    for lab, p in preds.items():
        rm, ma = er.wmetrics(y, cen(p), w); r = {"model": lab, "RMSE": rm, "MAE": ma}
        if lab != "additive_xgb (production today)":
            c = er.compare(y, cen(base), cen(p), w, ids_a, n=1000); r.update({"dRMSE vs production": c["dRMSE"], "RMSE 90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "P(MAE better)": c["P(MAE better)"]})
        rows.append(r)
    pd.set_option("display.width", 260); print(f"\nK%: fold 2 (2H2026 targets), n = {len(ids)} pitchers, level-free, frozen weights\n" + er.table(rows, 5))
    rows2 = []
    for a, b in (("additive_xgb (production today)", "prior_xgb"), ("additive_nn", "prior_nn"), ("additive_xgb (production today)", "additive_nn"), ("prior_xgb", "prior_nn")):
        c = er.compare(y, cen(preds[a]), cen(preds[b]), w, ids_a, n=1000); rows2.append({"comparison": f"{a.split(' ')[0]} -> {b}", "dRMSE": c["dRMSE"], "RMSE 90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "MAE 90%": c["MAE 90%"], "P(MAE better)": c["P(MAE better)"]})
    print("\nPAIRED COMPARISONS\n" + er.table(rows2, 5))
    tier = pd.cut(meta["PA"], [0, 100, 250, 1e9], labels=["thin (<100 PA)", "mid (100-250)", "thick (>250)"]).to_numpy(); rows3 = []
    for t in ["thin (<100 PA)", "mid (100-250)", "thick (>250)"]:
        m = tier == t
        if m.sum() < 8:
            continue
        r = {"PA tier": t, "n": int(m.sum())}
        for lab, p in preds.items():
            r[lab.split(" ")[0] + ("" if "(" not in lab else "")] = float(np.sqrt(np.average((y[m] - cen(p)[m]) ** 2, weights=w[m])))
        rows3.append(r)
    print("\nWEIGHTED RMSE BY FEATURE-WINDOW PA TIER\n" + pd.DataFrame(rows3).round(5).to_string(index=False))
    json.dump({"info": I, "table": rows, "paired": rows2, "tiers": rows3}, open("output/wh_sva_prior_k_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
