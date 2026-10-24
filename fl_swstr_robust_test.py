"""
fl_swstr_robust_test.py -- the four-pairing robustness check for the per-pitch swinging-strike stuff (swstr_ct)  (user 2026-10-24: "run the other tests yes")
Same transitions and pairings as fl_wh_sva_prior_xgb_robust_test.py:  T0 = features 1H2025 -> targets 2H2025; T1b = 2H2025 -> 1H2026; T1 = 2025 -> 1H2026; T2 = 2025 + 1H2026 -> 2H2026;  pairings T0->T1b | T1->T2 | T0->T2 | T2->T0 (fit on the first, score the second, nothing refit).
Models (each fit on the fit transition with that transition's own stuff estimates): published (additive XGBoost) | additive network (swing-based) | ADDITIVE swstr_ct | prior network (fading prior, NLS (c, n0_s)).
Standard inverse-variance weights from the PUBLISHED model's residuals on each test set (frozen), level-free, weighted RMSE and MAE, pitcher bootstrap (1,000).  Test population = pitchers with all four predictions.
Reading (pre-stated): swstr_ct additive is 'robust' if it beats published on RMSE and MAE in all four pairings; reported with the interval and P(better) for each, and against the network models.  Output: output/swstr_robust_test.json
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
import fl_wh_sva_prior_k_test as pk
import fl_wh_sva_prior_xgb_robust_test as rb
import mlb_marcel as mm

warnings.filterwarnings("ignore")
ADD = pp.SPEC["K"]; EST = ("xgb", "torch3", "swstr_ct"); _c = {}


def frame(raw, est, name):
    if (est, name) not in _c:
        fp.WH_SVA_ESTIMATOR = est; feat, lab = rb.TR[name]; df, n0_k, _ = bt.assemble(raw, feat, lab, with_changepoints=False); _c[(est, name)] = (df, n0_k, pk.league_k(raw, feat))
    return _c[(est, name)]


def fit(raw, name):
    M = {}
    for est in EST:
        df, n0_k, lg = frame(raw, est, name); d = df.dropna(subset=["K_rate_actual"] + ADD).copy(); w = mm.reliability_weight(d["PA_test"], n0_k).values; M[est] = {"add": bt.fit_weighted(d, "K_rate_actual", ADD, w)}
        if est == "torch3":
            M[est]["prior"] = kp.fit(d, n0_k, lg, w); print(f"   [{name}] network prior: c={M[est]['prior'].c:.3f} (x{M[est]['prior'].c / M[est]['prior'].c0:.2f}), n0_s={M[est]['prior'].n0_s:.1f}", flush=True)
        print(f"   [{name}] {est}: stuff coefficient {M[est]['add'].params['wh_sva_level']:.3f} (t {M[est]['add'].tvalues['wh_sva_level']:.2f}), n={len(d)}", flush=True)
    return M


def score(raw, M, test):
    P = {}; meta = None
    for est in EST:
        df, n0_k, lg = frame(raw, est, test); x = df.dropna(subset=ADD + ["K_rate_actual"]).copy(); x["lg_k_frame"] = lg; x["n0_k_frame"] = n0_k
        P[(est, "add")] = pd.Series(M[est]["add"].predict(x).to_numpy(), index=x["pitcher_id"].to_numpy())
        if "prior" in M[est]:
            P[(est, "prior")] = pd.Series(M[est]["prior"].predict(x).to_numpy(), index=x["pitcher_id"].to_numpy())
        if est == "xgb":
            meta = x.set_index("pitcher_id")[["K_rate_actual", "PA_test", "PA"]]
    ids = sorted(set.intersection(*(set(v.index) for v in P.values())) & set(meta.index)); meta = meta.loc[ids]; y = meta["K_rate_actual"].to_numpy(); n_t = meta["PA_test"].to_numpy(); ids_a = np.array(ids)
    preds = {"published": P[("xgb", "add")].loc[ids].to_numpy(), "additive network": P[("torch3", "add")].loc[ids].to_numpy(), "ADDITIVE swstr_ct": P[("swstr_ct", "add")].loc[ids].to_numpy(), "prior network": P[("torch3", "prior")].loc[ids].to_numpy()}
    w = ew.weights_from_base(y, preds["published"], n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w); out = {"n_test": len(ids)}
    for k, q in preds.items():
        out[k] = {"RMSE": er.wmetrics(y, cen(q), w)[0], "MAE": er.wmetrics(y, cen(q), w)[1]}
        if k != "published":
            c = er.compare(y, cen(preds["published"]), cen(q), w, ids_a, n=1000); out[k].update({"dRMSE": c["dRMSE"], "RMSE 90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "MAE 90%": c["MAE 90%"], "P(MAE better)": c["P(MAE better)"]})
    for k in ("additive network", "prior network"):
        c = er.compare(y, cen(preds[k]), cen(preds["ADDITIVE swstr_ct"]), w, ids_a, n=1000); out[f"swstr_ct vs {k}"] = {"dRMSE": c["dRMSE"], "RMSE 90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "P(MAE better)": c["P(MAE better)"]}
    return out


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); res = {}; rows = []
    for a, b in rb.PAIRS:
        M = fit(raw, a); S = score(raw, M, b); res[f"{a}->{b}"] = S; print(f"\n== fit {a} -> test {b} (n = {S['n_test']})", flush=True)
        for k in ("published", "additive network", "ADDITIVE swstr_ct", "prior network"):
            v = S[k]; tail = "" if k == "published" else f"  dRMSE {v['dRMSE']:+.5f} {v['RMSE 90%']} P {v['P(RMSE better)']:.2f} | dMAE {v['dMAE']:+.5f} {v['MAE 90%']} P {v['P(MAE better)']:.2f}"
            print(f"   {k:19s} RMSE {v['RMSE']:.5f}  MAE {v['MAE']:.5f}{tail}", flush=True)
            rows.append({"pairing": f"{a}->{b}", "model": k, "n": S["n_test"], "RMSE": v["RMSE"], "MAE": v["MAE"], "dRMSE": v.get("dRMSE"), "P(RMSE)": v.get("P(RMSE better)"), "dMAE": v.get("dMAE"), "P(MAE)": v.get("P(MAE better)")})
        for k in ("swstr_ct vs additive network", "swstr_ct vs prior network"):
            v = S[k]; print(f"   {k}: dRMSE {v['dRMSE']:+.5f} {v['RMSE 90%']} P {v['P(RMSE better)']:.2f} | dMAE {v['dMAE']:+.5f} P {v['P(MAE better)']:.2f}", flush=True)
    pd.set_option("display.width", 260); T = pd.DataFrame(rows); print("\nSUMMARY\n" + T.round(5).to_string(index=False))
    s = T[T["model"] == "ADDITIVE swstr_ct"]; print(f"\nADDITIVE swstr_ct beats published on RMSE and MAE in all four pairings: {bool(((s['dRMSE'] < 0) & (s['dMAE'] < 0)).all())}")
    json.dump({"pairs": res, "summary": rows}, open("output/swstr_robust_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
