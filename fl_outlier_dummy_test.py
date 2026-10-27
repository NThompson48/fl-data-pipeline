"""
fl_outlier_dummy_test.py -- does an OUTLIER DUMMY (0/1) improve the production K% model?  (user 2026-10-27)
Production K% = additive regression [naive_K_rate, fl_k_rate_bridge, wh_sva_level (network swstr_ct = Stuff+), spin_z_norm, extension].  Add ONE dummy = 1 if the pitcher's outlier index is in the top p% of the FITTING window (threshold frozen at the fitting window's quantile among the pitchers in the training regression, then applied unchanged to the
scoring window).  Indices from fl_outlier_test (held-out scoring against other pitchers): oi_if / ot_if = isolation-forest mean z / most-unusual-type z; oi / ot = the Mahalanobis versions.
PRIMARY (pre-stated): dummy on oi_if at the top 10%.  Exploratory: oi, ot, ot_if at 10%; oi_if at 5% / 20% / 25%.   Protocol = the project's four pairings (T0->T1b, T0->T2, T1->T2, T2->T0), frozen weights from the production model, level-free, weighted RMSE and MAE, pitcher bootstrap (1,000).
Also reported: the mean K% RESIDUAL of flagged pitchers (actual minus the production prediction, PA-weighted) -- do outliers beat or miss the production K% -- and the coefficient (b, t).   Output: output/outlier_dummy_test.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_outlier_test as ot
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_production_projection as pp
import fl_wh_sva_prior_xgb_robust_test as rb
import mlb_marcel as mm

warnings.filterwarnings("ignore")
ADD = pp.SPEC["K"]; DUMMIES = [("oi_if", .10), ("oi", .10), ("ot", .10), ("ot_if", .10), ("oi_if", .05), ("oi_if", .20), ("oi_if", .25)]


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); rows = []; info = []
    for a, b in rb.PAIRS:
        da, n0k, _, _, _ = ot.frame(raw, a); db_, _, _, _, _ = ot.frame(raw, b)
        dk = da.dropna(subset=["K_rate_actual"] + ADD + ["oi", "ot", "oi_if", "ot_if"]).copy(); wk = mm.reliability_weight(dk["PA_test"], n0k).values; tk = db_.dropna(subset=["K_rate_actual"] + ADD + ["oi", "ot", "oi_if", "ot_if"]).copy()
        specs = {"base": ADD}
        for col, p in DUMMIES:
            thr = float(dk[col].quantile(1 - p)); name = f"d_{col}_{int(p * 100):02d}"; dk[name] = (dk[col] >= thr).astype(float); tk[name] = (tk[col] >= thr).astype(float); specs[f"+ dummy {col} top {int(p * 100)}%"] = ADD + [name]
        P = {}; coef = {}
        for lab, spec in specs.items():
            m = bt.fit_weighted(dk, "K_rate_actual", spec, wk); P[lab] = m.predict(tk).to_numpy(); coef[lab] = {k: (round(float(m.params[k]), 4), round(float(m.tvalues[k]), 2)) for k in spec if k.startswith("d_")}
        y = tk["K_rate_actual"].to_numpy(); n_t = tk["PA_test"].to_numpy(); ids = tk["pitcher_id"].to_numpy(); w = ew.weights_from_base(y, P["base"], n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w); res = y - (P["base"] + np.average(y - P["base"], weights=w))
        for lab, q in P.items():
            rm, ma = er.wmetrics(y, cen(q), w); r = {"pairing": f"{a}->{b}", "n": len(y), "model": lab, "RMSE": rm, "MAE": ma}
            if lab != "base":
                c = er.compare(y, cen(P["base"]), cen(q), w, ids, n=1000); r.update({"dRMSE": c["dRMSE"], "90%": c["RMSE 90%"], "P(RMSE)": c["P(RMSE better)"], "dMAE": c["dMAE"], "P(MAE)": c["P(MAE better)"]})
            rows.append(r)
        flag = {f"{col} top {int(p * 100)}%": tk[f"d_{col}_{int(p * 100):02d}"].to_numpy().astype(bool) for col, p in DUMMIES}
        info.append({"pairing": f"{a}->{b}", "coef (b, t)": coef, "flagged: n, mean K% residual (pts, PA-weighted, level-free)": {k: (int(m.sum()), float(100 * np.average(res[m], weights=n_t[m]))) for k, m in flag.items() if m.sum() > 3}})
        print(f"{a}->{b} done", flush=True)
    T = pd.DataFrame(rows); pd.set_option("display.width", 270)
    print("\nK%: per pairing (change vs the production model; P = bootstrap P(better))\n" + T[T["model"] != "base"].round(5).to_string(index=False))
    g = T.groupby("model").agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean")).sort_values("RMSE"); print("\nMEAN OVER THE FOUR PAIRINGS\n" + g.round(5).to_string())
    print("\nCOEFFICIENTS (b, t) AND FLAGGED-PITCHER RESIDUALS\n" + json.dumps(info, indent=1)[:5000])
    json.dump({"table": rows, "info": info}, open("output/outlier_dummy_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
