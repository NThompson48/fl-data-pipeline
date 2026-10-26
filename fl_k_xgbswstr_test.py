"""
fl_k_xgbswstr_test.py -- four-pairing K% test of the XGBoost per-pitch swinging-strike measures against the network swstr_ct  (user 2026-10-26: "I would err on the side of stability unless one is convincingly best for the projection model")
Candidates (additive K% model, stuff term replaced; same protocol as fl_ensemble_test.py: T0->T1b, T1->T2, T0->T2, T2->T0, frozen weights from the published model, level-free, RMSE and MAE, pitcher bootstrap 1,000):
  pub = XGBoost wh_sva (published) | sw = network swstr_ct (production) | sx = XGBoost swstr_ct (same inputs, pitch type one-hot) | ss = network P(swstr | stuff) | sxs = XGBoost P(swstr | stuff).
Reported: levels per pairing, change vs published and vs the network swstr_ct (best-so-far), and the mean over the four pairings.   Output: output/k_xgbswstr_test.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_ensemble_test as et
import fl_eval_report as er
import fl_eval_weights as ew
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_prior_xgb_robust_test as rb

warnings.filterwarnings("ignore")
et.EST = {"pub": "xgb", "sw": "swstr_ct", "ss": "swstr_stuff", "sx": "swstr_ct_xgb", "sxs": "swstr_stuff_xgb"}; et.TARGET = {k: "x" for k in et.EST}; et.bb_specs = lambda: {"pub": et.BBS}


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); rows = []
    for a, b in rb.PAIRS:
        M = et.fit(raw, a); R = et.predictions(raw, M, b); y, n_t, ids, P = R["K"]; w = ew.weights_from_base(y, P["pub"], n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w)
        for k, q in P.items():
            rm, ma = er.wmetrics(y, cen(q), w); r = {"pairing": f"{a}->{b}", "n": len(y), "model": k, "RMSE": rm, "MAE": ma}
            if k != "pub":
                c = er.compare(y, cen(P["pub"]), cen(q), w, ids, n=1000); r.update({"dRMSE vs pub": c["dRMSE"], "P(RMSE) vs pub": c["P(RMSE better)"], "dMAE vs pub": c["dMAE"], "P(MAE) vs pub": c["P(MAE better)"]})
            if k not in ("pub", "sw"):
                c = er.compare(y, cen(P["sw"]), cen(q), w, ids, n=1000); r.update({"dRMSE vs sw": c["dRMSE"], "P(RMSE) vs sw": c["P(RMSE better)"], "dMAE vs sw": c["dMAE"], "P(MAE) vs sw": c["P(MAE better)"]})
            rows.append(r)
        print(f"{a}->{b} done", flush=True)
    T = pd.DataFrame(rows); pd.set_option("display.width", 300); print("\nK%: per pairing\n" + T.round(5).to_string(index=False))
    g = T.groupby("model").agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), dRMSE_pub=("dRMSE vs pub", "mean"), dMAE_pub=("dMAE vs pub", "mean"), dRMSE_sw=("dRMSE vs sw", "mean"), dMAE_sw=("dMAE vs sw", "mean")).sort_values("RMSE")
    print("\nMEAN OVER THE FOUR PAIRINGS\n" + g.round(5).to_string()); json.dump(rows, open("output/k_xgbswstr_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
