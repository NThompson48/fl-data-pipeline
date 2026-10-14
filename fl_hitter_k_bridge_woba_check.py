"""
fl_hitter_k_bridge_woba_check.py -- wOBA model-change report for the K% BRIDGE PRIOR, with the wOBA rebuilt from the UPDATED K% (user 2026-10-13: "when doing the wOBA update,
make sure it uses the updated K% and BB% projections ... when comparing to the previous model").
Same 56 hitter-transitions as fl_hitter_k_bridge_prior_validation.py (bridge-matched hitters, 2023->24 ... 2025->26, next-season PA >= 20).
  OLD wOBA = 0.5 x direct + 0.5 x (lg + b_K (K_old - lg) + b_BB (BB - lg) + b_HR (HR - lg))       K_old = K% regression shrunk to the league mean
  NEW wOBA = the same with K_new (shrunk toward the MiLB bridge)                                    direct half, BB%, HR/PA unchanged
  every regression leave-one-transition-out (n0 refit), production component weights (fl_hitter_model_coefs.json comp_beta).
  2025 -> 2026 rows also through the FULL production path (fl_hitter_pre2026.build_shifted with K_BRIDGE_PRIOR off): K% and BB% include the Whiff% / Swing% terms on both sides,
  wOBA includes the xwOBA nudge and the EV90 term; NEW = OLD + 0.5 x b_K x (bridge term), the bridge term computed exactly as production.
Scoring: next-season park-neutral wOBA, level-free, frozen standard weights, hitter bootstrap; weighted RMSE and MAE.
"""
import contextlib
import io
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew

warnings.filterwarnings("ignore")


def loto(q, s, min_next=20):
    import fl_hitter_baseline_backtest as bt
    import fl_hitter_depth3_test as d3
    import fl_hitter_projection as hp
    saved = bt.MIN_NEXT; bt.MIN_NEXT = min_next; d = d3.frame(q, s); bt.MIN_NEXT = saved; parts = []
    for t in sorted(d["t"].unique()):
        tr, te = d[d["t"] != t], d[d["t"] == t]; parts.append(te.assign(pred=d3.pred(te, d3.fit_n0(tr, hp.SPEC[s][0], 3), hp.SPEC[s][0], 3)))
    return pd.concat(parts)


def main():
    import fl_hitter_k_bridge_prior_validation as kv
    import fl_hitter_model_dev as dev
    beta = json.load(open("output/fl_hitter_model_coefs.json"))["comp_beta"]
    E = kv.build(20); E["k_old"] = kv.preds(E, 0.0); E["k_new"] = kv.preds(E, E["bridge_dev"])
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]
    W = loto(q, "woba")[["name_clean", "t", "pred", "x_next", "n_next", "lg_next"]].rename(columns={"pred": "woba_direct", "x_next": "woba_next", "n_next": "pa_next", "lg_next": "lg_woba"})
    B = loto(q, "bbr")[["name_clean", "t", "pred", "lg_next"]].rename(columns={"pred": "bb", "lg_next": "lg_bb"})
    H = loto(q, "hrr")[["name_clean", "t", "pred", "lg_next"]].rename(columns={"pred": "hr", "lg_next": "lg_hr"})
    J = E[["name_clean", "t", "k_old", "k_new", "lg_next", "hist", "k_n0" if "k_n0" in E else "n0", "nh", "bridge_dev"]].rename(columns={"lg_next": "lg_k"}).merge(W, on=["name_clean", "t"]).merge(B, on=["name_clean", "t"]).merge(H, on=["name_clean", "t"])
    comp = lambda k: J["lg_woba"] + beta["k"] * (k - J["lg_k"]) + beta["bbr"] * (J["bb"] - J["lg_bb"]) + beta["hrr"] * (J["hr"] - J["lg_hr"])
    J["woba_old"] = 0.5 * J["woba_direct"] + 0.5 * comp(J["k_old"]); J["woba_new"] = 0.5 * J["woba_direct"] + 0.5 * comp(J["k_new"])
    y = J["woba_next"].to_numpy(); n = J["pa_next"].to_numpy(); w = ew.weights_from_base(y, J["woba_old"].to_numpy(), n)[0]; c = lambda v: v + np.average(y - v, weights=w)
    rows = []
    r = er.compare(y, c(J["woba_old"].to_numpy()), c(J["woba_new"].to_numpy()), w, J["name_clean"].to_numpy(), n=1000)
    rows.append({"check": "history regressions, all transitions", "n": len(J), **{k: r[k] for k in ("RMSE old", "RMSE new", "dRMSE", "RMSE 90%", "P(RMSE better)", "MAE old", "MAE new", "dMAE", "P(MAE better)")}})
    for t in sorted(J["t"].unique()):
        m = (J["t"] == t).to_numpy(); cc = lambda v: v[m] + np.average(y[m] - v[m], weights=w[m])
        r = er.compare(y[m], cc(J["woba_old"].to_numpy()), cc(J["woba_new"].to_numpy()), w[m], J["name_clean"].to_numpy()[m], n=1000)
        rows.append({"check": f"transition {t} -> {t + 1}", "n": int(m.sum()), **{k: r[k] for k in ("RMSE old", "RMSE new", "dRMSE", "RMSE 90%", "P(RMSE better)", "MAE old", "MAE new", "dMAE", "P(MAE better)")}})
    # full production path, 2025 -> 2026
    import fl_hitter_pre2026 as pre
    with contextlib.redirect_stdout(io.StringIO()):
        d26, c26, _ = pre.build_shifted()
    F = d26.set_index("name_clean"); J5 = J[J["t"] == 2025].copy(); J5 = J5[J5["name_clean"].isin(F.index)]
    term = J5["k_new"] - J5["k_old"]                                               # the bridge term, same formula as production (n0_k x bridge_dev / (history weight + n0_k))
    J5["prod_old"] = J5["name_clean"].map(F["proj_woba"]); J5["prod_new"] = J5["prod_old"] + 0.5 * beta["k"] * term.to_numpy()
    y5 = J5["woba_next"].to_numpy(); w5 = ew.weights_from_base(y5, J5["prod_old"].to_numpy(), J5["pa_next"].to_numpy())[0]; c5 = lambda v: v + np.average(y5 - v, weights=w5)
    r = er.compare(y5, c5(J5["prod_old"].to_numpy()), c5(J5["prod_new"].to_numpy()), w5, J5["name_clean"].to_numpy(), n=1000)
    rows.append({"check": "FULL production path 2025 -> 2026 (Whiff% / Swing% / xwOBA / EV90 terms in both)", "n": len(J5), **{k: r[k] for k in ("RMSE old", "RMSE new", "dRMSE", "RMSE 90%", "P(RMSE better)", "MAE old", "MAE new", "dMAE", "P(MAE better)")}})
    pd.set_option("display.width", 250); R = pd.DataFrame(rows); print("wOBA with the components rebuilt from the UPDATED K% (bridge prior) vs the previous model\n" + R.round(5).to_string(index=False))
    d = (J["woba_new"] - J["woba_old"]).abs(); print(f"wOBA change on these rows: mean |{d.mean():.4f}|, max {d.max():.4f}")
    R.to_csv("output/hitter_k_bridge_woba_check.csv", index=False)


if __name__ == "__main__":
    main()
