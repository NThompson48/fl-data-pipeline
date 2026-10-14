"""
fl_hitter_weight_audit.py -- re-fit the hitter regression constants under the standard evaluation weight (inverse variance 1/(a + b/n), fl_eval_weights.py)
and compare with the placeholder n/(n+150) the first hitter tests used, and with the double-counted variant (reliability x inverse variance).

For each stat: the fitted variance law of next-year error (a = talent/misfit variance, b = sampling noise per observation, n0_eval = b/a),
the regression-to-the-league constant n0 and leave-one-transition-out RMSE under each weighting; then the wOBA model comparison and the
component weights; then how much the 2027 projections themselves would change.

Usage:  python fl_hitter_weight_audit.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev
import fl_hitter_projection as hpj

warnings.filterwarnings("ignore")


def main():
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]
    out = {}
    for mode in ("legacy150", "invvar", "product"):
        bt.WEIGHT_MODE = mode
        out[mode] = hpj.fit_all(p)
    rows = []
    for s in hpj.SPEC:
        r = {"stat": s}
        la = out["invvar"]["law"][s]
        r["law a"] = la["a"]; r["law b"] = la["b"]; r["n0_eval = b/a"] = la["b"] / max(la["a"], 1e-12)
        for mode in ("legacy150", "invvar", "product"):
            r[f"n0 [{mode}]"] = out[mode]["n0"][s]
        rows.append(r)
    pd.set_option("display.width", 250)
    print("REGRESSION CONSTANT n0 (PA, or AB for BABIP) and the fitted variance law, by weighting")
    print(pd.DataFrame(rows).round(5).to_string(index=False))
    print("\nLOTO RMSE of next-year rate, each scored with ITS OWN weighting (compare within a column group, not across)")
    rows = []
    for s in hpj.SPEC:
        r = {"stat": s}
        for mode in ("legacy150", "invvar"):
            m = out[mode]["rmse"][s]
            r[f"mean [{mode}]"] = m["league_mean"]; r[f"last yr [{mode}]"] = m["last_year"]; r[f"model [{mode}]"] = m["model"]
        rows.append(r)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nwOBA MODEL COMPARISON on the same rows (components' fitted weights are re-fit under each weighting)")
    rows = []
    for mode in ("legacy150", "invvar", "product"):
        r = out[mode]["rmse"]
        rows.append({"weighting": mode, "direct": r["woba_direct_same_rows"], "components": r["woba_components"], "50/50 blend": r["woba_blend_same_rows"],
                     "comp weights": ", ".join(f"{k} {v:+.2f}" for k, v in out[mode]["comp_beta"].items()), "rows": out[mode]["n_component_rows"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # how much would the 2027 projections move?
    print("\nEFFECT ON THE 2027 PROJECTIONS (PA >= 100 hitters): shipped constants vs constants re-fit under inverse-variance weights")
    bt.WEIGHT_MODE = "legacy150"; old = hpj.fit_all(p); d_old, _ = hpj.build_projections(p, old)
    bt.WEIGHT_MODE = "invvar"; new = hpj.fit_all(p); d_new, _ = hpj.build_projections(p, new)
    m = d_old[["name_clean", "pa_t", "proj_woba", "proj_k", "proj_bbr", "proj_hrr", "rar_per600"]].merge(d_new[["name_clean", "proj_woba", "proj_k", "proj_bbr", "proj_hrr", "rar_per600"]], on="name_clean", suffixes=("_old", "_new"))
    m = m[m["pa_t"] >= 100]
    rows = []
    for c in ("proj_woba", "proj_k", "proj_bbr", "proj_hrr", "rar_per600"):
        dlt = m[f"{c}_new"] - m[f"{c}_old"]
        rows.append({"projection": c, "n": len(m), "corr old/new": m[f"{c}_old"].corr(m[f"{c}_new"]), "mean change": dlt.mean(), "SD of change": dlt.std(), "max |change|": dlt.abs().max(),
                     "SD across hitters (old)": m[f"{c}_old"].std()})
    print(pd.DataFrame(rows).round(5).to_string(index=False))
    bt.WEIGHT_MODE = "invvar"


if __name__ == "__main__":
    main()
