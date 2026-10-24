"""
fl_wh_sva_prior_xgb_robust_test.py -- the XGBoost-only fading prior for stuff in K%, checked more thoroughly (user 2026-10-23: "also check xgboost only fading prior").
In fl_wh_sva_prior_k_test.py (train = full 2025 -> 1H2026, test = fold 2) the prior with XGBoost stuff beat the constant-weight form (RMSE -.00034 P .94, MAE -.00037 P .95), but (a) the K%-per-unit-of-stuff multiplier hit the top
of its grid (1.5) for both estimators, and (b) it rests on a single held-out fold.  Here: stuff = production XGBoost only (fp.WH_SVA_ESTIMATOR 'xgb'); multiplier grid widened to {.5, .75, 1, 1.25, 1.5, 2, 2.5, 3}; four
(fit transition -> test transition) pairings, each model fit on the first and scored on the second with nothing refit on the test:
  T0 = features 1H2025 -> targets 2H2025                         T1b = features 2H2025 -> targets 1H2026 (the project's experiment-1 test fold)
  T1 = features 2025 (full) -> targets 1H2026 (production training transition)       T2 = features 2025 + 1H2026 -> targets 2H2026 (fold 2)
  pairings: T0 -> T1b | T1 -> T2 (the earlier test, rerun with the wide grid) | T0 -> T2 | T2 -> T0 (fit on the later transition, score the earlier one: the 'reverse' check used elsewhere in the project)
Models: additive (production today) | nostuff (control) | prior (fading stuff prior).  Standard weights from the additive model's residuals on each test set (frozen), level-free, weighted RMSE and MAE, pitcher bootstrap (1,000).
Reading: the prior is 'robust' if additive -> prior has the same sign on RMSE and MAE in all four pairings.  Output: output/wh_sva_prior_xgb_robust_test.json
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
import fl_wh_sva_prior_k_test as pk
import mlb_marcel as mm

warnings.filterwarnings("ignore")
CMULT = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
TR = {"T0": ([(bt.START_2025, bt.CUTOFF_2025)], [(bt._DAY_AFTER_2025_CUT, bt.END_2025)]), "T1b": ([(bt._DAY_AFTER_2025_CUT, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)]),
      "T1": ([(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)]), "T2": ([(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)], [(bt._DAY_AFTER_2026_CUT, bt.END_2026)])}
PAIRS = (("T0", "T1b"), ("T1", "T2"), ("T0", "T2"), ("T2", "T0"))
_cache = {}


def frame(raw, name):
    if name not in _cache:
        feat, lab = TR[name]; df, n0_k, _ = bt.assemble(raw, feat, lab, with_changepoints=False); _cache[name] = (df, n0_k, pk.league_k(raw, feat))
    return _cache[name]


def fit(raw, name):
    df, n0_k, lg = frame(raw, name); d = df.dropna(subset=["K_rate_actual"] + pk.SPEC_ADD).copy(); w = mm.reliability_weight(d["PA_test"], n0_k).values; sm_ = float(d["wh_sva_level"].mean())
    m_add = bt.fit_weighted(d, "K_rate_actual", pk.SPEC_ADD, w); m_no = bt.fit_weighted(d, "K_rate_actual", pk.SPEC_NOSTUFF, w)
    ds = (d["wh_sva_level"] - sm_).to_numpy(); y = d["K_rate_actual"].to_numpy(); c0 = float(np.sum(w * ds * (y - np.average(y, weights=w))) / np.sum(w * ds ** 2)); best = None
    for cm in CMULT:
        c = c0 * cm; Kraw = d["naive_K_rate"] * (d["PA"] + n0_k) - n0_k * lg; n0_s = mm.fit_n0_bridge(Kraw, d["PA"], n0_k, lg, lg + c * ds, d["K_rate_actual"], w)
        db = pk.with_blend(d, n0_k, lg, c, n0_s, sm_); m = bt.fit_weighted(db, "K_rate_actual", pk.SPEC_PRIOR, w); sse = float(np.sum(w * (y - m.predict(db).to_numpy()) ** 2))
        if best is None or sse < best[0]:
            best = (sse, cm, c, n0_s, m)
    return {"n": len(d), "m_add": m_add, "m_no": m_no, "m_pr": best[4], "c": best[2], "cm": best[1], "n0_s": best[3], "stuff_mean": sm_, "coef": float(m_add.params["wh_sva_level"]), "t": float(m_add.tvalues["wh_sva_level"])}


def score(raw, F, test):
    df, n0_k, lg = frame(raw, test); x = df.dropna(subset=pk.SPEC_ADD + ["K_rate_actual"]).copy(); xb = pk.with_blend(x, n0_k, lg, F["c"], F["n0_s"], F["stuff_mean"])
    p = {"additive": F["m_add"].predict(xb).to_numpy(), "nostuff": F["m_no"].predict(xb).to_numpy(), "prior": F["m_pr"].predict(xb).to_numpy()}
    y = xb["K_rate_actual"].to_numpy(); n_t = xb["PA_test"].to_numpy(); ids = xb["pitcher_id"].to_numpy(); w = ew.weights_from_base(y, p["additive"], n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w); out = {"n_test": len(y)}
    for k, q in p.items():
        out[k] = er.wmetrics(y, cen(q), w)
    for k in ("nostuff", "prior"):
        c = er.compare(y, cen(p["additive"]), cen(q := p[k]), w, ids, n=1000); out[f"{k}_vs_additive"] = {"dRMSE": c["dRMSE"], "RMSE 90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "P(MAE better)": c["P(MAE better)"]}
    return out


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); fp.WH_SVA_ESTIMATOR = "xgb"; res = {}; rows = []
    for a, b in PAIRS:
        F = fit(raw, a); S = score(raw, F, b); res[f"{a}->{b}"] = {k: v for k, v in S.items()}
        print(f"\n== fit {a} (n = {F['n']}) -> test {b} (n = {S['n_test']}): stuff coefficient (additive) {F['coef']:.3f} (t {F['t']:.2f}); prior c multiplier {F['cm']} (c = {F['c']:.3f}), n0_s = {F['n0_s']:.0f} pseudo-PA", flush=True)
        for k in ("additive", "nostuff", "prior"):
            print(f"   {k:9s} RMSE {S[k][0]:.5f}  MAE {S[k][1]:.5f}")
        for k in ("nostuff", "prior"):
            v = S[f"{k}_vs_additive"]; print(f"   {k} vs additive: dRMSE {v['dRMSE']:+.5f} {v['RMSE 90%']} P(better) {v['P(RMSE better)']:.2f} | dMAE {v['dMAE']:+.5f} P(better) {v['P(MAE better)']:.2f}")
        rows.append({"pairing": f"{a}->{b}", "n test": S["n_test"], "c mult": F["cm"], "n0_s": F["n0_s"], "add RMSE": S["additive"][0], "prior RMSE": S["prior"][0], "nostuff RMSE": S["nostuff"][0], "prior dRMSE": S["prior_vs_additive"]["dRMSE"], "P(RMSE)": S["prior_vs_additive"]["P(RMSE better)"],
                     "add MAE": S["additive"][1], "prior MAE": S["prior"][1], "nostuff MAE": S["nostuff"][1], "prior dMAE": S["prior_vs_additive"]["dMAE"], "P(MAE)": S["prior_vs_additive"]["P(MAE better)"]})
    pd.set_option("display.width", 260); print("\nSUMMARY (XGBoost stuff; negative d = fading prior better than the constant-weight term)\n" + pd.DataFrame(rows).round(5).to_string(index=False))
    same = all(r["prior dRMSE"] < 0 and r["prior dMAE"] < 0 for r in rows); print(f"\nsame sign (prior better on RMSE and MAE) in all four pairings: {same}")
    json.dump({"pairs": res, "summary": rows}, open("output/wh_sva_prior_xgb_robust_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
