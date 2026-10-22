"""
fl_wh_sva_firstN_test.py -- is stuff an EARLIER signal of whiff talent than the pitcher's own whiff rate?  (user 2026-10-21: "Stuff is most useful as an earlier signal than Whiff%, not necessarily a better one over a
longer period"; "Sure" to testing it by sample size.)

DESIGN (fixed before running).  Per pitcher-season (2025 and 2026 separately; position players excluded; swings with all 9 shape / location inputs): order the swings chronologically; the FIRST N swings are what is
known 'early'; the target is the whiff rate on his REMAINING swings of that season (pitchers with >= N + 100 swings).  N = 20, 40, 80, 150, 250, 400 swings (about 45, 90, 180, 340, 560, 900 pitches).
Stuff: each estimator's per-pitch sva is cross-fitted by pitcher over the pooled 2025-26 swings (5-fold GroupKFold, so no pitcher's own pitches inform his sva); stuff_N = mean sva over his first N swings.
Estimators: production XGBoost | sk_nn3 (scikit-learn MLP, 3 seeds) | torch3 (PyTorch MLP 128-64-32, 3 seeds, unregularised baseline).
Models, all FIT ON ONE SEASON AND SCORED ON THE OTHER (2025 -> 2026 and 2026 -> 2025), nothing refit on the test season:
  own      league + (own whiff rate over the first N - league) * N / (N + n0), n0 chosen by a grid on the training season          ('Whiff% only')
  stuff    a + c * stuff_N                                                                                                          ('stuff only')
  add      b0 + b1 * own_shrunk + b2 * stuff_N                                                                                    (the constant-weight add-on the K% model uses)
  prior    prior = a + c * stuff_N ;  prediction = prior + (own - prior) * N / (N + n0')                                           (stuff as an INFORMATIVE PRIOR whose weight fades with N)
WLS with the standard inverse-variance weights at the target-swing count (fl_eval_weights, fitted on the own-only residuals and frozen).  Scored level-free: weighted RMSE AND MAE with pitcher bootstrap (1,000)
against the own-only model.  Pre-stated reading: stuff 'is an earlier signal' at N if stuff-only beats own-only on both metrics with P(better) >= .90; the informative prior 'adds value' at N if it beats own-only on both with
P >= .90; a network beats production at N if it beats production stuff in the same model on both with P >= .90.  Also the swing-weighted correlations of own_N and stuff_N with the target by N.
Output: output/wh_sva_firstN_test.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import fl_eval_report as er
import fl_eval_weights as ew
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_nn_test as nnt
import fl_wh_sva_whiff_target_test as wt

warnings.filterwarnings("ignore")
NS = (20, 40, 80, 150, 250, 400); GRID = [0, 25, 50, 100, 150, 200, 300, 500, 800, 1500]; MIN_REST = 100
EST = ("production", "sk_nn3", "torch3")


def pitcher_rows(pop, season, N):
    s = pop[pop["year"] == season].copy(); s["k"] = s.groupby("pitcher_id").cumcount(); tot = s.groupby("pitcher_id").size(); ok = tot[tot >= N + MIN_REST].index; s = s[s["pitcher_id"].isin(ok)]
    first, rest = s[s["k"] < N], s[s["k"] >= N]; T = first.groupby("pitcher_id").agg(own=("is_whiff", "mean"), **{e: (e, "mean") for e in EST}).join(rest.groupby("pitcher_id").agg(y=("is_whiff", "mean"), n_t=("is_whiff", "size")))
    T["N"] = N; T["lg"] = float(first["is_whiff"].mean()); return T


def shrunk(T, n0):
    return T["lg"] + (T["own"] - T["lg"]) * T["N"] / (T["N"] + n0)


def fit_models(T):
    """parameters on the training season: n0 (own), regression of stuff-only / additive, informative-prior (a, c, n0')."""
    wfit0 = T["n_t"].to_numpy(float); y = T["y"].to_numpy(); P = {}
    n0 = min(GRID, key=lambda k: float(np.sum(wfit0 * (y - shrunk(T, k)) ** 2))); own = shrunk(T, n0); w = ew.weights_from_base(y, own.to_numpy(), T["n_t"].to_numpy())[0]; P["n0"] = n0; P["w"] = w
    for e in EST:
        st = T[e].to_numpy(); P[("stuff", e)] = sm.WLS(y, sm.add_constant(st), weights=w).fit().params
        P[("add", e)] = sm.WLS(y, sm.add_constant(np.column_stack([own.to_numpy(), st])), weights=w).fit().params
        best = None
        for k in GRID:
            wt_ = T["N"].to_numpy() / (T["N"].to_numpy() + k); Z = np.column_stack([(1 - wt_), (1 - wt_) * st]); r = sm.WLS(y - wt_ * T["own"].to_numpy(), Z, weights=w).fit(); sse = float(np.sum(w * (y - wt_ * T["own"].to_numpy() - Z @ r.params) ** 2))
            if best is None or sse < best[0]:
                best = (sse, k, r.params)
        P[("prior", e)] = (best[1], best[2])
    return P


def predict(T, P):
    out = {"own": shrunk(T, P["n0"]).to_numpy()}; own = out["own"]
    for e in EST:
        st = T[e].to_numpy(); a = P[("stuff", e)]; out[("stuff", e)] = a[0] + a[1] * st; b = P[("add", e)]; out[("add", e)] = b[0] + b[1] * own + b[2] * st
        k, ac = P[("prior", e)]; wt_ = T["N"].to_numpy() / (T["N"].to_numpy() + k); out[("prior", e)] = wt_ * T["own"].to_numpy() + (1 - wt_) * (ac[0] + ac[1] * st)
    return out


def wcorr(a, b, w):
    a, b = np.asarray(a, float), np.asarray(b, float); ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    return float(np.average((a - ma) * (b - mb), weights=w) / np.sqrt(np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w)))


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); t0 = time.time()
    comp = nnt.components(raw); V = {"production": comp["whiff"], "sk_nn3": comp["nn3"], "torch3": wt.torch3(raw)}; print(f"pooled per-pitch stuff built for 3 estimators ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pop = raw[raw["is_swing"]].dropna(subset=wt.NEEDED).copy()[wt.KEYS + ["year", "is_whiff"]]; pop["is_whiff"] = pop["is_whiff"].astype(float)
    for e, fr in V.items():
        pop = pop.merge(fr[wt.KEYS + ["sva"]].rename(columns={"sva": e}), on=wt.KEYS, how="left")
    pop = pop.dropna(subset=list(EST)).sort_values(["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]).reset_index(drop=True); print(f"{len(pop):,} swings, 2025 {int((pop.year == 2025).sum()):,} / 2026 {int((pop.year == 2026).sum()):,}", flush=True)
    allrows, corr = [], []
    for N in NS:
        for train, test in ((2025, 2026), (2026, 2025)):
            Ttr, Tte = pitcher_rows(pop, train, N), pitcher_rows(pop, test, N)
            if len(Ttr) < 30 or len(Tte) < 30:
                continue
            P = fit_models(Ttr); pred = predict(Tte, P); y = Tte["y"].to_numpy(); n_t = Tte["n_t"].to_numpy(); ids = Tte.index.to_numpy(); w = ew.weights_from_base(y, pred["own"], n_t)[0]; cen = lambda p: p + np.average(y - p, weights=w)
            row = {"N swings": N, "fit -> test": f"{train}->{test}", "n train": len(Ttr), "n test": len(Tte), "n0 own": P["n0"]}
            rm0, ma0 = er.wmetrics(y, cen(pred["own"]), w); row["own RMSE"], row["own MAE"] = rm0, ma0
            for key, p in pred.items():
                if key == "own":
                    continue
                kind, e = key; r = er.compare(y, cen(pred["own"]), cen(p), w, ids, n=1000)
                allrows.append({**row, "model": kind, "estimator": e, "RMSE": r["RMSE new"], "MAE": r["MAE new"], "dRMSE vs own": r["dRMSE"], "P(RMSE better)": r["P(RMSE better)"], "dMAE vs own": r["dMAE"], "P(MAE better)": r["P(MAE better)"],
                                **({"prior n0'": P[("prior", e)][0]} if kind == "prior" else {})})
            for e in EST:
                corr.append({"N": N, "fit -> test": f"{train}->{test}", "estimator": e, "corr own_N with target": wcorr(Tte["own"], y, n_t), "corr stuff_N with target": wcorr(Tte[e], y, n_t), "corr stuff_N with own_N": wcorr(Tte[e], Tte["own"], n_t)})
    R = pd.DataFrame(allrows); pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    print("\nSTUFF ONLY vs Whiff% ONLY (d < 0 = stuff predicts the rest of the season better):")
    print(R[R["model"] == "stuff"][["N swings", "fit -> test", "estimator", "n test", "own RMSE", "RMSE", "dRMSE vs own", "P(RMSE better)", "own MAE", "MAE", "dMAE vs own", "P(MAE better)"]].round(5).to_string(index=False))
    print("\nSTUFF AS AN INFORMATIVE PRIOR vs Whiff% ONLY:")
    print(R[R["model"] == "prior"][["N swings", "fit -> test", "estimator", "n test", "prior n0'", "RMSE", "dRMSE vs own", "P(RMSE better)", "dMAE vs own", "P(MAE better)"]].round(5).to_string(index=False))
    print("\nCONSTANT-WEIGHT ADD-ON vs Whiff% ONLY:")
    print(R[R["model"] == "add"][["N swings", "fit -> test", "estimator", "n test", "RMSE", "dRMSE vs own", "P(RMSE better)", "dMAE vs own", "P(MAE better)"]].round(5).to_string(index=False))
    C = pd.DataFrame(corr); print("\nSWING-WEIGHTED CORRELATION WITH THE REST-OF-SEASON WHIFF RATE:\n" + C.round(3).to_string(index=False))
    json.dump({"results": allrows, "corr": corr}, open("output/wh_sva_firstN_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
