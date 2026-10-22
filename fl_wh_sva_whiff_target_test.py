"""
fl_wh_sva_whiff_target_test.py -- how well does each stuff estimator predict a pitcher's FUTURE whiff rate per swing?  (user 2026-10-21: "Let's test it")
The K% test (fl_wh_sva_nn_test.py) found no held-out gain from the network; K% is a noisy, indirect target for a model of whiffs.  Here the target is the quantity the stuff model is about: whiffs per swing.

WINDOWS = the K% harness's (fl_halfseason_backtest dates), so the two tests line up:
  T1 (fit)   features 2025 (full)               -> target 1H2026   (2026-05-07 .. 2026-07-07)
  T2 (test)  features 2025 + 1H2026             -> target 2H2026   (2026-07-08 .. 2026-09-06)
Each estimator's per-pitch sva is cross-fitted (5-fold GroupKFold by pitcher) INSIDE the feature window only, so no target-window swing touches a stuff level.  Pitcher level = mean sva over his feature-window swings.
Estimators (pre-specified; sva = full - location, same 9 inputs): production XGBoost (fp.compute_pooled_wh_sva) | sk_nn3 = scikit-learn MLP 128-64-32, 3 seeds (the earlier pilot / K% test network) |
torch3 = PyTorch MLP 128-64-32 (location 64-64), AdamW, lr 1e-3, wd 1e-4, dropout 0, pitcher-grouped early stopping on log-loss, 3 seeds averaged (the unregularised baseline confirmed on 2026).
MODEL (frozen on T1, applied to T2, nothing refit on T2): whiff_target ~ b0 + b1 * own_shrunk [+ b2 * stuff_level].  own_shrunk = league + (his feature-window whiff rate - league) * n / (n + n0), n0 chosen on T1 by a grid
(least squares weighted by target swings).  WLS on T1 with the standard inverse-variance weights at target swings (fl_eval_weights, fitted on the own-rate-only residuals; frozen).  Pitchers: >= 100 feature-window swings and
>= 60 target-window swings.  Prediction level-free (weighted mean error removed, as in the K% harness).
Scored on T2: weighted RMSE AND MAE (weights frozen from the own-rate-only model's T2 residuals), pitcher bootstrap (1,000): (a) each stuff model vs own-rate-only (does stuff add anything beyond a pitcher's own whiff record?),
(b) the two networks vs production stuff.  Also the swing-weighted correlation of each stuff level with the T2 target (raw and partial of own_shrunk), and each level's T1 coefficient and t.
READING RULE (pre-stated): a network 'measures whiff talent better' if it beats production stuff on BOTH RMSE and MAE in (b) with P(better) >= .90; a network 'adds to own-rate' if it beats own-rate-only in (a) on both with P >= .90.
Output: output/wh_sva_whiff_target_test.json
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_nn_test as nnt
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
KEYS = ["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]
NEEDED = fp.FULL_COLS + ["pitcher_id", "is_whiff", "date", "game_id", "pa_id", "pitch_no"]


def _tjob(tag, cfg, seed, k, tr, te, X, y, g):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg=cfg); return tag, seed, te, p


def torch3(df):
    pop = df[df["is_swing"]].dropna(subset=NEEDED).copy(); y = pop["is_whiff"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy()
    Xf = pop[fp.FULL_COLS].astype(float).to_numpy(); Xl = pop[fp.LOC_COLS].astype(float).to_numpy(); folds = list(GroupKFold(5).split(Xf, y, g))
    jobs = [delayed(_tjob)("full", {}, s, k, tr, te, Xf, y, g) for s in (1, 2, 3) for k, (tr, te) in enumerate(folds)] + [delayed(_tjob)("loc", {"hidden": (64, 64)}, s, k, tr, te, Xl, y, g) for s in (1, 2, 3) for k, (tr, te) in enumerate(folds)]
    O = {}
    for tag, s, te, p in Parallel(n_jobs=8, backend="loky")(jobs):
        O.setdefault((tag, s), np.zeros(len(y)))[te] = p
    sva = np.mean([O[("full", s)] for s in (1, 2, 3)], axis=0) - np.mean([O[("loc", s)] for s in (1, 2, 3)], axis=0)
    return pop[KEYS].assign(sva=sva)


def variants(df):
    comp = nnt.components(df)                                   # production XGBoost ('whiff') and scikit-learn nn / nn3 / blend, memoised
    return {"production": comp["whiff"], "sk_nn3": comp["nn3"], "torch3": torch3(df)}


def pitcher_table(feat, targ, V):
    pop = feat[feat["is_swing"]].dropna(subset=NEEDED); f = pop.groupby("pitcher_id").agg(n_f=("is_whiff", "size"), own=("is_whiff", "mean"))
    tg = targ[targ["is_swing"] & targ["is_whiff"].notna()].groupby("pitcher_id").agg(n_t=("is_whiff", "size"), y=("is_whiff", "mean"))
    T = f.join(tg, how="inner")
    for name, fr in V.items():
        T[name] = fr.groupby("pitcher_id")["sva"].mean()
    T = T[(T["n_f"] >= 100) & (T["n_t"] >= 60)].dropna(); T["lg"] = float(pop["is_whiff"].mean()); return T


def shrunk(T, n0):
    return T["lg"] + (T["own"] - T["lg"]) * T["n_f"] / (T["n_f"] + n0)


def wcorr(a, b, w):
    a, b = np.asarray(a, float), np.asarray(b, float); ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    return float(np.average((a - ma) * (b - mb), weights=w) / np.sqrt(np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w)))


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); t0 = time.time()
    f25 = fp.filter_period(raw, bt.START_2025, bt.END_2025); h1 = fp.filter_period(raw, bt.START_2026, bt.CUTOFF_2026); h2 = fp.filter_period(raw, bt._DAY_AFTER_2026_CUT, bt.END_2026)
    feat1, feat2 = f25, pd.concat([f25, h1], ignore_index=True)
    V1 = variants(feat1); print(f"T1 stuff levels built ({(time.time() - t0) / 60:.1f} min)", flush=True); V2 = variants(feat2); print(f"T2 stuff levels built ({(time.time() - t0) / 60:.1f} min)", flush=True)
    T1 = pitcher_table(feat1, h1, V1); T2 = pitcher_table(feat2, h2, V2); names = list(V1)
    print(f"T1: {len(T1)} pitchers (target 1H2026), T2: {len(T2)} pitchers (target 2H2026); league whiff/swing T1 features {T1['lg'].iloc[0]:.4f}, T2 features {T2['lg'].iloc[0]:.4f}; mean target swings {T1['n_t'].mean():.0f} / {T2['n_t'].mean():.0f}", flush=True)
    grid = [0, 25, 50, 100, 150, 200, 300, 500, 800, 1500]; n0 = min(grid, key=lambda k: float(np.sum(T1["n_t"] * (T1["y"] - shrunk(T1, k)) ** 2))); print(f"own-rate shrinkage n0 chosen on T1: {n0} swings")
    X1 = lambda extra=None: sm.add_constant(pd.DataFrame({"own": shrunk(T1, n0), **({extra: T1[extra]} if extra else {})}))
    base_fit = sm.WLS(T1["y"], X1(), weights=T1["n_t"]).fit(); wT1 = ew.weights_from_base(T1["y"].to_numpy(), base_fit.predict(X1()).to_numpy(), T1["n_t"].to_numpy())[0]
    fits = {"own_only": sm.WLS(T1["y"], X1(), weights=wT1).fit()};
    for n in names:
        fits[n] = sm.WLS(T1["y"], X1(n), weights=wT1).fit()
    print("\nT1 fits (frozen):  b_own  b_stuff  t_stuff")
    for n in ["own_only"] + names:
        p = fits[n].params; print(f"  {n:11s} {p['own']:.3f}  {p.get(n, np.nan):8.3f}  {fits[n].tvalues.get(n, np.nan):6.2f}")
    X2 = lambda extra=None: sm.add_constant(pd.DataFrame({"own": shrunk(T2, n0), **({extra: T2[extra]} if extra else {})}), has_constant="add")
    y = T2["y"].to_numpy(); nT = T2["n_t"].to_numpy(); ids = T2.index.to_numpy(); pred = {"own_only": fits["own_only"].predict(X2()).to_numpy()}
    for n in names:
        pred[n] = fits[n].predict(X2(n)).to_numpy()
    w = ew.weights_from_base(y, pred["own_only"], nT)[0]; cen = lambda p: p + np.average(y - p, weights=w)
    rows = []
    for n, p in pred.items():
        rm, ma = er.wmetrics(y, cen(p), w); r = {"model": ("own whiff rate only" if n == "own_only" else f"own + {n} stuff"), "RMSE": rm, "MAE": ma}
        if n != "own_only":
            a = er.compare(y, cen(pred["own_only"]), cen(p), w, ids, n=1000); r.update({"dRMSE vs own-only": a["dRMSE"], "RMSE 90%": a["RMSE 90%"], "P(RMSE better)": a["P(RMSE better)"], "dMAE": a["dMAE"], "P(MAE better)": a["P(MAE better)"]})
        rows.append(r)
    pd.set_option("display.width", 250); print(f"\nT2 (features 2025 + 1H2026 -> 2H2026 whiff per swing), n = {len(T2)} pitchers, level-free, frozen weights:\n" + er.table(rows, 5))
    rows2 = []
    for n in ["sk_nn3", "torch3"]:
        a = er.compare(y, cen(pred["production"]), cen(pred[n]), w, ids, n=1000)
        rows2.append({"vs production stuff": n, "RMSE": a["RMSE new"], "MAE": a["MAE new"], "dRMSE": a["dRMSE"], "RMSE 90%": a["RMSE 90%"], "P(RMSE better)": a["P(RMSE better)"], "dMAE": a["dMAE"], "MAE 90%": a["MAE 90%"], "P(MAE better)": a["P(MAE better)"]})
    print("\nnetworks vs production stuff (both with own rate):\n" + er.table(rows2, 5))
    own2 = shrunk(T2, n0); rows3 = []
    for n in names:
        res_y = y - np.polyval(np.polyfit(own2, y, 1), own2); res_s = T2[n].to_numpy() - np.polyval(np.polyfit(own2, T2[n].to_numpy(), 1), own2)
        rows3.append({"stuff level": n, "corr with 2H whiff/swing": wcorr(T2[n], y, nT), "partial corr (own rate removed)": wcorr(res_s, res_y, nT), "corr with own rate": wcorr(T2[n], own2, nT)})
    print("\nswing-weighted correlations on T2:\n" + pd.DataFrame(rows3).set_index("stuff level").round(3).to_string())
    for lab, tab in (("T1", T1), ("T2", T2)):
        print(f"\n{lab} correlation between pitcher-level stuff levels:\n" + tab[names].corr().round(3).to_string())
    json.dump({"n0": n0, "T1": len(T1), "T2": len(T2), "fits": {k: {"params": v.params.to_dict(), "t": v.tvalues.to_dict()} for k, v in fits.items()}, "T2_table": rows, "vs_production": rows2, "corr": rows3}, open("output/wh_sva_whiff_target_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
