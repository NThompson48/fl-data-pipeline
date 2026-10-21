"""
fl_wh_sva_reg_lr_confirm.py -- STAGE 2 of the regularisation / learning-rate study: confirm on 2026 swings (never used to choose the configuration) what stage 1 (fl_wh_sva_reg_lr_sweep.py, 2025 only) selected.
Compared on 2026 swings (5-fold GroupKFold by pitcher, position players excluded), each network as the 3-seed average (the deployable estimator), sva = full - location in every case:
  production   XGBoost (fl_pitcher_features.XGB_PARAMS)
  baseline     torch MLP 128-64-32, ReLU, lr 1e-3, wd 1e-4, dropout 0
  selected     the configuration chosen in stage 1 (output/wh_sva_reg_lr_sweep.json 'selected'; the plain 'A lr 0.001 | reg none' means nothing beat the baseline)
The location-only network is the SAME for baseline and selected (hidden 64-64, baseline regularisation), so any difference comes from the full model.
Reported: per-pitch OOF log-loss (single-seed mean and 3-seed ensemble), AUC, stuff increment (loc - full log-loss), pitcher-level reliability (alternating games, >= 150 pitches per half), correlation with production
at the pitcher level and with K%.
PRE-STATED: go on to the K% harness (fl_wh_sva_nn_test.py protocol) only if the selected configuration beats the baseline network's 2026 ensemble log-loss by at least one stage-1 noise unit AND its reliability is not
lower by more than .01.  Output: output/wh_sva_reg_lr_confirm.json
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_nn_pilot as pilot
import fl_wh_sva_reg_lr_sweep as sw
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
SEEDS = (1, 2, 3)


def job(tag, cfg, seed, k, tr, te, X, y, g):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg=cfg); return tag, seed, k, te, p, ep


def main():
    S = json.load(open("output/wh_sva_reg_lr_sweep.json")); noise = S["noise_unit"]; C = sw.configs(); sel_name = S["selected"]; sel = C[sel_name]
    raw = ppf.exclude(fp.load_fl_pitches_cached()); pop = raw[raw["is_swing"] & (raw["year"] == 2026)].dropna(subset=fp.FULL_COLS + ["pitcher_id", "is_whiff", "date", "game_id", "pa_id", "pitch_no"]).reset_index(drop=True)
    y = pop["is_whiff"].astype(int).to_numpy(); Xf = pop[fp.FULL_COLS].astype(float).to_numpy(); Xl = pop[fp.LOC_COLS].astype(float).to_numpy(); g = pop["pitcher_id"].to_numpy(); folds = list(GroupKFold(5).split(Xf, y, g))
    print(f"2026 swings {len(y):,}, pitchers {len(np.unique(g))}, whiff rate {y.mean():.4f}; selected in stage 1: '{sel_name}' {sel}; noise unit {noise:.5f}", flush=True)
    loc_cfg = dict(hidden=(64, 64)); full_cfgs = {"baseline": {}, "selected": sel}; t0 = time.time()
    jobs = [delayed(job)(f"full_{n}", c, s, k, tr, te, Xf, y, g) for n, c in full_cfgs.items() for s in SEEDS for k, (tr, te) in enumerate(folds)]
    jobs += [delayed(job)("loc", loc_cfg, s, k, tr, te, Xl, y, g) for s in SEEDS for k, (tr, te) in enumerate(folds)]
    res = Parallel(n_jobs=8, backend="loky")(jobs); print(f"networks trained in {(time.time() - t0) / 60:.1f} min", flush=True)
    O = {}; eps = {}
    for tag, s, k, te, p, ep in res:
        O.setdefault((tag, s), np.zeros(len(y)))[te] = p; eps.setdefault(tag, []).append(ep)
    xo_l, xo_f = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in folds:
        xo_l[te] = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(Xl[tr], y[tr]).predict_proba(Xl[te])[:, 1]; xo_f[te] = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(Xf[tr], y[tr]).predict_proba(Xf[te])[:, 1]
    clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6); loc_ens = np.mean([O[("loc", s)] for s in SEEDS], axis=0); out = {}
    P = {"production": (xo_f, xo_l)}
    for n in full_cfgs:
        P[n] = (np.mean([O[(f"full_{n}", s)] for s in SEEDS], axis=0), loc_ens)
    sva = {n: f - l for n, (f, l) in P.items()}; rel = pilot.reliability(raw[raw["year"] == 2026], sva, pop)
    lv = {n: pd.Series(v).groupby(g).mean() for n, v in sva.items()}; term = raw[(raw["year"] == 2026) & raw["is_terminal"]]; kr = term.groupby("pitcher_id").agg(PA=("is_terminal", "size"), K=("k_or_bb", lambda s: (s == "strikeout").mean())); kr = kr[kr["PA"] >= 100]
    for n, (f, l) in P.items():
        single = np.mean([log_loss(y, clip(O[(f"full_{n}", s)])) for s in SEEDS]) if n != "production" else np.nan
        common = lv[n].index.intersection(lv["production"].index); j = kr.join(lv[n].rename("lv"), how="inner")
        out[n] = {"logloss_single_seed_mean": single, "logloss_ensemble": log_loss(y, clip(f)), "auc": roc_auc_score(y, f), "stuff_increment": log_loss(y, clip(l)) - log_loss(y, clip(f)), "median_epochs": float(np.median(eps[f"full_{n}"])) if n != "production" else np.nan,
                  "sva_sd": float(sva[n].std()), **rel[n], "corr_with_production_pitcher": float(np.corrcoef(lv[n].loc[common], lv["production"].loc[common])[0, 1]), "corr_with_K": float(np.corrcoef(j["K"], j["lv"])[0, 1])}
    T = pd.DataFrame(out).T; pd.set_option("display.width", 250); print("\n" + T[["logloss_single_seed_mean", "logloss_ensemble", "auc", "stuff_increment", "median_epochs", "sva_sd", "split_half_r", "full_SB", "n0_pitches", "corr_with_production_pitcher", "corr_with_K"]].round(5).to_string())
    d_ll = out["baseline"]["logloss_ensemble"] - out["selected"]["logloss_ensemble"]; d_rel = out["selected"]["full_SB"] - out["baseline"]["full_SB"]
    go = d_ll >= noise and d_rel >= -0.01
    print(f"\nselected vs baseline on 2026: ensemble log-loss better by {d_ll:+.5f} ({d_ll / noise:+.1f} noise units), reliability {d_rel:+.3f}  ->  {'GO ON to the K% harness' if go else 'STOP: no confirmed gain over the baseline network'}")
    json.dump({"selected": sel_name, "config": sel, "noise_unit": noise, "table": out, "go": bool(go)}, open("output/wh_sva_reg_lr_confirm.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
