"""
fl_wh_sva_reg_lr_sweep.py -- regularisation and learning-rate sweep for the wh_sva network (user 2026-10-21: "add more regularization to the model and try different learning rates").

STAGE 1 = TUNING, on 2025 swings ONLY (about 107k swings; position players excluded).  5-fold GroupKFold BY PITCHER inside 2025; the network is the full model P(whiff | swing, the 9 shape / location inputs) of
fl_pitcher_features.FULL_COLS; every configuration is trained with 2 seeds (1, 2) in every fold.  Trainer = fl_wh_sva_torch (AdamW, early stopping on validation log-loss with a pitcher-grouped validation split, patience 8,
max 100 epochs, batch 1024).  Baseline = hidden (128, 64, 32), ReLU, lr 1e-3, weight decay 1e-4, dropout 0, no input noise.
  PART A  learning rate {3e-4, 1e-3, 3e-3, 1e-2} x regularisation level {none: dropout 0 / wd 1e-4; light: .1 / 1e-2; medium: .2 / .1; heavy: .3 / .3}                          (16 configurations)
  PART B  one regulariser at a time at lr 1e-3: dropout {.1, .2, .3}; weight decay {1e-2, .1, .3, 1.0}; input noise {.05, .1, .2} SD; smaller networks (64, 32) and (32, 16)    (12 configurations)
Score = out-of-fold per-pitch LOG-LOSS of the full model (lower is better), reported per seed (so the seed-to-seed noise is visible), as the mean of the 2 seeds, and for the 2-seed average prediction (ensemble).
NOISE SCALE = mean |logloss(seed 1) - logloss(seed 2)| / 2 across configurations (the standard error of a 2-seed mean).  PRE-STATED SELECTION RULE: the winner is the configuration with the lowest 2-seed mean log-loss;
configurations within one noise unit of the winner are ties and the tie-break goes to the one with the fewer parameters, then the stronger regularisation; a configuration counts as an improvement over the baseline only
if it is better by more than 2 noise units.  Reference rows: production XGBoost on the same folds, and the base rate.
STAGE 2 (separate script, fl_wh_sva_reg_lr_confirm.py): the selected configuration on 2026 swings, never used for the choice.
Output: output/wh_sva_reg_lr_sweep.json
"""
import json
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
SEEDS = (1, 2)
REG = {"none": dict(dropout=0.0, wd=1e-4), "light": dict(dropout=0.1, wd=1e-2), "medium": dict(dropout=0.2, wd=1e-1), "heavy": dict(dropout=0.3, wd=3e-1)}


def configs():
    c = {}
    for lr, (rn, r) in product((3e-4, 1e-3, 3e-3, 1e-2), REG.items()):
        c[f"A lr {lr:g} | reg {rn}"] = dict(lr=lr, **r)
    for d in (0.1, 0.2, 0.3):
        c[f"B dropout {d}"] = dict(dropout=d)
    for w in (1e-2, 1e-1, 3e-1, 1.0):
        c[f"B weight decay {w:g}"] = dict(wd=w)
    for n in (0.05, 0.1, 0.2):
        c[f"B input noise {n}"] = dict(noise=n)
    for h in ((64, 32), (32, 16)):
        c[f"B smaller net {h}"] = dict(hidden=h)
    return c


def n_params(cfg):
    h = {**ft.DEFAULT, **cfg}["hidden"]; dims = [9, *h, 1]; return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))


def job(name, cfg, seed, k, tr, te, X, y, g):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg=cfg); return name, seed, k, te, p, ep, best


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); pop = raw[raw["is_swing"] & (raw["year"] == 2025)].dropna(subset=fp.FULL_COLS + ["pitcher_id", "is_whiff"]).reset_index(drop=True)
    X = pop[fp.FULL_COLS].astype(float).to_numpy(); y = pop["is_whiff"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); folds = list(GroupKFold(5).split(X, y, g)); C = configs()
    print(f"2025 swings {len(y):,}, pitchers {len(np.unique(g))}, whiff rate {y.mean():.4f}; {len(C)} configurations x {len(SEEDS)} seeds x 5 folds = {len(C) * len(SEEDS) * 5} fits", flush=True); t0 = time.time()
    res = Parallel(n_jobs=8, backend="loky")(delayed(job)(n, c, s, k, tr, te, X, y, g) for n, c in C.items() for s in SEEDS for k, (tr, te) in enumerate(folds))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    oof = {(n, s): np.zeros(len(y)) for n in C for s in SEEDS}; eps = {n: [] for n in C}
    for n, s, k, te, p, ep, best in res:
        oof[(n, s)][te] = p; eps[n].append(ep)
    clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6); rows = []
    for n, c in C.items():
        l1, l2 = (log_loss(y, clip(oof[(n, s)])) for s in SEEDS); ens = 0.5 * (oof[(n, 1)] + oof[(n, 2)])
        rows.append({"config": n, "params": n_params(c), "logloss_mean": 0.5 * (l1 + l2), "logloss_seed1": l1, "logloss_seed2": l2, "logloss_ensemble": log_loss(y, clip(ens)), "auc_ensemble": roc_auc_score(y, ens), "median_epochs": float(np.median(eps[n])), **{k: v for k, v in c.items() if k != "hidden"}, "hidden": str(c.get("hidden", ft.DEFAULT["hidden"]))})
    T = pd.DataFrame(rows).set_index("config"); noise = float((T["logloss_seed1"] - T["logloss_seed2"]).abs().mean() / 2); base = T.loc["A lr 0.001 | reg none", "logloss_mean"]
    xp = np.zeros(len(y))
    for tr, te in folds:
        xp[te] = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    ref = {"production XGBoost": log_loss(y, clip(xp)), "base rate": log_loss(y, np.full(len(y), y.mean()))}
    T["vs_baseline"] = T["logloss_mean"] - base; T["noise_units_vs_baseline"] = T["vs_baseline"] / noise; T = T.sort_values("logloss_mean")
    best = T["logloss_mean"].iloc[0]; tied = T[T["logloss_mean"] <= best + noise].sort_values(["params", "dropout", "wd"], ascending=[True, False, False])
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print(f"\nreference: production XGBoost {ref['production XGBoost']:.5f} | base rate {ref['base rate']:.5f} | baseline network {base:.5f} | NOISE UNIT (SE of a 2-seed mean) {noise:.5f}")
    print(T[["params", "logloss_mean", "logloss_seed1", "logloss_seed2", "logloss_ensemble", "auc_ensemble", "median_epochs", "vs_baseline", "noise_units_vs_baseline"]].round(5).to_string())
    print(f"\nwinner by log-loss: {T.index[0]} ({best:.5f}); configurations within one noise unit of it: {list(tied.index)}; tie-break pick (fewest parameters, then strongest regularisation): {tied.index[0]}")
    print(f"improvement over the baseline network = {base - best:.5f} = {(base - best) / noise:.1f} noise units (rule: > 2 to count)")
    json.dump({"noise_unit": noise, "reference": ref, "baseline": base, "table": T.reset_index().to_dict("records"), "selected": tied.index[0]}, open("output/wh_sva_reg_lr_sweep.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
