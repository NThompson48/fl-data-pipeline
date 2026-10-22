"""
fl_wh_sva_big_sweep.py -- bigger / longer-running PyTorch networks for the wh_sva stuff model (user 2026-10-21: "try a bigger/longer running PyTorch model, since training time is not a bottleneck for this project").

WHAT 'BIGGER / LONGER' MEANS HERE, and what we already know:  capacity has not been the limit (a 2,753-parameter net matched the 11,649-parameter one; deeper XGBoost trees were no better), regularisation helps a single network
but is redundant once 3 seeds are averaged (the ensemble gain was about 3x the regularisation gain).  So big networks are run WITH strong regularisation (dropout, weight decay), a learning-rate that halves whenever the validation
loss stalls (ReduceLROnPlateau) and much longer patience / epoch budgets, and every configuration is a 5-SEED ENSEMBLE so the ensemble-size effect (1, 3, 5 seeds) is measured too.
STAGE 1 (tuning) = 2025 swings only (position players excluded), 5-fold GroupKFold BY PITCHER, out-of-fold per-pitch log-loss of the full model (9 inputs), seeds 1-5, trainer fl_wh_sva_torch (pitcher-grouped validation,
early stopping on validation log-loss).  Configurations (hidden widths; the baseline is the network confirmed on 2026):
  base          128-64-32, dropout 0, lr 1e-3, patience 8                                            (11,649 parameters)
  base_long     the same + plateau LR decay, patience 15, up to 200 epochs
  wide          512-256-128, dropout .3, wd 1e-2, lr 5e-4, batch 2048, plateau, patience 15, up to 200 epochs
  wide_heavy    512-256-128, dropout .5, wd .3, lr 5e-4, batch 2048, plateau, patience 15, up to 200 epochs
  deep_wide     1024-512-256-128, dropout .4, wd 1e-2, lr 3e-4, batch 2048, plateau, patience 15, up to 200 epochs
  resnet        residual 6 x 256 (LayerNorm blocks), dropout .3, wd 1e-2, lr 3e-4, batch 2048, plateau, patience 15, up to 200 epochs
Reported: single-seed log-loss (mean and SD over seeds), 1 / 3 / 5-seed ensemble log-loss, AUC of the 5-seed ensemble, median epochs, mean seconds per fit, parameters; production XGBoost on the same folds.
NOISE UNIT = mean over configurations of SD(single-seed log-loss) / sqrt(5) (the standard error of a 5-seed mean).  PRE-STATED RULE: a configuration beats the baseline only if its 5-seed ensemble log-loss is lower by more than
2 noise units; the best such configuration (ties within one noise unit -> fewer parameters) goes to STAGE 2 (2026 swings, never used for the choice; separate script) with 3 / 10 / 20-seed ensembles vs the baseline.
Output: output/wh_sva_big_sweep.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
SEEDS = (1, 2, 3, 4, 5)
LONG = dict(sched="plateau", patience=15, max_epochs=200)
BIG = dict(batch=2048, wd=1e-2, **LONG)
CONFIGS = {
    "base": {},
    "base_long": dict(**LONG),
    "wide": dict(hidden=(512, 256, 128), dropout=0.3, lr=5e-4, **BIG),
    "wide_heavy": dict(hidden=(512, 256, 128), dropout=0.5, lr=5e-4, **{**BIG, "wd": 0.3}),
    "deep_wide": dict(hidden=(1024, 512, 256, 128), dropout=0.4, lr=3e-4, **BIG),
    "resnet": dict(hidden=(256,) * 6, res=True, dropout=0.3, lr=3e-4, **BIG),
}


def n_params(cfg):
    return sum(p.numel() for p in ft.Net(9, {**ft.DEFAULT, **cfg}).parameters())


def job(name, cfg, seed, k, tr, te, X, y, g):
    torch.set_num_threads(2); t = time.time(); p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg=cfg); return name, seed, k, te, p, ep, time.time() - t


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); pop = raw[raw["is_swing"] & (raw["year"] == 2025)].dropna(subset=fp.FULL_COLS + ["pitcher_id", "is_whiff"]).reset_index(drop=True)
    X = pop[fp.FULL_COLS].astype(float).to_numpy(); y = pop["is_whiff"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); folds = list(GroupKFold(5).split(X, y, g))
    print(f"2025 swings {len(y):,}, {len(CONFIGS)} configurations x {len(SEEDS)} seeds x 5 folds = {len(CONFIGS) * len(SEEDS) * 5} fits; parameters " + ", ".join(f"{n} {n_params(c):,}" for n, c in CONFIGS.items()), flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=8, backend="loky", verbose=0)(delayed(job)(n, c, s, k, tr, te, X, y, g) for n, c in CONFIGS.items() for s in SEEDS for k, (tr, te) in enumerate(folds))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    oof = {(n, s): np.zeros(len(y)) for n in CONFIGS for s in SEEDS}; eps = {n: [] for n in CONFIGS}; secs = {n: [] for n in CONFIGS}
    for n, s, k, te, p, ep, sec in res:
        oof[(n, s)][te] = p; eps[n].append(ep); secs[n].append(sec)
    clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6); rows = []
    for n, c in CONFIGS.items():
        singles = [log_loss(y, clip(oof[(n, s)])) for s in SEEDS]; e3 = np.mean([oof[(n, s)] for s in SEEDS[:3]], axis=0); e5 = np.mean([oof[(n, s)] for s in SEEDS], axis=0)
        rows.append({"config": n, "params": n_params(c), "single_mean": float(np.mean(singles)), "single_sd": float(np.std(singles, ddof=1)), "ens3": log_loss(y, clip(e3)), "ens5": log_loss(y, clip(e5)), "auc_ens5": roc_auc_score(y, e5),
                     "median_epochs": float(np.median(eps[n])), "sec_per_fit": float(np.mean(secs[n]))})
    T = pd.DataFrame(rows).set_index("config"); noise = float((T["single_sd"] / np.sqrt(len(SEEDS))).mean()); base = T.loc["base", "ens5"]
    xp = np.zeros(len(y))
    for tr, te in folds:
        xp[te] = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    T["ens5_vs_base"] = T["ens5"] - base; T["noise_units"] = T["ens5_vs_base"] / noise; T = T.sort_values("ens5")
    best = T["ens5"].iloc[0]; tied = T[T["ens5"] <= best + noise].sort_values("params")
    pd.set_option("display.width", 250)
    print(f"\nreference: production XGBoost {log_loss(y, clip(xp)):.5f} | base rate {log_loss(y, np.full(len(y), y.mean())):.5f} | NOISE UNIT {noise:.5f}\n" + T.round(5).to_string())
    print(f"\nbest 5-seed ensemble: {T.index[0]} ({best:.5f}); within one noise unit: {list(tied.index)}; improvement over base = {base - best:.5f} = {(base - best) / noise:.1f} noise units (rule: > 2 to count); stage 2 candidate: {tied.index[0] if (base - best) / noise > 2 else 'none (base stays)'}")
    json.dump({"noise_unit": noise, "xgb": log_loss(y, clip(xp)), "table": T.reset_index().to_dict("records"), "stage2": (tied.index[0] if (base - best) / noise > 2 else None), "configs": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in CONFIGS.items()}},
              open("output/wh_sva_big_sweep.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
