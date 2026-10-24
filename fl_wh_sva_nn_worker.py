"""
fl_wh_sva_nn_worker.py -- the neural-network wh_sva estimator, run as a SEPARATE PROCESS:  python fl_wh_sva_nn_worker.py <swings.parquet> <sva.parquet>
(called by fl_pitcher_features.compute_pooled_wh_sva(estimator='torch3'); a separate process because on this Windows machine torch only loads if it is imported BEFORE xgboost / scikit-learn, which the pipeline
processes cannot guarantee.)
Same definition as production: per swing, sva = P(whiff | swing, shape + location)_oof - P(whiff | swing, location)_oof, cross-fitted by pitcher (5-fold GroupKFold by pitcher_id) so no pitcher's own pitches inform his
prediction.  Both probabilities come from PyTorch MLPs averaged over 3 seeds (the configuration validated in the pilot, the 2026 confirmation and the whiff-rate tests):
  full model      9 inputs (fp.FULL_COLS) -> 128 -> 64 -> 32 -> 1, ReLU, AdamW lr 1e-3 / wd 1e-4, batch 1024, early stopping on the validation log-loss of a pitcher-grouped 10% split (patience 8)    (11,649 parameters)
  location model  2 inputs (fp.LOC_COLS)  -> 64 -> 64 -> 1, same training                                                                                                                         (4,417 parameters)
Input parquet: the swings with all inputs (FULL_COLS + pitcher_id, is_whiff, date, game_id, pa_id, pitch_no).  Output parquet: pitcher_id, date, game_id, pa_id, pitch_no, sva (same row order).
"""
import torch  # MUST come first (see above)
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

import fl_pitcher_features as fp
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
SEEDS = (1, 2, 3)
KEYS = ["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]


def _job(tag, cfg, seed, tr, te, X, y, g):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg=cfg); return tag, seed, te, p


def compute(pop):
    y = pop["is_whiff"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); Xf = pop[fp.FULL_COLS].astype(float).to_numpy(); Xl = pop[fp.LOC_COLS].astype(float).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(Xf, y, g)); jobs = []
    for tag, X, cfg in (("full", Xf, {}), ("loc", Xl, {"hidden": (64, 64)})):
        jobs += [delayed(_job)(tag, cfg, s, tr, te, X, y, g) for s in SEEDS for tr, te in folds]
    O = {}
    for tag, s, te, p in Parallel(n_jobs=8, backend="loky")(jobs):
        O.setdefault((tag, s), np.zeros(len(y)))[te] = p
    return np.mean([O[("full", s)] for s in SEEDS], axis=0) - np.mean([O[("loc", s)] for s in SEEDS], axis=0), y


if __name__ == "__main__":
    t0 = time.time(); pop = pd.read_parquet(sys.argv[1]); sva, y = compute(pop)
    pop[KEYS].assign(sva=sva).to_parquet(sys.argv[2], index=False)
    print(f"  [pooled wh_sva, torch3] n={len(pop):,}, whiff_rate={y.mean():.4f}, sva_std={sva.std():.4f}, took {time.time() - t0:.0f}s", flush=True)
