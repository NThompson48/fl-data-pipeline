"""
fl_wh_sva_swstr_worker.py -- per-pitch swinging-strike stuff estimator with pitch type and count, run as a SEPARATE PROCESS (torch must load before xgboost / scikit-learn on this Windows machine):  python fl_wh_sva_swstr_worker.py <pitches.parquet> <sva.parquet>
Called by fl_pitcher_features.compute_pooled_wh_sva(estimator='swstr_ct').  Per pitch, sva = P(swinging strike | 9 inputs, pitch type [4-d embedding], balls, strikes) - P(swinging strike | plate_x_as, plate_z, pitch type, balls, strikes), 3-seed network
(128-64-32 / location 64-64), 5-fold GroupKFold by pitcher_id.  Input: FULL_COLS + pitcher_id, date, game_id, pa_id, pitch_no, pitch_type, balls, strikes, target.  Output: pitcher_id, date, game_id, pa_id, pitch_no, sva.
"""
import torch  # MUST come first
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
SEEDS = (1, 2, 3); KEYS = ["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]


def _job(tag, cfg, seed, tr, te, X, y, g, C):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg={**cfg, "n_cat": int(C.max()) + 1}, cat_tr=C[tr], cat_te=C[te]); return tag, seed, te, p


def compute(pop):
    y = pop["target"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); CNT = ["balls", "strikes"]; Xf = pop[fp.FULL_COLS + CNT].astype(float).to_numpy(); Xl = pop[fp.LOC_COLS + CNT].astype(float).to_numpy()
    C = pd.Categorical(pop["pitch_type"]).codes.astype(int); folds = list(GroupKFold(n_splits=5).split(Xf, y, g)); jobs = []
    for tag, X, cfg in (("full", Xf, {}), ("loc", Xl, {"hidden": (64, 64)})):
        jobs += [delayed(_job)(tag, cfg, s, tr, te, X, y, g, C) for s in SEEDS for tr, te in folds]
    O = {}
    for tag, s, te, p in Parallel(n_jobs=8, backend="loky")(jobs):
        O.setdefault((tag, s), np.zeros(len(y)))[te] = p
    return np.mean([O[("full", s)] for s in SEEDS], axis=0) - np.mean([O[("loc", s)] for s in SEEDS], axis=0), y


if __name__ == "__main__":
    t0 = time.time(); pop = pd.read_parquet(sys.argv[1]); sva, y = compute(pop); pop[KEYS].assign(sva=sva).to_parquet(sys.argv[2], index=False)
    print(f"  [pooled swstr_ct sva] n={len(pop):,}, swstr_rate={y.mean():.4f}, sva_std={sva.std():.4f}, took {time.time() - t0:.0f}s", flush=True)
