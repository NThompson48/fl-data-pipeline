"""
fl_wh_sva_strike_worker.py -- STRIKE stuff estimator (network), run as a SEPARATE PROCESS (torch must load before xgboost / scikit-learn on this Windows machine):
    python fl_wh_sva_strike_worker.py <pitches.parquet> <sva.parquet> <mode>      mode = loc | ct
Called by fl_pitcher_features.compute_pooled_wh_sva(estimator='strike_loc' | 'strike_ct').
Per pitch, sva = P(strike | 9 stuff + location inputs [+ balls, strikes in mode ct]) - P(strike | plate_x_as, plate_z [+ balls, strikes in mode ct]), 3-seed PyTorch MLP (128-64-32 / location 64-64), 5-fold GroupKFold by pitcher_id.
'strike' = called strike, swinging strike or foul (vs ball called / in dirt); balls in play, hit batters, intentional and automatic calls are excluded.  No pitch-type input (the request named stuff, count and location only).
Input: FULL_COLS + pitcher_id, date, game_id, pa_id, pitch_no, balls, strikes, target.  Output: pitcher_id, date, game_id, pa_id, pitch_no, sva.
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


def _job(tag, cfg, seed, tr, te, X, y, g):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg=cfg); return tag, seed, te, p


def compute(pop, mode):
    cnt = ["balls", "strikes"] if mode == "ct" else []
    y = pop["target"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); Xf = pop[fp.FULL_COLS + cnt].astype(float).to_numpy(); Xl = pop[fp.LOC_COLS + cnt].astype(float).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(Xf, y, g)); jobs = []
    for tag, X, cfg in (("full", Xf, {}), ("loc", Xl, {"hidden": (64, 64)})):
        jobs += [delayed(_job)(tag, cfg, s, tr, te, X, y, g) for s in SEEDS for tr, te in folds]
    O = {}
    for tag, s, te, p in Parallel(n_jobs=8, backend="loky")(jobs):
        O.setdefault((tag, s), np.zeros(len(y)))[te] = p
    return np.mean([O[("full", s)] for s in SEEDS], axis=0) - np.mean([O[("loc", s)] for s in SEEDS], axis=0), y


if __name__ == "__main__":
    t0 = time.time(); pop = pd.read_parquet(sys.argv[1]); mode = sys.argv[3]; sva, y = compute(pop, mode); pop[KEYS].assign(sva=sva).to_parquet(sys.argv[2], index=False)
    print(f"  [pooled strike sva, {mode}] n={len(pop):,}, strike_rate={y.mean():.4f}, sva_std={sva.std():.4f}, took {time.time() - t0:.0f}s", flush=True)
