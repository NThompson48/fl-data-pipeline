"""
fl_wh_sva_whiffstuff_worker.py -- P(whiff | stuff) network, run as a SEPARATE PROCESS (torch must load before xgboost / scikit-learn on this Windows machine):  python fl_wh_sva_whiffstuff_worker.py <pitches.parquet> <out.parquet>
Called by fl_pitcher_features.compute_pooled_wh_sva(estimator='whiff_stuff' | 'whiff_stuff_sw').
MODEL: P(whiff | swing, stuff) with stuff = the 7 shape / release inputs (velocity, ivb, arm-side horizontal break, spin rate, extension, release height, release side) -- NO location and NO count, so there is no baseline to subtract and the prediction is the pitch's
whiff potential given its stuff alone.  Trained on SWINGS (a whiff needs a swing), 3-seed MLP 128-64-32, 5-fold GroupKFold by pitcher_id: each pitcher's pitches are scored by models that never saw him; the prediction is made for EVERY pitch he threw (swing or not).
Output: pitcher_id, date, game_id, pa_id, pitch_no, is_swing, sva  (sva = prediction minus the mean prediction over all pitches in the window; the level is the mean over all pitches ('whiff_stuff') or over swings only ('whiff_stuff_sw')).
Input: STUFF_COLS + pitcher_id, date, game_id, pa_id, pitch_no, is_swing, is_whiff.
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
SEEDS = (1, 2, 3); KEYS = ["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]; STUFF = [c for c in fp.FULL_COLS if c not in fp.LOC_COLS]


def _job(seed, tr, te, X, y, g, train_mask):
    torch.set_num_threads(2); trs = tr[train_mask[tr]]
    p, ep, best = ft.fit_predict(X[trs], y[trs], X[te], g[trs], seed=seed, cfg={}); return seed, te, p


def compute(pop, mode="whiff"):
    """mode 'whiff': P(whiff | swing, stuff), trained on swings.  mode 'swstr': P(swinging strike | stuff) per PITCH, trained on every pitch (target = swing and whiff)."""
    sw = pop["is_swing"].to_numpy().astype(bool); wh = pop["is_whiff"].fillna(0).astype(int).to_numpy(); y = wh if mode == "whiff" else (sw & (wh == 1)).astype(int); train_mask = sw if mode == "whiff" else np.ones(len(y), dtype=bool)
    g = pop["pitcher_id"].to_numpy(); X = pop[STUFF].astype(float).to_numpy(); folds = list(GroupKFold(n_splits=5).split(X, y, g)); O = {}
    for seed, te, p in Parallel(n_jobs=8, backend="loky")(delayed(_job)(s, tr, te, X, y, g, train_mask) for s in SEEDS for tr, te in folds):
        O.setdefault(seed, np.zeros(len(y)))[te] = p
    p = np.mean([O[s] for s in SEEDS], axis=0); return p - p.mean(), y, sw


if __name__ == "__main__":
    t0 = time.time(); mode = sys.argv[3] if len(sys.argv) > 3 else "whiff"; pop = pd.read_parquet(sys.argv[1]); sva, y, sw = compute(pop, mode); pop[KEYS + ["is_swing"]].assign(sva=sva).to_parquet(sys.argv[2], index=False)
    print(f"  [pooled {mode}|stuff] n={len(pop):,} pitches ({int(sw.sum()):,} swings), target rate={y[sw].mean() if mode == 'whiff' else y.mean():.4f}, sva_std={sva.std():.4f}, took {time.time() - t0:.0f}s", flush=True)
