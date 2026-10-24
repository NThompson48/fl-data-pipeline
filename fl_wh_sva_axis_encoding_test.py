"""
fl_wh_sva_axis_encoding_test.py -- does the spin axis need the sine / cosine encoding?  (user 2026-10-21: "Does it need to be encoded for a neural network?")
Same cross-fit as fl_wh_sva_axis_test.py (each season separately, swings with all 9 inputs and an axis, 5-fold GroupKFold by pitcher, full-model out-of-fold log-loss), 10 inputs or 11 inputs depending on the encoding of the
MIRRORED axis (axis_as, see that script):
  sincos    axis_sin, axis_cos                                            (2 inputs, circular, no seam)
  raw       axis_as / 360                                                 (1 input; the 0 / 360 seam falls through the curveball cluster: 11.8% of curveballs lie within 20 degrees of it)
  shifted   ((axis_as - 300) mod 360) / 360                               (1 input; seam moved to 300 degrees, the sparsest 60-degree window: 0.98% of pitches)
Estimators: PyTorch network (128-64-32, 3 seeds, scores for the 3-seed ensemble and the single-seed mean) and production XGBoost.
PRE-STATED reading (noise unit .00023 log-loss): the encoding matters if raw is worse than sincos by >= 1 noise unit; shifted is 'as good as' sincos if within 1 noise unit.  Output: output/wh_sva_axis_encoding_test.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load (WinError 1114)
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_axis_test as at

warnings.filterwarnings("ignore")
SEEDS = (1, 2, 3); NOISE = 0.00023


def main():
    raw = at.add_spin_axis(ppf.exclude(fp.load_fl_pitches_cached())); raw["axis_raw"] = raw["axis_as"] / 360.0; raw["axis_shift"] = ((raw["axis_as"] - 300.0) % 360.0) / 360.0
    ENC = {"sincos": ["axis_sin", "axis_cos"], "raw": ["axis_raw"], "shifted": ["axis_shift"]}; out = {}
    for yr in (2025, 2026):
        pop = raw[raw["is_swing"] & (raw["year"] == yr)].dropna(subset=fp.FULL_COLS + ["axis_sin", "axis_raw", "pitcher_id", "is_whiff"]).reset_index(drop=True)
        y = pop["is_whiff"].astype(int).to_numpy(); g = pop["pitcher_id"].to_numpy(); folds = list(GroupKFold(5).split(y, y, g)); M = {e: pop[fp.FULL_COLS + c].astype(float).to_numpy() for e, c in ENC.items()}; t0 = time.time()
        jobs = [delayed(at.tjob)(e, None, {}, s, k, tr, te, M[e], y, g) for e in ENC for s in SEEDS for k, (tr, te) in enumerate(folds)]; O = {}
        for tag, s, te, p in Parallel(n_jobs=8, backend="loky")(jobs):
            O.setdefault((tag, s), np.zeros(len(y)))[te] = p
        X = {e: np.zeros(len(y)) for e in ENC}
        for tr, te in folds:
            for e in ENC:
                X[e][te] = XGBClassifier(**{**fp.XGB_PARAMS, "n_jobs": 8}).fit(M[e][tr], y[tr]).predict_proba(M[e][te])[:, 1]
        clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6); rows = {}
        for e in ENC:
            rows[e] = {"torch ensemble": log_loss(y, clip(np.mean([O[(e, s)] for s in SEEDS], axis=0))), "torch single-seed mean": float(np.mean([log_loss(y, clip(O[(e, s)])) for s in SEEDS])), "XGBoost": log_loss(y, clip(X[e]))}
        T = pd.DataFrame(rows).T; D = T - T.loc["sincos"]; print(f"\n== {yr}: {len(y):,} swings ({(time.time() - t0) / 60:.1f} min)\nfull-model log-loss:\n{T.round(5).to_string()}\ndifference vs sincos (positive = worse), in noise units of {NOISE}:\n{(D / NOISE).round(1).to_string()}", flush=True)
        out[yr] = {"logloss": T.to_dict("index"), "diff_vs_sincos": D.to_dict("index")}
    json.dump(out, open("output/wh_sva_axis_encoding_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
