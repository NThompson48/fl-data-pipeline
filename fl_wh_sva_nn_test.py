"""
fl_wh_sva_nn_test.py -- downstream test of a neural-network wh_sva (user 2026-10-21), after the pilot (fl_wh_sva_nn_pilot.py) met its pre-stated rule for the single-seed and 3-seed networks
(per-pitch out-of-fold log-loss .4722 -> .4705 / .4691, AUC .731 -> .735 / .737; pitcher-level split-half reliability .949 -> .972 / .966, implied n0 56 -> 31 / 37 pitches).

PROTOCOL = fl_wh_sva_variants_test.py unchanged (that module's main() is run with the estimator swapped):
  every variant's pitcher-level wh_sva_level replaces production's; the frozen K% model is REFIT with it (training transition full 2025 -> actual 1H2026); fold 2 = features 2025 + 1H2026, targets actual 2H2026;
  position players excluded; level-free; standard inverse-variance weights from the PRODUCTION variant's residuals, frozen (fl_eval_weights); weighted RMSE AND MAE; pitcher bootstrap (1,000) vs production.
  Also (a) the BB% model + each variant's level, (b) split-half reliability of the pitcher level and its implied n0, (c) concurrent correlation with K%.
Variants (labels as the harness expects; 'whiff' IS production, run through the production code):
  whiff    production: XGBoost (depth 4, 100 trees) full - location, 5-fold GroupKFold by pitcher
  nn       sklearn MLP 128-64-32 (location model 64-64), one seed, same cross-fit
  nn3      the same MLP averaged over 3 seeds (1, 2, 3)
  blend    mean of the production sva and the nn3 sva (per pitch)       [did NOT meet the pilot rule (reliability +.012 < .03); run as an exploratory add-on, flagged]
Adoption rule (project standard): RMSE and MAE both better than production with P(better) >= .90.  The pilot's hyper-parameters were fixed in advance and are not tuned here.
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import fl_pitcher_features as fp
import fl_wh_sva_nn_pilot as pilot
import fl_wh_sva_variants_test as vt

warnings.filterwarnings("ignore")
KEYS = ["pitcher_id", "date", "game_id", "pa_id", "pitch_no"]
_CACHE = {}


def _nn_fold(tr, te, X_loc, X_full, y):
    out = {}
    with threadpool_limits(limits=3):
        for tag, X, hidden in (("loc", X_loc, (64, 64)), ("full", X_full, (128, 64, 32))):
            ps = [pilot._mlp(X[tr], y[tr], X[te], s, hidden)[0] for s in pilot.SEEDS]
            out[f"nn_{tag}"] = ps[0]; out[f"nn3_{tag}"] = np.mean(ps, axis=0)
    return te, out


def components(df, min_rows=2000):
    """production / nn / nn3 / blend sva frames for one window of pitches (memoised: the harness asks for the same window several times)."""
    prod = vt._orig(df, min_rows)
    if len(prod) == 0:
        return {k: prod for k in ("whiff", "nn", "nn3", "blend")}
    key = int(pd.util.hash_pandas_object(prod[["pitcher_id", "game_id", "pa_id", "pitch_no"]].astype(str), index=False).sum())
    if key in _CACHE:
        return _CACHE[key]
    pop = df[df["is_swing"]].dropna(subset=fp.FULL_COLS + ["pitcher_id", "is_whiff", "date", "game_id", "pa_id", "pitch_no"]).copy()
    assert len(pop) == len(prod) and (pop["game_id"].to_numpy() == prod["game_id"].to_numpy()).all()
    y = pop["is_whiff"].astype(int).to_numpy(); X_loc = pop[fp.LOC_COLS].astype(float).to_numpy(); X_full = pop[fp.FULL_COLS].astype(float).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X_full, y, pop["pitcher_id"].to_numpy()))
    res = Parallel(n_jobs=5, backend="loky")(delayed(_nn_fold)(tr, te, X_loc, X_full, y) for tr, te in folds)
    P = {k: np.zeros(len(y)) for k in ("nn_loc", "nn_full", "nn3_loc", "nn3_full")}
    for te, o in res:
        for k in P:
            P[k][te] = o[k]
    frames = {"whiff": prod}
    for lab in ("nn", "nn3"):
        frames[lab] = prod.assign(sva=P[f"{lab}_full"] - P[f"{lab}_loc"])
    frames["blend"] = prod.assign(sva=0.5 * (prod["sva"].to_numpy() + frames["nn3"]["sva"].to_numpy()))
    print(f"    [nn components] window of {len(pop):,} swings: sva sd prod {prod['sva'].std():.4f} nn {frames['nn']['sva'].std():.4f} nn3 {frames['nn3']['sva'].std():.4f}", flush=True)
    _CACHE[key] = frames; return frames


def dispatch(df, min_rows=2000):
    lab = vt.CUR["target"]
    return vt._orig(df, min_rows) if lab == "whiff" else components(df, min_rows)[lab]


if __name__ == "__main__":
    vt.VARIANTS = [("whiff", False), ("nn", False), ("nn3", False), ("blend", False)]
    vt.sva_variant = dispatch
    vt.main()
