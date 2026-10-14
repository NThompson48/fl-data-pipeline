"""
mlb_shape_location_interaction.py -- does the OPTIMAL location for a breaking
ball/offspeed pitch depend on how it moves (a true interaction), not just
"location matters" and "movement matters" as separate additive effects?

Test: for each family (BRK, OFF), swing-conditioned whiff probability from
location + movement features, fit with:
  - ADDITIVE model: XGBoost with max_depth=1 (stumps) -- a depth-1 tree can only
    split on ONE feature per tree, so the ensemble sum is provably additive,
    no interactions possible between features, only their independent effects
  - INTERACTION model: XGBoost with the project's normal max_depth=4 -- trees
    can split on combinations of features, so a location split's threshold can
    differ depending on movement (exactly what "optimal location depends on
    movement" means)
Compare via GroupKFold(5, groups=pitcher) OOF AUC/logloss -- if INTERACTION
beats ADDITIVE beyond noise, that's direct evidence the optimal location
genuinely shifts with movement, not just that both matter independently.

If real, quantify it: bin horizontal break into terciles and, for each bin,
find the predicted-whiff-maximizing location on a grid -- does the peak
actually move?

Usage:
    python mlb_shape_location_interaction.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from xgboost import XGBClassifier

import mlb_pitcher_features as mf

pd.set_option("display.width", 160, "display.max_columns", 40)

LOC_COLS = ["plate_x_as", "plate_z"]
MOVE_COLS_BRK = ["horz_break_as", "ivb"]
MOVE_COLS_OFF = ["horz_break_as", "ivb", "effective_speed"]  # velocity separation matters more for OFF

ADD_PARAMS = dict(n_estimators=300, max_depth=1, learning_rate=0.1, subsample=0.8,
                   colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist")
INT_PARAMS = dict(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
                   colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist")


def compare_additive_vs_interaction(pop: pd.DataFrame, feature_cols: list[str], label: str):
    d = pop.dropna(subset=feature_cols + ["pitcher", "is_whiff"]).copy()
    y = d["is_whiff"].astype(int).values
    groups = d["pitcher"].values
    X = d[feature_cols].astype(float).values
    gkf = GroupKFold(n_splits=5)

    oof_add = cross_val_predict(XGBClassifier(**ADD_PARAMS), X, y, cv=gkf, groups=groups,
                                 method="predict_proba", n_jobs=5)[:, 1]
    oof_int = cross_val_predict(XGBClassifier(**INT_PARAMS), X, y, cv=gkf, groups=groups,
                                 method="predict_proba", n_jobs=5)[:, 1]

    auc_add, auc_int = roc_auc_score(y, oof_add), roc_auc_score(y, oof_int)
    ll_add, ll_int = log_loss(y, np.clip(oof_add, 1e-6, 1 - 1e-6)), log_loss(y, np.clip(oof_int, 1e-6, 1 - 1e-6))
    print(f"  [{label}] n={len(d):,}")
    print(f"    ADDITIVE (depth=1, no interactions):     AUC={auc_add:.4f}  logloss={ll_add:.4f}")
    print(f"    INTERACTION (depth=4, full model):       AUC={auc_int:.4f}  logloss={ll_int:.4f}")
    print(f"    delta: AUC {auc_int-auc_add:+.4f}, logloss {ll_int-ll_add:+.4f} (negative = interaction model better)")
    return d, oof_int


def quantify_shift(d: pd.DataFrame, feature_cols: list[str], break_col: str, label: str):
    """Fit the INTERACTION model on everything, then check: does the predicted-
    whiff-maximizing location move across terciles of `break_col`?"""
    y = d["is_whiff"].astype(int).values
    X = d[feature_cols].astype(float).values
    model = XGBClassifier(**INT_PARAMS).fit(X, y)

    terciles = d[break_col].quantile([1/3, 2/3]).values
    bins = pd.cut(d[break_col], bins=[-99, terciles[0], terciles[1], 99], labels=["low", "mid", "high"])
    grid_x = np.linspace(d["plate_x_as"].quantile(.05), d["plate_x_as"].quantile(.95), 25)
    grid_z = np.linspace(d["plate_z"].quantile(.05), d["plate_z"].quantile(.95), 25)
    gx, gz = np.meshgrid(grid_x, grid_z)

    print(f"\n  Optimal-location shift by {break_col} tercile ({label}):")
    for tname in ["low", "mid", "high"]:
        sub = d[bins == tname]
        fixed = {c: sub[c].median() for c in feature_cols if c not in LOC_COLS}
        grid_df = pd.DataFrame({"plate_x_as": gx.ravel(), "plate_z": gz.ravel()})
        for c, v in fixed.items():
            grid_df[c] = v
        pred = model.predict_proba(grid_df[feature_cols].astype(float).values)[:, 1]
        i_max = np.argmax(pred)
        print(f"    {tname:5s} {break_col} (median={sub[break_col].median():+.2f}, n={len(sub):,}): "
              f"peak whiff-prob location = (plate_x_as={grid_df['plate_x_as'].iloc[i_max]:+.2f}, "
              f"plate_z={grid_df['plate_z'].iloc[i_max]:.2f}), peak P(whiff)={pred[i_max]:.3f}")


if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)

    for family, move_cols, break_col in [("BRK", MOVE_COLS_BRK, "horz_break_as"),
                                           ("OFF", MOVE_COLS_OFF, "horz_break_as")]:
        print(f"\n{'='*70}\n{family}\n{'='*70}")
        pop = raw[(raw["pitch_family"] == family) & raw["is_swing"]]
        feature_cols = LOC_COLS + move_cols
        d, oof_int = compare_additive_vs_interaction(pop, feature_cols, family)
        quantify_shift(d, feature_cols, break_col, family)
        quantify_shift(d, feature_cols, "ivb", family)
