"""
mlb_bb_model_edge_distance_backtest.py -- adds edge_dist_even and
edge_dist_behind (count-leverage-specific signed distance from the true zone
edge) to the BB% walk-forward backtest, on top of the existing horiz_dist/
vert_dist(+deltas) features. Uses the lighter-weight full-history-fit harness
(mlb_kbb_model.py's FOLDS/fit machinery) -- already confirmed in
mlb_kbb_walkforward.py to track the fully-rigorous per-fold refit within
<0.01 R2_w for the wh_sva features, and these two new features involve no
model fitting at all (just pitch-level aggregation), so there's no
look-ahead-through-hyperparameters risk to begin with.

Usage:
    python mlb_bb_model_edge_distance_backtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_kbb_model import (BB_FEATURES, FOLDS, MIN_PA, PANEL_PATH, _zscore_apply,
                            _zscore_fit, build_model_frame)
from mlb_count_leverage_command import leverage_bucket
from mlb_signed_edge_distance import signed_edge_distance

pd.set_option("display.width", 160, "display.max_columns", 40)

FAMILIES = ["FB", "BRK", "OFF"]
MIN_N_LEV = 50


def build_leverage_edge_features(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[raw["pitch_family"].isin(FAMILIES) & raw["plate_x"].notna() & raw["plate_z"].notna()].copy()
    raw["edge_dist"] = signed_edge_distance(raw["plate_x"].values, raw["plate_z"].values)
    raw["leverage"] = leverage_bucket(raw["balls"].values, raw["strikes"].values)

    out = None
    for lev in ["even", "behind"]:
        pop = raw[raw["leverage"] == lev]
        agg = pop.groupby(["pitcher", "game_year"])["edge_dist"].agg(
            [(f"edge_dist_{lev}", "mean"), (f"n_edge_{lev}", "size")]).reset_index()
        agg = agg[agg[f"n_edge_{lev}"] >= MIN_N_LEV]
        out = agg if out is None else out.merge(agg, on=["pitcher", "game_year"], how="outer")
    return out


def fit_eval(train, test, feature_cols, n0):
    d = train.dropna(subset=["Marcel_BB_rate", "BB_rate", "PA"]).copy()
    d["y"] = d["BB_rate"] * 100
    d["base"] = d["Marcel_BB_rate"] * 100
    d["wt"] = mm.reliability_weight(d["PA"], n0)
    stats = _zscore_fit(d, feature_cols)
    d = _zscore_apply(d, feature_cols, stats)
    zcols = [f"{c}_z" for c in feature_cols]
    formula = "y ~ base + age_c + age_decline + " + " + ".join(zcols)
    model = smf.wls(formula, data=d, weights=d["wt"]).fit()
    dt = test.dropna(subset=["Marcel_BB_rate", "BB_rate", "PA"]).copy()
    dt["base"] = dt["Marcel_BB_rate"] * 100
    dt = _zscore_apply(dt, feature_cols, stats)
    pred = model.predict(dt).values / 100
    return mm.evaluate_weighted(dt["BB_rate"].values, pred, dt["PA"].values, n0, label="")


if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)

    print("Building count-leverage edge-distance features...")
    edge_feat = build_leverage_edge_features(raw)
    print(f"n pitcher-seasons with edge features: {len(edge_feat)}")

    panel = pd.read_parquet(PANEL_PATH)
    mframe = build_model_frame(panel)

    # lag the new features by one year, matching the rest of LAGGED_COLS convention
    lag = edge_feat.rename(columns={"game_year": "target_year_feat"})
    lag["target_year"] = lag["target_year_feat"] + 1
    lag = lag.drop(columns="target_year_feat")
    mframe = mframe.merge(lag, on=["pitcher", "target_year"], how="left")
    for c in ["edge_dist_even", "edge_dist_behind"]:
        mframe[c] = mframe[c].fillna(mframe[c].median())

    print("\n=== Walk-forward BB% backtest: does count-leverage edge distance help? ===")
    feature_sets = [
        ("baseline (current BB model)", BB_FEATURES),
        ("+ edge_dist_even", BB_FEATURES + ["edge_dist_even"]),
        ("+ edge_dist_behind", BB_FEATURES + ["edge_dist_behind"]),
        ("+ both edge_dist_even & behind", BB_FEATURES + ["edge_dist_even", "edge_dist_behind"]),
    ]
    for label, feats in feature_sets:
        rows = []
        for train_years, test_year in FOLDS:
            train = mframe[mframe["target_year"].isin(train_years)]
            test = mframe[mframe["target_year"] == test_year]
            elig_train = train[train["PA"] >= MIN_PA]
            n0_bb, _ = mm.estimate_n0(elig_train["BB_rate"], elig_train["PA"])
            rows.append(fit_eval(train, test, feats, n0_bb))
        bt = pd.DataFrame(rows)
        print(f"  {label:36s}  R2_w={bt['R2_w'].mean():.4f}  RMSE_w={bt['RMSE_w_pp'].mean():.3f}pp  "
              f"MAE_w={bt['MAE_w_pp'].mean():.3f}pp")
