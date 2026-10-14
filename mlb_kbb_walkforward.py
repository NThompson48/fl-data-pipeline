"""
mlb_kbb_walkforward.py -- the RIGOROUS walk-forward backtest: unlike
mlb_kbb_model.py's first pass (which reused the full-history wh_sva/swstr_sva
panel for speed), this refits REG_K/REG_BB, n0, the shrinkage params, AND the
wh_sva/swstr_sva XGBoost stage using ONLY each fold's training years -- so the
test year's pitches never inform any hyperparameter, not just the final
regression. Per the approved plan's validation section.

horiz_dist/vert_dist need no fold-specific refit (deterministic per-season
aggregates, no modeling involved) -- only the two XGBoost-based pieces
(family-level wh_sva/swstr_sva, and the pooled overall wh_sva used for the
delta) carry real look-ahead risk if fit on the full 8-year table.

Usage:
    python mlb_kbb_walkforward.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_kbb_model import (BB_FEATURES, FOLDS, K_FEATURES_SWSTR, K_FEATURES_WHIFF, MIN_PA,
                            _zscore_apply, _zscore_fit)

pd.set_option("display.width", 160, "display.max_columns", 40)


def build_fold_frame(season, marcel_fold, family_sva_fold, dist, deltas_fold, age, target_years):
    feat_cols = [c for c in family_sva_fold.columns if c not in ("pitcher", "game_year")]
    feat = family_sva_fold[["pitcher", "game_year"] + feat_cols].copy()
    feat = feat.merge(dist, on=["pitcher", "game_year"], how="outer")
    feat = feat.merge(deltas_fold, on=["pitcher", "game_year"], how="outer")
    feat = feat.merge(season[["pitcher", "game_year", "BF_per_G"]], on=["pitcher", "game_year"], how="outer")
    feat["target_year"] = feat["game_year"] + 1
    feat = feat.drop(columns="game_year")

    target = season[season["game_year"].isin(target_years)][
        ["pitcher", "game_year", "K_rate", "BB_rate", "PA"]].rename(columns={"game_year": "target_year"})
    target = target.merge(marcel_fold[marcel_fold["game_year"].isin(target_years)].rename(
        columns={"game_year": "target_year"}), on=["pitcher", "target_year"], how="left")
    target = target.merge(age.rename(columns={"game_year": "target_year"}),
                           on=["pitcher", "target_year"], how="left")

    m = target.merge(feat, on=["pitcher", "target_year"], how="left")
    lagged_cols = [c for c in feat.columns if c not in ("pitcher", "target_year")]
    for c in lagged_cols:
        if c not in m.columns:
            continue
        if pd.api.types.is_bool_dtype(m[c]) or str(m[c].dtype) == "boolean":
            m[c] = m[c].fillna(False)
        else:
            m[c] = m[c].fillna(0.0)
    return m


def fit_rate_model(train, rate, feature_cols, n0):
    d = train.dropna(subset=[f"Marcel_{rate}_rate", f"{rate}_rate", "PA"]).copy()
    d["y"] = d[f"{rate}_rate"] * 100
    d["base"] = d[f"Marcel_{rate}_rate"] * 100
    d["wt"] = mm.reliability_weight(d["PA"], n0)
    stats = _zscore_fit(d, feature_cols)
    d = _zscore_apply(d, feature_cols, stats)
    zcols = [f"{c}_z" for c in feature_cols]
    formula = "y ~ base + age_c + age_decline + " + " + ".join(zcols)
    model = smf.wls(formula, data=d, weights=d["wt"]).fit()
    return model, stats


def predict_rate(model, stats, feature_cols, df, rate):
    d = df.copy()
    d["base"] = d[f"Marcel_{rate}_rate"] * 100
    d = _zscore_apply(d, feature_cols, stats)
    return model.predict(d).values / 100


def evaluate(model, stats, feature_cols, test, rate, n0, label):
    d = test.dropna(subset=[f"Marcel_{rate}_rate", f"{rate}_rate", "PA"]).copy()
    pred_full = predict_rate(model, stats, feature_cols, d, rate)
    pred_base = d[f"Marcel_{rate}_rate"].values
    y, n = d[f"{rate}_rate"].values, d["PA"].values
    return pd.DataFrame([
        mm.evaluate_weighted(y, pred_full, n, n0, label=f"{label} full model"),
        mm.evaluate_weighted(y, pred_base, n, n0, label=f"{label} Marcel-alone baseline"),
    ])


if __name__ == "__main__":
    t_start = time.time()
    print("Loading pitch-level data + flags (once, reused across all folds)...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)
    season = mf.season_rates(raw)
    dist = mf.compute_dist_features(raw)
    age = mf.compute_age(season)
    print(f"shared setup done in {time.time()-t_start:.0f}s")

    results = {"K_whiff": [], "K_swstr": [], "BB": []}
    for train_years, test_year in FOLDS:
        t0 = time.time()
        print(f"\n=== Fold: train={train_years} -> test={test_year} ===")

        elig_train_season = season[season["game_year"].isin(train_years) & (season["PA"] >= MIN_PA)]
        reg_k, _ = mm.estimate_n0(elig_train_season["K_rate"], elig_train_season["PA"])
        reg_bb, _ = mm.estimate_n0(elig_train_season["BB_rate"], elig_train_season["PA"])
        n0_k, _ = mm.estimate_n0(elig_train_season["K_rate"], elig_train_season["PA"])
        n0_bb, _ = mm.estimate_n0(elig_train_season["BB_rate"], elig_train_season["PA"])
        marcel_fold = mf.marcel_projections(season, reg_k, reg_bb)

        print("  refitting wh_sva/swstr_sva family models on training years only...")
        family_sva_fold = None
        for condition in mf.CONDITIONS:
            prefix = "wh_sva" if condition == "swing" else "swstr_sva"
            for fam in mf.FAMILIES:
                agg = mf.compute_shape_value_added(raw, condition=condition, family=fam, years=train_years)
                fl = fam.lower()
                agg = agg.rename(columns={"sva": f"{prefix}_{fl}", "n": f"n_{prefix}_{fl}",
                                           "rate": f"rate_{prefix}_{fl}"})
                agg[f"is_imputed_{prefix}_{fl}"] = agg[f"n_{prefix}_{fl}"] < mf.MIN_N_FAMILY
                agg.loc[agg[f"is_imputed_{prefix}_{fl}"], f"{prefix}_{fl}"] = 0.0
                family_sva_fold = agg if family_sva_fold is None else family_sva_fold.merge(
                    agg, on=["pitcher", "game_year"], how="outer")

        print("  refitting pooled wh_sva (for delta) on training years only...")
        wh_sva_pitch_fold = mf.pooled_wh_sva_per_pitch(raw, years=train_years)
        raw_train = raw[raw["game_year"].isin(train_years)]
        deltas_fold = mf.compute_within_season_deltas(raw_train, wh_sva_pitch_fold)

        mframe = build_fold_frame(season, marcel_fold, family_sva_fold, dist, deltas_fold, age,
                                   target_years=list(train_years) + [test_year])
        train = mframe[mframe["target_year"].isin(train_years)]
        test = mframe[mframe["target_year"] == test_year]
        print(f"  train rows={len(train)}, test rows={len(test)}")

        mk_w, sk_w = fit_rate_model(train, "K", K_FEATURES_WHIFF, n0_k)
        mk_s, sk_s = fit_rate_model(train, "K", K_FEATURES_SWSTR, n0_k)
        mbb, sbb = fit_rate_model(train, "BB", BB_FEATURES, n0_bb)

        results["K_whiff"].append(evaluate(mk_w, sk_w, K_FEATURES_WHIFF, test, "K", n0_k, f"->{test_year}"))
        results["K_swstr"].append(evaluate(mk_s, sk_s, K_FEATURES_SWSTR, test, "K", n0_k, f"->{test_year}"))
        results["BB"].append(evaluate(mbb, sbb, BB_FEATURES, test, "BB", n0_bb, f"->{test_year}"))
        print(f"  fold done in {time.time()-t0:.0f}s")

    print("\n\n========== RIGOROUS WALK-FORWARD RESULTS (no hyperparameter look-ahead) ==========")
    for name, rows in results.items():
        bt = pd.concat(rows, ignore_index=True)
        print(f"\n--- {name} ---")
        print(bt.to_string(index=False))
        full = bt[bt["label"].str.contains("full model")]
        base = bt[bt["label"].str.contains("baseline")]
        print(f"AVG: full R2_w={full['R2_w'].mean():.4f} RMSE_w={full['RMSE_w_pp'].mean():.3f}pp | "
              f"Marcel-alone R2_w={base['R2_w'].mean():.4f} RMSE_w={base['RMSE_w_pp'].mean():.3f}pp")
        bt.to_csv(f"output/walkforward_{name}.csv", index=False)

    print(f"\ntotal runtime: {time.time()-t_start:.0f}s")
