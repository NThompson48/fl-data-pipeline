"""
mlb_kbb_model.py -- K% and BB% projected independently from
output/mlb_pitcher_season_features.parquet, extending k26_projection.py's
architecture: reliability-weighted WLS on a Marcel-regressed base + age +
engineered arsenal/location features, replacing k26's Stuff+/arsenal terms with
the by-pitch-type wh_sva/swstr_sva family (K model) and horiz_dist/vert_dist
(BB model), both plus their within-season deltas.

Couplet structure (mirrors k26_projection.py's couplets(), but the "base" term
is already trailing-safe): for target season T, join
  - Marcel_K_rate/Marcel_BB_rate at row (pitcher, T)  -- already built ONLY from
    years < T (mlb_pitcher_features.marcel_projections), so no extra lag needed
  - age_c/age_decline at row (pitcher, T)              -- age DURING the projected
    season, not lagged
  - wh_sva/swstr_sva family, horiz_dist/vert_dist, BF_per_G, and their within-
    season deltas at row (pitcher, T-1)                -- these ARE lagged one
    year, since they're skill indicators from the season BEFORE the one being
    projected (same logic as k26_projection.py's stuff1 -> k2), not concurrent-
    season measurements of the very target we're predicting
  - K_rate/BB_rate/PA at row (pitcher, T)               -- the actual target

Usage:
    python mlb_kbb_model.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mlb_marcel as mm

PANEL_PATH = "output/mlb_pitcher_season_features.parquet"
MIN_PA = 50

LAGGED_COLS = ["wh_sva_fb", "wh_sva_brk", "wh_sva_off",
               "swstr_sva_fb", "swstr_sva_brk", "swstr_sva_off",
               "wh_sva_delta", "horiz_dist", "vert_dist",
               "horiz_dist_delta", "vert_dist_delta", "BF_per_G"]

K_FEATURES_WHIFF = ["wh_sva_fb", "wh_sva_brk", "wh_sva_off", "wh_sva_delta", "BF_per_G"]
K_FEATURES_SWSTR = ["swstr_sva_fb", "swstr_sva_brk", "swstr_sva_off", "wh_sva_delta", "BF_per_G"]
BB_FEATURES = ["horiz_dist", "vert_dist", "horiz_dist_delta", "vert_dist_delta"]


def build_model_frame(panel: pd.DataFrame) -> pd.DataFrame:
    feat = panel[["pitcher", "game_year"] + LAGGED_COLS].copy()
    feat["target_year"] = feat["game_year"] + 1
    feat = feat.drop(columns="game_year")

    target = panel[["pitcher", "game_year", "player_name", "Marcel_K_rate", "Marcel_BB_rate",
                     "Marcel_weighted_PA", "K_rate", "BB_rate", "PA", "age", "age_c",
                     "age_decline"]].rename(columns={"game_year": "target_year"})

    m = target.merge(feat, on=["pitcher", "target_year"], how="left")
    for c in LAGGED_COLS:
        m[f"has_{c}"] = m[c].notna()
        m[c] = m[c].fillna(0.0)
    return m


def _zscore_fit(train: pd.DataFrame, cols):
    stats = {c: (train[c].mean(), train[c].std() or 1.0) for c in cols}
    return stats


def _zscore_apply(df: pd.DataFrame, cols, stats):
    df = df.copy()
    for c in cols:
        mu, sd = stats[c]
        df[f"{c}_z"] = (df[c] - mu) / sd
    return df


def fit_rate_model(train: pd.DataFrame, rate: str, feature_cols: list[str], n0: float):
    """rate: 'K' or 'BB'. Returns fitted WLS model + z-score stats + formula used."""
    d = train.dropna(subset=[f"Marcel_{rate}_rate", f"{rate}_rate", "PA"]).copy()
    d["y"] = d[f"{rate}_rate"] * 100
    d["base"] = d[f"Marcel_{rate}_rate"] * 100
    d["wt"] = mm.reliability_weight(d["PA"], n0)
    stats = _zscore_fit(d, feature_cols)
    d = _zscore_apply(d, feature_cols, stats)
    zcols = [f"{c}_z" for c in feature_cols]
    formula = "y ~ base + age_c + age_decline + " + " + ".join(zcols)
    model = smf.wls(formula, data=d, weights=d["wt"]).fit()
    return model, stats, formula


def predict_rate(model, stats, feature_cols, df: pd.DataFrame, rate: str) -> np.ndarray:
    d = df.copy()
    d["base"] = d[f"Marcel_{rate}_rate"] * 100
    d = _zscore_apply(d, feature_cols, stats)
    return model.predict(d).values / 100


def evaluate_fold(model, stats, feature_cols, test: pd.DataFrame, rate: str, n0: float, label: str):
    d = test.dropna(subset=[f"Marcel_{rate}_rate", f"{rate}_rate", "PA"]).copy()
    pred_full = predict_rate(model, stats, feature_cols, d, rate)
    pred_base = d[f"Marcel_{rate}_rate"].values  # raw-Marcel-alone baseline
    y = d[f"{rate}_rate"].values
    n = d["PA"].values
    rows = [
        mm.evaluate_weighted(y, pred_full, n, n0, label=f"{label} full model"),
        mm.evaluate_weighted(y, pred_base, n, n0, label=f"{label} Marcel-alone baseline"),
    ]
    return pd.DataFrame(rows)


FOLDS = [((2018, 2019, 2020), 2021), ((2019, 2020, 2021), 2022), ((2020, 2021, 2022), 2023),
         ((2021, 2022, 2023), 2024), ((2022, 2023, 2024), 2025)]


def walk_forward_backtest(mframe: pd.DataFrame, k_feature_cols: list[str]):
    """Refits the WLS regression itself out-of-fold every time (never sees the test
    year's target rows). NOTE: this reuses the full-history-fit wh_sva/swstr_sva
    features, Marcel REG, and n0/reliability weights computed once over all years --
    a lighter-weight backtest than a fully rigorous re-derivation of those
    hyperparameters per fold (which would mean re-running the XGBoost wh_sva stage
    5 more times, ~30-40 min of extra compute). This still gives an honest read on
    whether the REGRESSION beats raw Marcel out-of-sample; it is a mildly optimistic
    upper bound specifically on how much the wh_sva/dist features would help in a
    fully re-derived walk-forward setting, since those features "saw" the fold's own
    test-year pitches during their own OOF-by-pitcher (not OOF-by-year) construction."""
    results_k, results_bb = [], []
    for train_years, test_year in FOLDS:
        train = mframe[mframe["target_year"].isin(train_years)]
        test = mframe[mframe["target_year"] == test_year]
        if len(train) < 200 or len(test) < 50:
            print(f"  skip fold ->{test_year}: train={len(train)} test={len(test)}")
            continue
        elig_train = train[train["PA"] >= MIN_PA]
        n0_k, _ = mm.estimate_n0(elig_train["K_rate"], elig_train["PA"])
        n0_bb, _ = mm.estimate_n0(elig_train["BB_rate"], elig_train["PA"])

        mk, sk, _ = fit_rate_model(train, "K", k_feature_cols, n0_k)
        mbb, sbb, _ = fit_rate_model(train, "BB", BB_FEATURES, n0_bb)

        results_k.append(evaluate_fold(mk, sk, k_feature_cols, test, "K", n0_k, f"->{test_year}"))
        results_bb.append(evaluate_fold(mbb, sbb, BB_FEATURES, test, "BB", n0_bb, f"->{test_year}"))
    return pd.concat(results_k, ignore_index=True), pd.concat(results_bb, ignore_index=True)


if __name__ == "__main__":
    pd.set_option("display.width", 160, "display.max_columns", 40)
    print("Loading panel...")
    panel = pd.read_parquet(PANEL_PATH)
    print(panel.shape)

    mframe = build_model_frame(panel)
    print(f"model frame: {mframe.shape}")
    elig = mframe[mframe["PA"] >= MIN_PA]
    print(f"eligible (PA>={MIN_PA}) rows with a target: {len(elig)}")
    print(f"  rows with lagged features present (has_BF_per_G): {elig['has_BF_per_G'].sum()} "
          f"/ {len(elig)}")

    print("\n=== K model: whiff-conditioned (wh_sva) family features ===")
    k_bt_whiff, bb_bt = walk_forward_backtest(mframe, K_FEATURES_WHIFF)
    print(k_bt_whiff.to_string(index=False))

    print("\n=== K model: unconditional (swstr_sva) family features ===")
    k_bt_swstr, _ = walk_forward_backtest(mframe, K_FEATURES_SWSTR)
    print(k_bt_swstr.to_string(index=False))

    print("\n=== BB model ===")
    print(bb_bt.to_string(index=False))

    print("\n=== Summary: reliability-weighted R2, averaged across folds ===")
    for name, bt in [("K (whiff-conditioned)", k_bt_whiff), ("K (unconditional swstr)", k_bt_swstr),
                      ("BB", bb_bt)]:
        full = bt[bt["label"].str.contains("full model")]
        base = bt[bt["label"].str.contains("baseline")]
        print(f"{name}: full R2_w={full['R2_w'].mean():.4f} RMSE_w={full['RMSE_w_pp'].mean():.3f}pp | "
              f"Marcel-alone R2_w={base['R2_w'].mean():.4f} RMSE_w={base['RMSE_w_pp'].mean():.3f}pp")
