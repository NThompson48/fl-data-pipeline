"""
mlb_aging_spec_backtest.py -- does a more flexible age functional form actually
beat the current linear+kink(30) shape out-of-sample, or does it just overfit?

Reuses the lighter-weight (full-history wh_sva) walk-forward harness from
mlb_kbb_model.py -- already confirmed in mlb_kbb_walkforward.py to track the
fully-rigorous per-fold refit within <0.01 R2_w, so it's a valid, much cheaper
testbed for THIS specific ablation (the age-term choice is orthogonal to how
wh_sva itself is fit). Tests, for K% (BB% aging was empirically flat/noisy in
mlb_aging_exploration.py, so also checks whether DROPPING age entirely costs
anything for BB%):
  - current: age_c + age_decline (linear, kink at 30)
  - pure linear (no kink)
  - kink at 33 instead of 30 (matches the late-in-decade acceleration seen in
    the empirical delta-method curve at age 35 across every subgroup)
  - flexible age bins (fully non-parametric, one dummy per ~3-year bin)

Usage:
    python mlb_aging_spec_backtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mlb_marcel as mm
from mlb_kbb_model import (BB_FEATURES, FOLDS, K_FEATURES_SWSTR, MIN_PA, PANEL_PATH,
                            _zscore_apply, _zscore_fit, build_model_frame)

pd.set_option("display.width", 160, "display.max_columns", 40)

AGE_BIN_EDGES = [0, 25, 28, 31, 34, 99]
AGE_BIN_LABELS = ["<=25", "26-28", "29-31", "32-34", "35+"]


def add_age_variants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_decline_33"] = (df["age"] - 33).clip(lower=0)
    df["age_bin"] = pd.cut(df["age"], bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS)
    return df


AGE_SPECS = {
    "linear+kink30 (current)": ["age_c", "age_decline"],
    "pure linear (no kink)": ["age_c"],
    "linear+kink33": ["age_c", "age_decline_33"],
    "no age term": [],
}


def fit_and_eval(train, test, rate, feature_cols, age_terms, n0, use_bins=False):
    d = train.dropna(subset=[f"Marcel_{rate}_rate", f"{rate}_rate", "PA"]).copy()
    d["y"] = d[f"{rate}_rate"] * 100
    d["base"] = d[f"Marcel_{rate}_rate"] * 100
    d["wt"] = mm.reliability_weight(d["PA"], n0)
    stats = _zscore_fit(d, feature_cols)
    d = _zscore_apply(d, feature_cols, stats)
    zcols = [f"{c}_z" for c in feature_cols]
    if use_bins:
        age_part = "C(age_bin, Treatment(reference='<=25'))"
    else:
        age_part = " + ".join(age_terms) if age_terms else None
    rhs = ["base"] + ([age_part] if age_part else []) + zcols
    formula = "y ~ " + " + ".join(rhs)
    model = smf.wls(formula, data=d, weights=d["wt"]).fit()

    dt = test.dropna(subset=[f"Marcel_{rate}_rate", f"{rate}_rate", "PA"]).copy()
    dt["base"] = dt[f"Marcel_{rate}_rate"] * 100
    dt = _zscore_apply(dt, feature_cols, stats)
    pred = model.predict(dt).values / 100
    y, n = dt[f"{rate}_rate"].values, dt["PA"].values
    return mm.evaluate_weighted(y, pred, n, n0, label="")


if __name__ == "__main__":
    panel = pd.read_parquet(PANEL_PATH)
    mframe = add_age_variants(build_model_frame(panel))

    print("=== K% model: age-spec comparison, averaged across all 5 folds ===")
    for spec_name, age_terms in AGE_SPECS.items():
        if spec_name == "no age term":
            continue  # K% clearly ages; skip the degenerate no-age case for K
        rows = []
        for train_years, test_year in FOLDS:
            train = mframe[mframe["target_year"].isin(train_years)]
            test = mframe[mframe["target_year"] == test_year]
            elig_train = train[train["PA"] >= MIN_PA]
            n0_k, _ = mm.estimate_n0(elig_train["K_rate"], elig_train["PA"])
            rows.append(fit_and_eval(train, test, "K", K_FEATURES_SWSTR, age_terms, n0_k,
                                      use_bins=(spec_name == "flexible age bins")))
        bt = pd.DataFrame(rows)
        print(f"  {spec_name:28s}  R2_w={bt['R2_w'].mean():.4f}  RMSE_w={bt['RMSE_w_pp'].mean():.3f}pp  "
              f"MAE_w={bt['MAE_w_pp'].mean():.3f}pp")

    # flexible age bins, separately (different formula path)
    rows = []
    for train_years, test_year in FOLDS:
        train = mframe[mframe["target_year"].isin(train_years)]
        test = mframe[mframe["target_year"] == test_year]
        elig_train = train[train["PA"] >= MIN_PA]
        n0_k, _ = mm.estimate_n0(elig_train["K_rate"], elig_train["PA"])
        rows.append(fit_and_eval(train, test, "K", K_FEATURES_SWSTR, [], n0_k, use_bins=True))
    bt = pd.DataFrame(rows)
    print(f"  {'flexible age bins':28s}  R2_w={bt['R2_w'].mean():.4f}  RMSE_w={bt['RMSE_w_pp'].mean():.3f}pp  "
          f"MAE_w={bt['MAE_w_pp'].mean():.3f}pp")

    print("\n=== BB% model: does age help AT ALL out-of-sample? ===")
    for spec_name, age_terms in AGE_SPECS.items():
        if spec_name == "flexible age bins":
            continue
        rows = []
        for train_years, test_year in FOLDS:
            train = mframe[mframe["target_year"].isin(train_years)]
            test = mframe[mframe["target_year"] == test_year]
            elig_train = train[train["PA"] >= MIN_PA]
            n0_bb, _ = mm.estimate_n0(elig_train["BB_rate"], elig_train["PA"])
            rows.append(fit_and_eval(train, test, "BB", BB_FEATURES, age_terms, n0_bb))
        bt = pd.DataFrame(rows)
        print(f"  {spec_name:28s}  R2_w={bt['R2_w'].mean():.4f}  RMSE_w={bt['RMSE_w_pp'].mean():.3f}pp  "
              f"MAE_w={bt['MAE_w_pp'].mean():.3f}pp")
