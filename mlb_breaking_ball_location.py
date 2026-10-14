"""
mlb_breaking_ball_location.py -- does BREAKING-BALL and OFFSPEED location (not
shape) matter for K%, and does it add anything beyond wh_sva_*/swstr_sva_*?

wh_sva_*/swstr_sva_* are already LOCATION-RESIDUALIZED by construction (the
two-stage architecture explicitly nets location out to isolate the shape
contribution) -- so location's own effect on off-speed-family whiffs has never
been directly tested in this project. For EACH of BRK and OFF, checks:
  1. Raw horiz_dist/vert_dist (distance from zone center) vs. K_rate
  2. Specifically "burying" the pitch below the zone (a common approach idea,
     esp. relevant for changeups/splitters too) vs. K_rate/whiff rate
  3. The family's location-ONLY two-stage OOF score (the same
     P(whiff|swing, location) stage already computed inside wh_sva's own
     construction, just never surfaced on its own before) vs. K_rate
  4. Incremental value: does adding that family's location to the already-
     validated swstr_sva-family model change next-year K% predictive power?

Usage:
    python mlb_breaking_ball_location.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from xgboost import XGBClassifier

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_kbb_model import (FOLDS, K_FEATURES_SWSTR, MIN_PA, PANEL_PATH, _zscore_apply,
                            _zscore_fit, build_model_frame)

pd.set_option("display.width", 160, "display.max_columns", 40)

def analyze_family(raw: pd.DataFrame, panel: pd.DataFrame, family: str):
    fl = family.lower()
    fam = raw[raw["pitch_family"] == family].copy()
    print(f"\n{'#'*70}\n#  FAMILY = {family}  ({len(fam):,} pitches)\n{'#'*70}")

    # ---- 1 & 2: raw location metrics ---- #
    zone_center_z = (fam["sz_top"] + fam["sz_bot"]) / 2
    fam["horiz_dist_raw"] = fam["plate_x"].abs()
    fam["vert_dist_raw"] = (fam["plate_z"] - zone_center_z).abs()
    fam["buried"] = fam["plate_z"] < fam["sz_bot"]

    lo_h, hi_h = fam["horiz_dist_raw"].quantile([.01, .99])
    lo_v, hi_v = fam["vert_dist_raw"].quantile([.01, .99])
    fam["horiz_dist_w"] = fam["horiz_dist_raw"].clip(lo_h, hi_h)
    fam["vert_dist_w"] = fam["vert_dist_raw"].clip(lo_v, hi_v)
    fam_swing = fam[fam["is_swing"]]
    whiff_by_season = fam_swing.groupby(["pitcher", "game_year"])["is_whiff"].mean().rename(f"whiff_{fl}")

    season_loc = fam.groupby(["pitcher", "game_year"]).agg(
        **{f"horiz_dist_{fl}": ("horiz_dist_w", "mean"), f"vert_dist_{fl}": ("vert_dist_w", "mean"),
           f"buried_rate_{fl}": ("buried", "mean"), f"n_{fl}": ("buried", "size")}
    ).reset_index().merge(whiff_by_season, on=["pitcher", "game_year"], how="left")
    season_loc = season_loc[season_loc[f"n_{fl}"] >= 100]

    d = panel.merge(season_loc, on=["pitcher", "game_year"], how="inner")
    elig = d[d["PA"] >= MIN_PA]
    print(f"\nn eligible pitcher-seasons with >=100 {family} pitches: {len(elig)}")
    print(f"corr(horiz_dist_{fl}, K_rate) = {elig[f'horiz_dist_{fl}'].corr(elig['K_rate']):.4f}")
    print(f"corr(vert_dist_{fl}, K_rate) = {elig[f'vert_dist_{fl}'].corr(elig['K_rate']):.4f}")
    print(f"corr(buried_rate_{fl}, K_rate) = {elig[f'buried_rate_{fl}'].corr(elig['K_rate']):.4f}")
    print(f"corr(buried_rate_{fl}, {family} whiff rate) = {elig[f'buried_rate_{fl}'].corr(elig[f'whiff_{fl}']):.4f}")

    sva_col = f"wh_sva_{fl}"
    print(f"\noverlap check -- corr(horiz_dist_{fl}, {sva_col}) = {elig[f'horiz_dist_{fl}'].corr(elig[sva_col]):.4f}")
    print(f"overlap check -- corr(vert_dist_{fl}, {sva_col}) = {elig[f'vert_dist_{fl}'].corr(elig[sva_col]):.4f}")
    print(f"overlap check -- corr(buried_rate_{fl}, {sva_col}) = {elig[f'buried_rate_{fl}'].corr(elig[sva_col]):.4f}")

    # ---- 3: location-ONLY two-stage OOF score ---- #
    print(f"\nFitting {family} location-only two-stage model (swing-conditioned)...")
    swing_pop = fam[fam["is_swing"]].dropna(subset=mf.LOC_COLS + ["pitcher", "is_whiff"]).copy()
    y = swing_pop["is_whiff"].astype(int).values
    groups = swing_pop["pitcher"].values
    gkf = GroupKFold(n_splits=5)
    oof_loc = cross_val_predict(XGBClassifier(**mf.XGB_PARAMS), swing_pop[mf.LOC_COLS].astype(float).values,
                                 y, cv=gkf, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    swing_pop["loc_only_whiff_prob"] = oof_loc
    loc_season = swing_pop.groupby(["pitcher", "game_year"]).agg(
        **{f"loc_value_{fl}": ("loc_only_whiff_prob", "mean"), f"n_sw_{fl}": ("loc_only_whiff_prob", "size")}
    ).reset_index()
    loc_season = loc_season[loc_season[f"n_sw_{fl}"] >= 50]

    d2 = elig.merge(loc_season, on=["pitcher", "game_year"], how="inner")
    print(f"\nn = {len(d2)}")
    print(f"corr(loc_value_{fl} [location-only P(whiff)], K_rate) = {d2[f'loc_value_{fl}'].corr(d2['K_rate']):.4f}")
    print(f"corr(loc_value_{fl}, {sva_col} [the shape RESIDUAL]) = {d2[f'loc_value_{fl}'].corr(d2[sva_col]):.4f}  "
          f"(should be near-zero by construction -- shape residual is defined orthogonal to the location stage)")

    return season_loc.merge(loc_season, on=["pitcher", "game_year"], how="left")


def backtest_with_location(panel, season_loc, family, feature_cols_extra_sets):
    fl = family.lower()
    mframe = build_model_frame(panel)
    loc_feat = season_loc.rename(columns={"game_year": "target_year_feat"})
    loc_feat["target_year"] = loc_feat["target_year_feat"] + 1
    cols = [f"horiz_dist_{fl}", f"vert_dist_{fl}", f"buried_rate_{fl}", f"loc_value_{fl}"]
    cols = [c for c in cols if c in loc_feat.columns]
    mframe = mframe.merge(loc_feat[["pitcher", "target_year"] + cols], on=["pitcher", "target_year"], how="left")
    for c in cols:
        mframe[c] = mframe[c].fillna(mframe[c].median())

    def fit_eval(train, test, feature_cols, n0):
        dd = train.dropna(subset=["Marcel_K_rate", "K_rate", "PA"]).copy()
        dd["y"] = dd["K_rate"] * 100
        dd["base"] = dd["Marcel_K_rate"] * 100
        dd["wt"] = mm.reliability_weight(dd["PA"], n0)
        stats = _zscore_fit(dd, feature_cols)
        dd = _zscore_apply(dd, feature_cols, stats)
        zcols = [f"{c}_z" for c in feature_cols]
        import statsmodels.formula.api as smf
        formula = "y ~ base + age_c + age_decline + " + " + ".join(zcols)
        model = smf.wls(formula, data=dd, weights=dd["wt"]).fit()
        dt = test.dropna(subset=["Marcel_K_rate", "K_rate", "PA"]).copy()
        dt["base"] = dt["Marcel_K_rate"] * 100
        dt = _zscore_apply(dt, feature_cols, stats)
        pred = model.predict(dt).values / 100
        return mm.evaluate_weighted(dt["K_rate"].values, pred, dt["PA"].values, n0, label="")

    print(f"\n=== Walk-forward K% backtest: does {family} location add anything? ===")
    for label, extra in feature_cols_extra_sets:
        rows = []
        for train_years, test_year in FOLDS:
            train = mframe[mframe["target_year"].isin(train_years)]
            test = mframe[mframe["target_year"] == test_year]
            elig_train = train[train["PA"] >= MIN_PA]
            n0_k, _ = mm.estimate_n0(elig_train["K_rate"], elig_train["PA"])
            rows.append(fit_eval(train, test, K_FEATURES_SWSTR + extra, n0_k))
        bt = pd.DataFrame(rows)
        print(f"  {label:52s}  R2_w={bt['R2_w'].mean():.4f}  RMSE_w={bt['RMSE_w_pp'].mean():.3f}pp")


if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)
    panel = pd.read_parquet(PANEL_PATH)

    season_loc_brk = analyze_family(raw, panel, "BRK")
    season_loc_off = analyze_family(raw, panel, "OFF")

    backtest_with_location(panel, season_loc_brk, "BRK", [
        ("baseline (current swstr_sva model)", []),
        ("+ vert_dist_brk", ["vert_dist_brk"]),
        ("+ horiz_dist_brk + vert_dist_brk + buried_rate_brk",
         ["horiz_dist_brk", "vert_dist_brk", "buried_rate_brk"]),
        ("+ loc_value_brk (modeled location-only score)", ["loc_value_brk"]),
    ])
    backtest_with_location(panel, season_loc_off, "OFF", [
        ("baseline (current swstr_sva model)", []),
        ("+ vert_dist_off", ["vert_dist_off"]),
        ("+ horiz_dist_off + vert_dist_off + buried_rate_off",
         ["horiz_dist_off", "vert_dist_off", "buried_rate_off"]),
        ("+ loc_value_off (modeled location-only score)", ["loc_value_off"]),
    ])
