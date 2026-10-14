"""
mlb_count_leverage_command.py -- count-leverage-conditioned "Command" score,
inspired by the OpenCommand/"Simple Command" framework (Foolish Bailey video,
per-pitch-type Good/Great location zones split by ahead/even/behind count).

Instead of hand-drawn zone boundaries, this reuses the project's established
two-stage architecture: fit P(whiff|swing) from LOCATION alone, once with just
(plate_x_as, plate_z), once adding (balls, strikes) -- if count genuinely
changes the optimal location (not just whiff rate overall), the model WITH
count should beat location-only on held-out AUC. The count-aware OOF score is
then this project's "Command" value (cmd_value), aggregated per pitcher-season,
and checked for:
  1. Does count add real signal beyond location alone? (AUC comparison)
  2. Descriptive: whiff rate / location by ahead-even-behind bucket, per family
  3. Relationship with K_rate and BB_rate (concurrent)
  4. Stickiness: year-over-year correlation (couplets, matching build_couplets)
  5. Stability: n0 via both estimate_n0 (method-of-moments) and the empirical
     YoY reliability curve -- same units as every other stat in this project,
     directly comparable to K% (n0=73.7) and BB% (n0=160.4)

Usage:
    python mlb_count_leverage_command.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from xgboost import XGBClassifier

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_kbb_model import MIN_PA, PANEL_PATH

pd.set_option("display.width", 160, "display.max_columns", 40)

LOC_COLS = ["plate_x_as", "plate_z"]
LOC_COUNT_COLS = LOC_COLS + ["balls", "strikes"]
FAMILIES = ["FB", "BRK", "OFF"]


def leverage_bucket(balls, strikes):
    return np.where(strikes > balls, "ahead", np.where(balls > strikes, "behind", "even"))


def fit_family(pop: pd.DataFrame, family: str):
    d = pop[(pop["pitch_family"] == family) & pop["is_swing"]].dropna(
        subset=LOC_COUNT_COLS + ["pitcher", "is_whiff"]).copy()
    y = d["is_whiff"].astype(int).values
    groups = d["pitcher"].values
    gkf = GroupKFold(n_splits=5)

    oof_loc = cross_val_predict(XGBClassifier(**mf.XGB_PARAMS), d[LOC_COLS].astype(float).values,
                                 y, cv=gkf, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    oof_locct = cross_val_predict(XGBClassifier(**mf.XGB_PARAMS), d[LOC_COUNT_COLS].astype(float).values,
                                   y, cv=gkf, groups=groups, method="predict_proba", n_jobs=5)[:, 1]
    auc_loc, auc_locct = roc_auc_score(y, oof_loc), roc_auc_score(y, oof_locct)
    print(f"  [{family}] n={len(d):,}  location-only AUC={auc_loc:.4f}  "
          f"location+count AUC={auc_locct:.4f}  delta={auc_locct-auc_loc:+.4f}")

    d["cmd_value"] = oof_locct
    d["leverage"] = leverage_bucket(d["balls"].values, d["strikes"].values)
    desc = d.groupby("leverage").agg(n=("is_whiff", "size"), whiff_rate=("is_whiff", "mean"),
                                       mean_plate_x_as=("plate_x_as", "mean"), mean_plate_z=("plate_z", "mean"))
    print(f"    by leverage:\n{desc.round(4).to_string()}")

    season = d.groupby(["pitcher", "game_year"]).agg(
        **{f"cmd_value_{family.lower()}": ("cmd_value", "mean"), f"n_cmd_{family.lower()}": ("cmd_value", "size")}
    ).reset_index()
    return season


if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)

    seasons = []
    for family in FAMILIES:
        print(f"\n{'='*70}\n{family}\n{'='*70}")
        seasons.append(fit_family(raw, family))

    cmd = seasons[0]
    for s in seasons[1:]:
        cmd = cmd.merge(s, on=["pitcher", "game_year"], how="outer")

    n_cols = [c for c in cmd.columns if c.startswith("n_cmd_")]
    val_cols = [c for c in cmd.columns if c.startswith("cmd_value_")]
    # blended overall command score, PA-weighted across the 3 families (only counting
    # families with enough pitches, mirroring the family-sva imputation convention)
    for vc, nc in zip(val_cols, n_cols):
        cmd.loc[cmd[nc].fillna(0) < mf.MIN_N_FAMILY, vc] = np.nan
    cmd["cmd_value_overall"] = cmd[val_cols].mean(axis=1, skipna=True)
    cmd["n_cmd_overall"] = cmd[n_cols].fillna(0).sum(axis=1)

    panel = pd.read_parquet(PANEL_PATH)
    d = panel.merge(cmd, on=["pitcher", "game_year"], how="inner")
    elig = d[(d["PA"] >= MIN_PA) & d["cmd_value_overall"].notna()]
    print(f"\nn eligible pitcher-seasons with an overall command score: {len(elig)}")

    print("\n=== Relationship with K_rate and BB_rate (concurrent) ===")
    for col in val_cols + ["cmd_value_overall"]:
        ck = elig[col].corr(elig["K_rate"])
        cb = elig[col].corr(elig["BB_rate"])
        print(f"  {col:22s}  corr(K_rate)={ck:+.4f}   corr(BB_rate)={cb:+.4f}")

    print("\n=== Stickiness: year-over-year correlation ===")
    b = elig[["pitcher", "game_year", "cmd_value_overall", "n_cmd_overall"]].copy()
    b["game_year"] -= 1
    cp = elig.merge(b, on=["pitcher", "game_year"], suffixes=("1", "2"))
    print(f"  n couplets = {len(cp)}")
    print(f"  raw corr(cmd_value_overall_Y, cmd_value_overall_Y+1) = "
          f"{cp['cmd_value_overall1'].corr(cp['cmd_value_overall2']):.4f}")

    print("\n=== Stability: n0 (method-of-moments + empirical YoY curve) ===")
    # cmd_value_overall is a whiff-probability-like score already in [0,1], so
    # estimate_n0's p(1-p) sampling-variance assumption applies directly
    n0_mm, info = mm.estimate_n0(elig["cmd_value_overall"], elig["n_cmd_overall"], min_n=30)
    print(f"  n0 (method-of-moments) = {n0_mm:.1f}   {info}")

    curve = mm.empirical_reliability_curve(cp, "cmd_value_overall1", "cmd_value_overall2",
                                             "n_cmd_overall1", "n_cmd_overall2")
    print(f"\n  empirical YoY reliability curve:\n{curve.round(4).to_string(index=False)}")
    n0_yoy = mm.fit_n0_from_curve(curve)
    print(f"  n0 (fit to YoY curve) = {n0_yoy:.1f}")
    print(f"\n  for comparison, this project's other n0 values: K% n0=73.7, BB% n0=160.4")

    cmd.to_parquet("output/count_leverage_command.parquet", index=False)
    print("\nSaved -> output/count_leverage_command.parquet")
