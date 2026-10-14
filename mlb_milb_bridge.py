"""
mlb_milb_bridge.py -- AA/AAA -> MLB K%/BB% translation and blending, built from
the validated bridge analysis (mlb_milb_bridge_aa_aaa.py). Two bridges:
  - "prior":     AA/AAA year Y  -> MLB year Y+1   (preseason: thin-MLB-history pitchers)
  - "same_year": AA/AAA year Y  -> MLB year Y     (in-season: post-callup ROS projection,
                  valid because the MiLB stats endpoint and the MLB Statcast panel each
                  only cover time spent at that level -- no date-splitting needed)

Both fits are done with a train/test split (years <=2023 fit, 2024-2025 held out)
so the reported combo weights are validated out-of-sample, not just in-sample --
matching this project's standing practice of not trusting a small-sample fit
without a holdout check.

Usage:
    import mlb_milb_bridge as mb
    coefs = mb.fit_all()
    enhanced = mb.enhance_marcel_with_milb(season, marcel, coefs)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mlb_marcel as mm

MILB_PATH = "output/milb_pitching_2018_2025.parquet"
MLB_PANEL_PATH = "output/mlb_pitcher_season_features.parquet"
LEVELS = ["AAA", "AA"]
MIN_MILB_BF = 100
MIN_MLB_PA = 30
TRAIN_MAX_YEAR = 2023  # fit years <=2023, hold out 2024-2025 for validation


def load_milb() -> pd.DataFrame:
    milb = pd.read_parquet(MILB_PATH)
    milb = milb.sort_values("battersFaced", ascending=False).drop_duplicates(
        subset=["pitcher", "season"], keep="first")
    return milb[(milb["battersFaced"] >= MIN_MILB_BF) & milb["level"].isin(LEVELS)]


def _build_bridge(milb: pd.DataFrame, mlb: pd.DataFrame, year_offset: int) -> pd.DataFrame:
    bridge = milb.merge(mlb.rename(columns={"game_year": "season_mlb"}), on="pitcher", how="inner")
    bridge = bridge[bridge["season_mlb"] == bridge["season"] + year_offset]
    return bridge[bridge["PA"] >= MIN_MLB_PA]


def fit_bridge(milb: pd.DataFrame, mlb: pd.DataFrame, year_offset: int, label: str) -> dict:
    """Fits the translation (MiLB rate -> MLB rate) AND the combo blend
    (Marcel_rate + translated MiLB rate -> actual MLB rate), on TRAIN years
    only, then reports held-out R2 on TEST years as an honest validation
    check before trusting the coefficients for production use."""
    bridge = _build_bridge(milb, mlb, year_offset)
    train = bridge[bridge["season"] <= TRAIN_MAX_YEAR].dropna(subset=["Marcel_K_rate", "Marcel_BB_rate"])
    test = bridge[bridge["season"] > TRAIN_MAX_YEAR].dropna(subset=["Marcel_K_rate", "Marcel_BB_rate"])
    print(f"\n[{label}] train n={len(train)}, test n={len(test)}")
    if len(train) < 100 or len(test) < 30:
        print(f"  WARNING: sample too small for a trustworthy train/test split -- using full sample, "
              f"treat coefficients as provisional")
        train = bridge.dropna(subset=["Marcel_K_rate", "Marcel_BB_rate"])
        test = train

    out = {}
    for rate, milb_col, marcel_col in [("K", "K_rate_x", "Marcel_K_rate"), ("BB", "BB_rate_x", "Marcel_BB_rate")]:
        trans = smf.ols(f"K_rate_y ~ {milb_col}" if rate == "K" else f"BB_rate_y ~ {milb_col}", data=train).fit()
        trans_intercept, trans_slope = trans.params.iloc[0], trans.params.iloc[1]

        combo = smf.ols(f"{'K' if rate=='K' else 'BB'}_rate_y ~ {marcel_col} + {milb_col}", data=train).fit()
        n0, _ = mm.estimate_n0(train[f"{rate}_rate_y"], train["PA"], min_n=10)
        w_test = mm.reliability_weight(test["PA"], n0).values
        pred_marcel = test[marcel_col].values
        pred_combo = combo.predict(test).values
        r2_marcel = mm.weighted_r2(test[f"{rate}_rate_y"].values, pred_marcel, w_test)
        r2_combo = mm.weighted_r2(test[f"{rate}_rate_y"].values, pred_combo, w_test)
        print(f"  {rate}%: translation MLB = {trans_intercept*100:.2f}pp + {trans_slope:.3f}*MiLB  |  "
              f"held-out R2_w: Marcel alone={r2_marcel:.4f}, combo={r2_combo:.4f}")
        out[rate] = {"trans_intercept": trans_intercept, "trans_slope": trans_slope,
                      "combo_intercept": combo.params.iloc[0], "combo_marcel_coef": combo.params.iloc[1],
                      "combo_milb_coef": combo.params.iloc[2], "n0": n0,
                      "holdout_r2_marcel": r2_marcel, "holdout_r2_combo": r2_combo}
    return out


def fit_all() -> dict:
    milb = load_milb()
    mlb = pd.read_parquet(MLB_PANEL_PATH)[
        ["pitcher", "game_year", "PA", "K_rate", "BB_rate", "Marcel_K_rate", "Marcel_BB_rate"]]
    return {"prior": fit_bridge(milb, mlb, 1, "YEAR-PRIOR"),
            "same_year": fit_bridge(milb, mlb, 0, "SAME-YEAR")}


def fit_bridge_production(milb: pd.DataFrame, mlb: pd.DataFrame, year_offset: int) -> dict:
    """Refits on ALL years (no held-out split) for the final deployed
    coefficients -- only after fit_bridge()'s train/test split has already
    confirmed the methodology holds up out-of-sample (same validate-then-
    refit-on-everything pattern as the main K%/BB% production model)."""
    bridge = _build_bridge(milb, mlb, year_offset).dropna(subset=["Marcel_K_rate", "Marcel_BB_rate"])
    out = {}
    for rate, milb_col, marcel_col in [("K", "K_rate_x", "Marcel_K_rate"), ("BB", "BB_rate_x", "Marcel_BB_rate")]:
        combo = smf.ols(f"{'K' if rate=='K' else 'BB'}_rate_y ~ {marcel_col} + {milb_col}", data=bridge).fit()
        out[rate] = {"combo_intercept": combo.params.iloc[0], "combo_marcel_coef": combo.params.iloc[1],
                      "combo_milb_coef": combo.params.iloc[2]}
    return out


def fit_all_production() -> dict:
    milb = load_milb()
    mlb = pd.read_parquet(MLB_PANEL_PATH)[
        ["pitcher", "game_year", "PA", "K_rate", "BB_rate", "Marcel_K_rate", "Marcel_BB_rate"]]
    return {"prior": fit_bridge_production(milb, mlb, 1), "same_year": fit_bridge_production(milb, mlb, 0)}


def get_milb_lookup(target_year: int, bridge_type: str) -> pd.DataFrame:
    """AA/AAA rate for the relevant MiLB season: target_year-1 for 'prior',
    target_year itself for 'same_year'."""
    milb = load_milb()
    milb_year = target_year - 1 if bridge_type == "prior" else target_year
    sub = milb[milb["season"] == milb_year][["pitcher", "K_rate", "BB_rate", "battersFaced", "level"]]
    return sub.rename(columns={"K_rate": "milb_K_rate", "BB_rate": "milb_BB_rate", "battersFaced": "milb_BF"})


def apply_combo(marcel_k, marcel_bb, milb_k, milb_bb, coefs: dict) -> tuple[np.ndarray, np.ndarray]:
    ck, cbb = coefs["K"], coefs["BB"]
    proj_k = ck["combo_intercept"] + ck["combo_marcel_coef"] * marcel_k + ck["combo_milb_coef"] * milb_k
    proj_bb = cbb["combo_intercept"] + cbb["combo_marcel_coef"] * marcel_bb + cbb["combo_milb_coef"] * milb_bb
    return proj_k, proj_bb


def enhance_marcel_with_milb(season: pd.DataFrame, marcel: pd.DataFrame, coefs: dict,
                              thin_reliability: float = 0.5) -> pd.DataFrame:
    """For every target_year in `marcel`, blends in a translated AA/AAA year-prior
    rate for pitchers whose PURE-MLB Marcel reliability (weighted PA vs. n0) is
    below `thin_reliability` -- i.e., Marcel's own trailing-MLB signal is weak
    enough that MiLB information is worth incorporating. ALSO adds rows for
    pitchers with qualifying MiLB history but NO trailing MLB Marcel row at all
    (previously invisible to the panel entirely)."""
    n0_k, _ = mm.estimate_n0(season[season["PA"] >= 50]["K_rate"], season[season["PA"] >= 50]["PA"])
    marcel = marcel.copy()
    marcel["marcel_reliability"] = mm.reliability_weight(marcel["Marcel_weighted_PA"], n0_k)

    out_rows = []
    for target_year in sorted(marcel["game_year"].unique()):
        milb_lookup = get_milb_lookup(target_year, "prior")
        yr = marcel[marcel["game_year"] == target_year]
        merged = yr.merge(milb_lookup, on="pitcher", how="left")
        thin = merged["marcel_reliability"] < thin_reliability
        has_milb = merged["milb_K_rate"].notna()

        blend_mask = thin & has_milb
        pk, pbb = apply_combo(merged["Marcel_K_rate"], merged["Marcel_BB_rate"],
                               merged["milb_K_rate"], merged["milb_BB_rate"], coefs)
        merged["Marcel_K_rate_v2"] = np.where(blend_mask, pk, merged["Marcel_K_rate"])
        merged["Marcel_BB_rate_v2"] = np.where(blend_mask, pbb, merged["Marcel_BB_rate"])
        merged["milb_blended"] = blend_mask
        out_rows.append(merged)

        # pitchers with qualifying MiLB but NO Marcel row at all this target_year (true rookies)
        existing_pitchers = set(yr["pitcher"])
        new_rookies = milb_lookup[~milb_lookup["pitcher"].isin(existing_pitchers)].copy()
        if len(new_rookies):
            new_rookies["game_year"] = target_year
            lg_k = season[season["game_year"] == target_year - 1]["K_rate"].mean()
            lg_bb = season[season["game_year"] == target_year - 1]["BB_rate"].mean()
            pk_r, pbb_r = apply_combo(lg_k, lg_bb, new_rookies["milb_K_rate"], new_rookies["milb_BB_rate"], coefs)
            new_rookies["Marcel_K_rate_v2"] = pk_r
            new_rookies["Marcel_BB_rate_v2"] = pbb_r
            new_rookies["Marcel_K_rate"] = np.nan
            new_rookies["Marcel_BB_rate"] = np.nan
            new_rookies["Marcel_weighted_PA"] = 0.0
            new_rookies["milb_blended"] = True
            out_rows.append(new_rookies)

    return pd.concat(out_rows, ignore_index=True)


if __name__ == "__main__":
    coefs = fit_all()
    print("\n\nFitted coefficients:")
    import json
    print(json.dumps(coefs, indent=2, default=float))
