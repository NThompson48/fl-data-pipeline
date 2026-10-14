"""
mlb_kbb_project_2026_v2.py -- final production K%/BB% projection, extending
mlb_kbb_project_2026.py with the validated AA/AAA MiLB bridge: for pitchers
whose trailing-MLB Marcel reliability is thin (<0.5) or absent entirely (true
rookies with a qualifying 2025 AA/AAA season), the Marcel_K_rate/BB_rate INPUT
to the already-validated K%/BB% regression is replaced with the MiLB-blended
combo estimate -- everything downstream (age, arsenal-shape features, the
fitted regression coefficients) is unchanged from the original model.

Usage:
    python mlb_kbb_project_2026_v2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mlb_marcel as mm
import mlb_milb_bridge as mbb
from mlb_kbb_model import (BB_FEATURES, K_FEATURES_SWSTR, MIN_PA, PANEL_PATH,
                            _zscore_apply, _zscore_fit, build_model_frame)

pd.set_option("display.width", 200, "display.max_columns", 40)
TARGET_YEAR = 2026


def fit_full(mframe, rate, feature_cols, n0):
    d = mframe.dropna(subset=[f"Marcel_{rate}_rate", f"{rate}_rate", "PA"]).copy()
    d["y"] = d[f"{rate}_rate"] * 100
    d["base"] = d[f"Marcel_{rate}_rate"] * 100
    d["wt"] = mm.reliability_weight(d["PA"], n0)
    stats = _zscore_fit(d, feature_cols)
    d = _zscore_apply(d, feature_cols, stats)
    zcols = [f"{c}_z" for c in feature_cols]
    formula = "y ~ base + age_c + age_decline + " + " + ".join(zcols)
    model = smf.wls(formula, data=d, weights=d["wt"]).fit()
    fitted = model.fittedvalues.values / 100
    resid = d["y"].values / 100 - fitted
    wrmse = np.sqrt(np.average(resid ** 2, weights=d["wt"].values))
    return model, stats, d, fitted, resid, wrmse


if __name__ == "__main__":
    panel = pd.read_parquet(PANEL_PATH)
    mframe = build_model_frame(panel)
    elig_hist = mframe[(mframe["PA"] >= MIN_PA) & (mframe["target_year"] <= 2025)]

    n0_k, _ = mm.estimate_n0(elig_hist["K_rate"], elig_hist["PA"])
    n0_bb, _ = mm.estimate_n0(elig_hist["BB_rate"], elig_hist["PA"])

    # the validated regression itself is trained EXACTLY as before, on established
    # pitchers with real Marcel history -- the MiLB blend only changes what gets fed
    # into "base" for thin/rookie pitchers at PROJECTION time, not how the model is fit
    mk, sk, dk, fit_k, resid_k, wrmse_k = fit_full(elig_hist, "K", K_FEATURES_SWSTR, n0_k)
    mbb_model, sbb, dbb, fit_bb, resid_bb, wrmse_bb = fit_full(elig_hist, "BB", BB_FEATURES, n0_bb)

    both = dk[["pitcher", "target_year"]].assign(resid_k=resid_k)
    bb_r = dbb[["pitcher", "target_year"]].assign(resid_bb=resid_bb)
    matched = both.merge(bb_r, on=["pitcher", "target_year"])
    rho = matched["resid_k"].corr(matched["resid_bb"])
    print(f"measured residual correlation corr(K resid, BB resid) = {rho:.3f}  (n={len(matched)})")

    # ---- build 2026 target rows: established pitchers (unchanged) + MiLB-enhanced ---- #
    feat_2025 = panel[panel["game_year"] == 2025].copy()
    season_all = panel[["pitcher", "game_year", "PA", "K", "BB", "K_rate", "BB_rate"]].copy()

    pivot_pa = panel.pivot(index="pitcher", columns="game_year", values="PA")
    pivot_k = panel.pivot(index="pitcher", columns="game_year", values="K")
    pivot_bb = panel.pivot(index="pitcher", columns="game_year", values="BB")
    league = panel.groupby("game_year").apply(
        lambda g: pd.Series({"K_rate": g["K"].sum() / g["PA"].sum(), "BB_rate": g["BB"].sum() / g["PA"].sum()}))
    reg_k_full, _ = mm.estimate_n0(panel[panel["PA"] >= MIN_PA]["K_rate"], panel[panel["PA"] >= MIN_PA]["PA"])
    reg_bb_full, _ = mm.estimate_n0(panel[panel["PA"] >= MIN_PA]["BB_rate"], panel[panel["PA"] >= MIN_PA]["PA"])
    trailing = [2025, 2024, 2023]
    weights = [mm.MARCEL_WEIGHTS[lag] for lag in (0, 1, 2)]
    pitchers = pivot_pa.index.values
    num_k = np.zeros(len(pitchers)); num_bb = np.zeros(len(pitchers)); den = np.zeros(len(pitchers))
    lg_num_k, lg_num_bb, lg_den = 0.0, 0.0, 0.0
    for wt, yr in zip(weights, trailing):
        if yr not in pivot_pa.columns:
            continue
        pa = pivot_pa[yr].fillna(0).values; k = pivot_k[yr].fillna(0).values; bb = pivot_bb[yr].fillna(0).values
        num_k += wt * k; num_bb += wt * bb; den += wt * pa
        lg_num_k += wt * league.loc[yr, "K_rate"]; lg_num_bb += wt * league.loc[yr, "BB_rate"]; lg_den += wt
    lg_k_blend, lg_bb_blend = lg_num_k / lg_den, lg_num_bb / lg_den
    with np.errstate(divide="ignore", invalid="ignore"):
        marcel_k_2026 = (num_k + reg_k_full * lg_k_blend) / (den + reg_k_full)
        marcel_bb_2026 = (num_bb + reg_bb_full * lg_bb_blend) / (den + reg_bb_full)
    mask = den > 0
    marcel_established = pd.DataFrame({"pitcher": pitchers[mask], "Marcel_K_rate": marcel_k_2026[mask],
                                        "Marcel_BB_rate": marcel_bb_2026[mask], "Marcel_weighted_PA": den[mask]})

    # ---- MiLB blend: thin-reliability existing pitchers + true rookies ---- #
    print("\nFitting MiLB bridge coefficients (production: all years)...")
    coefs = mbb.fit_all_production()
    marcel_established["marcel_reliability"] = mm.reliability_weight(
        marcel_established["Marcel_weighted_PA"], n0_k)
    milb_lookup_prior = mbb.get_milb_lookup(TARGET_YEAR, "prior")  # 2025 AA/AAA -> 2026
    merged = marcel_established.merge(milb_lookup_prior, on="pitcher", how="left")
    thin = merged["marcel_reliability"] < 0.5
    has_milb = merged["milb_K_rate"].notna()
    blend_mask = thin & has_milb
    pk, pbb = mbb.apply_combo(merged["Marcel_K_rate"], merged["Marcel_BB_rate"],
                               merged["milb_K_rate"], merged["milb_BB_rate"], coefs["prior"])
    merged["Marcel_K_rate"] = np.where(blend_mask, pk, merged["Marcel_K_rate"])
    merged["Marcel_BB_rate"] = np.where(blend_mask, pbb, merged["Marcel_BB_rate"])
    merged["milb_blended"] = blend_mask
    print(f"thin-reliability pitchers blended with MiLB: {blend_mask.sum()}")

    # true rookies: qualifying 2025 AA/AAA, zero MLB history at all
    existing_ids = set(marcel_established["pitcher"])
    rookies = milb_lookup_prior[~milb_lookup_prior["pitcher"].isin(existing_ids)].copy()
    lg_k_2025, lg_bb_2025 = league.loc[2025, "K_rate"], league.loc[2025, "BB_rate"]
    pk_r, pbb_r = mbb.apply_combo(lg_k_2025, lg_bb_2025, rookies["milb_K_rate"], rookies["milb_BB_rate"], coefs["prior"])
    rookies["Marcel_K_rate"] = pk_r
    rookies["Marcel_BB_rate"] = pbb_r
    rookies["Marcel_weighted_PA"] = 0.0
    rookies["milb_blended"] = True
    print(f"true rookies added (no prior MLB history, qualifying 2025 AA/AAA): {len(rookies)}")

    proj = pd.concat([merged.drop(columns=["milb_K_rate", "milb_BB_rate", "milb_BF", "level", "marcel_reliability"]),
                       rookies.drop(columns=["milb_K_rate", "milb_BB_rate", "milb_BF", "level"])], ignore_index=True)

    bio = pd.read_parquet("output/pitcher_bio.parquet")[["pitcher_id", "birth_date"]].rename(
        columns={"pitcher_id": "pitcher"})
    proj = proj.merge(bio, on="pitcher", how="left")
    ref_date = pd.Timestamp(f"{TARGET_YEAR}-04-01")
    proj["age"] = (ref_date - proj["birth_date"]).dt.days / 365.25
    proj["age_c"] = proj["age"] - 29
    proj["age_decline"] = (proj["age"] - 30).clip(lower=0)

    lagged_cols_k = ["swstr_sva_fb", "swstr_sva_brk", "swstr_sva_off", "wh_sva_delta", "BF_per_G"]
    lagged_cols_bb = ["horiz_dist", "vert_dist", "horiz_dist_delta", "vert_dist_delta"]
    lag_cols = sorted(set(lagged_cols_k + lagged_cols_bb))
    proj = proj.merge(feat_2025[["pitcher"] + lag_cols], on="pitcher", how="left")
    for c in lag_cols:
        proj[c] = proj[c].fillna(0.0)

    proj["base"] = proj["Marcel_K_rate"] * 100
    proj = _zscore_apply(proj, K_FEATURES_SWSTR, sk)
    proj["K_2026_hat"] = mk.predict(proj) / 100
    proj["base"] = proj["Marcel_BB_rate"] * 100
    proj = _zscore_apply(proj, BB_FEATURES, sbb)
    proj["BB_2026_hat"] = mbb_model.predict(proj) / 100
    proj["KBB_2026_hat"] = proj["K_2026_hat"] - proj["BB_2026_hat"]

    season_2025 = feat_2025[["pitcher", "PA", "K_rate", "BB_rate"]].rename(
        columns={"PA": "PA_2025", "K_rate": "K_2025", "BB_rate": "BB_2025"})
    proj = proj.merge(season_2025, on="pitcher", how="left")
    proj["PA_proxy"] = proj["PA_2025"].fillna(proj["Marcel_weighted_PA"]).fillna(0).clip(lower=30)
    rel_k = mm.reliability_weight(proj["PA_proxy"], n0_k)
    rel_bb = mm.reliability_weight(proj["PA_proxy"], n0_bb)
    proj["SE_K"] = wrmse_k / np.sqrt(rel_k.clip(lower=0.05))
    proj["SE_BB"] = wrmse_bb / np.sqrt(rel_bb.clip(lower=0.05))
    proj["SE_KBB"] = mm.propagate_kbb_uncertainty(proj["SE_K"], proj["SE_BB"], rho)
    # rookies carry extra translation uncertainty beyond the pure-MLB SE formula --
    # flag them so this is visible rather than silently understated
    proj["is_rookie_milb_projection"] = proj["milb_blended"] & (proj["Marcel_weighted_PA"].fillna(0) == 0)

    id_map = pd.read_parquet("output/pitcher_id_name_map.parquet").rename(columns={"pitcher_id": "pitcher"})
    proj = proj.merge(id_map, on="pitcher", how="left")

    out = proj[["pitcher", "name", "age", "milb_blended", "is_rookie_milb_projection", "PA_2025", "K_2025",
                "K_2026_hat", "SE_K", "BB_2025", "BB_2026_hat", "SE_BB", "KBB_2026_hat", "SE_KBB"]]
    out = out.sort_values("KBB_2026_hat", ascending=False)
    out.to_csv("output/kbb_projection_2026_v2.csv", index=False)
    print(f"\nSaved -> output/kbb_projection_2026_v2.csv, {len(out)} pitchers "
          f"(was 655 in the original v1 -- {len(out)-655} more, mostly new rookies)")

    print("\n-- Top 10 rookies (never had trailing MLB history) --")
    rookies_out = out[out["is_rookie_milb_projection"]].sort_values("KBB_2026_hat", ascending=False)
    print(rookies_out.head(10).round(4).to_string(index=False))

    print("\n-- Sanity check: established stars unchanged from v1? --")
    for nm in ["Skenes", "Skubal", "Wheeler"]:
        row = out[out["name"].str.contains(nm, case=False, na=False)]
        if len(row):
            print(row.round(4).to_string(index=False, header=False))
