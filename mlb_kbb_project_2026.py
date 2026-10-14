"""
mlb_kbb_project_2026.py -- final production K%/BB% model, fit on ALL validated
history (2019-2025 target-year rows), used to project 2026. The walk-forward
backtest in mlb_kbb_walkforward.py already validated this architecture
out-of-sample; this script trains the strongest single fit on everything for
the actual deployed projection, per k26_projection.py's own project_2026()
precedent (validate out-of-sample first, then refit on everything for the
final forecast).

Uses the unconditional (SwStr%-SVA) family features for K%, since it won (by
a narrow but consistent margin) in the walk-forward backtest.

Per-pitcher K-BB% uncertainty: SE_K/SE_BB scaled by each pitcher's own
reliability (using 2025 PA as the proxy for 2026 playing time -- the true
2026 PA is unknown at projection time), combined via
mlb_marcel.propagate_kbb_uncertainty using the EMPIRICALLY measured residual
correlation between the two fitted models (never assumed independent).

Usage:
    python mlb_kbb_project_2026.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlb_marcel as mm
from mlb_kbb_model import (BB_FEATURES, K_FEATURES_SWSTR, MIN_PA, PANEL_PATH,
                            _zscore_apply, _zscore_fit, build_model_frame)
import statsmodels.formula.api as smf

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
    print(f"n0_K={n0_k:.1f}  n0_BB={n0_bb:.1f}  (fit on all 2019-2025 target rows)")

    mk, sk, dk, fit_k, resid_k, wrmse_k = fit_full(elig_hist, "K", K_FEATURES_SWSTR, n0_k)
    mbb, sbb, dbb, fit_bb, resid_bb, wrmse_bb = fit_full(elig_hist, "BB", BB_FEATURES, n0_bb)
    print(f"\nFinal K model (wRMSE={wrmse_k*100:.3f}pp):\n{mk.summary2().tables[1].round(4)}")
    print(f"\nFinal BB model (wRMSE={wrmse_bb*100:.3f}pp):\n{mbb.summary2().tables[1].round(4)}")

    # residual correlation for K-BB uncertainty propagation -- matched pairs only
    both = dk[["pitcher", "target_year"]].copy()
    both["resid_k"] = resid_k
    bb_r = dbb[["pitcher", "target_year"]].copy()
    bb_r["resid_bb"] = resid_bb
    matched = both.merge(bb_r, on=["pitcher", "target_year"])
    rho = matched["resid_k"].corr(matched["resid_bb"])
    print(f"\nmeasured residual correlation corr(K resid, BB resid) = {rho:.3f}  (n={len(matched)})")

    # ---- build the 2026 target rows: 2025 season's own values as the "feature year" ---
    feat_2025 = panel[panel["game_year"] == 2025].copy()
    season_2025 = feat_2025[["pitcher", "player_name", "PA", "K_rate", "BB_rate", "BF_per_G"]]

    marcel_2026_rows = []
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
    marcel_2026 = pd.DataFrame({"pitcher": pitchers[mask], "Marcel_K_rate": marcel_k_2026[mask],
                                 "Marcel_BB_rate": marcel_bb_2026[mask], "Marcel_weighted_PA": den[mask]})

    bio = pd.read_parquet("output/pitcher_bio.parquet")[["pitcher_id", "birth_date"]].rename(
        columns={"pitcher_id": "pitcher"})
    proj = marcel_2026.merge(bio, on="pitcher", how="left")
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
    proj["BB_2026_hat"] = mbb.predict(proj) / 100
    proj["KBB_2026_hat"] = proj["K_2026_hat"] - proj["BB_2026_hat"]

    # uncertainty: PA proxy = pitcher's own 2025 PA (fallback: Marcel_weighted_PA)
    proj = proj.merge(season_2025[["pitcher", "PA"]].rename(columns={"PA": "PA_2025"}), on="pitcher", how="left")
    proj["PA_proxy"] = proj["PA_2025"].fillna(proj["Marcel_weighted_PA"])
    rel_k = mm.reliability_weight(proj["PA_proxy"], n0_k)
    rel_bb = mm.reliability_weight(proj["PA_proxy"], n0_bb)
    proj["SE_K"] = wrmse_k / np.sqrt(rel_k.clip(lower=0.05))
    proj["SE_BB"] = wrmse_bb / np.sqrt(rel_bb.clip(lower=0.05))
    proj["SE_KBB"] = mm.propagate_kbb_uncertainty(proj["SE_K"], proj["SE_BB"], rho)

    id_map = pd.read_parquet("output/pitcher_id_name_map.parquet").rename(columns={"pitcher_id": "pitcher"})
    proj = proj.merge(id_map, on="pitcher", how="left")
    proj = proj.merge(season_2025[["pitcher", "K_rate", "BB_rate"]].rename(
        columns={"K_rate": "K_2025", "BB_rate": "BB_2025"}), on="pitcher", how="left")
    proj["KBB_2025"] = proj["K_2025"] - proj["BB_2025"]

    out = proj[["pitcher", "name", "age", "PA_2025", "K_2025", "K_2026_hat", "SE_K",
                "BB_2025", "BB_2026_hat", "SE_BB", "KBB_2025", "KBB_2026_hat", "SE_KBB"]]
    out = out[out["PA_2025"].fillna(0) >= MIN_PA].sort_values("KBB_2026_hat", ascending=False)
    out.to_csv("output/kbb_projection_2026.csv", index=False)
    print(f"\nSaved -> output/kbb_projection_2026.csv, {len(out)} pitchers")

    print("\n-- Top 15 projected K-BB% for 2026 --")
    print(out.head(15).round(4).to_string(index=False))
    print("\n-- Sanity check: 10 well-known names --")
    known = ["Skenes", "Cole", "Skubal", "Wheeler", "Burnes", "Gilbert", "Sale",
             "Webb", "Ohtani", "Strider"]
    for nm in known:
        row = out[out["name"].str.contains(nm, case=False, na=False)]
        if len(row):
            print(row.round(4).to_string(index=False, header=False))
        else:
            print(f"  ({nm}: not found -- may not meet PA>= {MIN_PA} in 2025 or missing bio)")
