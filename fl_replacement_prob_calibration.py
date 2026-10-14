"""
fl_replacement_prob_calibration.py -- is P(> Repl.) actually calibrated? i.e.
among pitchers assigned probability ~p of beating replacement level, do
roughly p of them actually do so?

Uses the ONE real transition available (2025 actual K%/BB% -> 2026 actual
RA9, war_pitchers, the same n=96 population fit_ra9_baseline_model trains
on). Critically, this is NOT scored with the single frozen production model
(that would be in-sample for every pitcher used to fit it) -- it refits
RA9_SPEC via leave-one-pitcher-out for each of the 96, so each pitcher's
predicted RA9 (and therefore P(> Repl.)) comes from a model that never saw
their own 2026 outcome. Same discipline as every other held-out check in
this project, and the same mechanism RA9_RESID_SD=1.6087 itself came from
(reproduced here from scratch since that LOOCV run wasn't preserved as a
script -- this run should land close to 1.6087 as a sanity check).

Usage:
    python fl_replacement_prob_calibration.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

from fl_supabase import read_sql_df
import mlb_marcel as mm
from fl_ra9_trustmark_projection import RA9_SPEC, RA9_N0_EVAL, RA9_RESID_SD, LEAGUE_RA9_REPLACEMENT

N_BINS = 5


def build_training_frame(war: pd.DataFrame) -> pd.DataFrame:
    w25 = war[war["year"] == 2025].copy()
    w26 = war[war["year"] == 2026].copy()
    m = w25.merge(w26, on="player", suffixes=("_2025", "_2026"))
    m = m[(m["bf_2025"] >= 100) & (m["bf_2026"] >= 100)].copy()
    m["K_pred"] = m["so_2025"] / m["bf_2025"]
    m["BB_pred"] = m["bb_2025"] / m["bf_2025"]
    m["target"] = m["ra9_2026"]
    return m.reset_index(drop=True)


def loocv_predict(m: pd.DataFrame) -> np.ndarray:
    """Leave-one-pitcher-out: predicted RA9 for pitcher i comes from a model
    fit on the other 95, never on pitcher i's own 2026 outcome."""
    preds = np.full(len(m), np.nan)
    for i in range(len(m)):
        train = m.drop(index=i)
        w = mm.reliability_weight(train["ip_2026"], RA9_N0_EVAL).values
        model = smf.wls(RA9_SPEC, data=train, weights=w).fit()
        preds[i] = model.predict(m.iloc[[i]]).values[0]
    return preds


def reliability_table(p_pred: np.ndarray, outcome: np.ndarray, weight: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Bin by predicted probability, compare mean predicted vs. observed
    (both unweighted-by-count and IP-weighted) -- the classic reliability
    diagram, in table form since no plotting dependency is assumed here."""
    order = np.argsort(p_pred)
    bins = np.array_split(order, n_bins)
    rows = []
    for b in bins:
        rows.append({
            "n": len(b),
            "pred_range": f"{p_pred[b].min():.0%}-{p_pred[b].max():.0%}",
            "mean_predicted": p_pred[b].mean(),
            "observed_rate (unweighted)": outcome[b].mean(),
            "observed_rate (IP-weighted)": np.average(outcome[b], weights=weight[b]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Loading war_pitchers (real 2025, 2026 actuals)...")
    war = read_sql_df("select player, year, bf, ip, so, bb, ra, ra9, rar from war_pitchers where game_type='reg_season'")
    m = build_training_frame(war)
    print(f"n={len(m)} real 2025-actual -> 2026-actual transitions (bf>=100 both years)")

    print("\nRunning leave-one-pitcher-out RA9 predictions (refits the model 96 times)...")
    m["ra9_pred_loocv"] = loocv_predict(m)

    w_eval = mm.reliability_weight(m["ip_2026"], RA9_N0_EVAL).values
    resid = m["target"].values - m["ra9_pred_loocv"].values
    resid_sd_check = mm.weighted_rmse(m["target"].values, m["ra9_pred_loocv"].values, w_eval)
    print(f"LOOCV RMSE_w (sanity check vs. hardcoded RA9_RESID_SD={RA9_RESID_SD}): {resid_sd_check:.4f}")

    # predicted P(beat replacement), from the LOOCV (held-out) point estimate,
    # using the SAME residual SD the real pipeline uses
    z = (LEAGUE_RA9_REPLACEMENT - m["ra9_pred_loocv"].values) / RA9_RESID_SD
    p_pred = norm.cdf(z)
    # real outcome: did this pitcher's ACTUAL 2026 RA9 actually beat replacement?
    outcome = (m["target"].values < LEAGUE_RA9_REPLACEMENT).astype(float)

    print(f"\nOverall: mean predicted P(> Repl.)={p_pred.mean():.3f}  "
          f"actual beat-replacement rate={outcome.mean():.3f}  "
          f"(IP-weighted actual rate={np.average(outcome, weights=w_eval):.3f})")

    brier = np.mean((p_pred - outcome) ** 2)
    brier_climatology = np.mean((outcome.mean() - outcome) ** 2)
    print(f"Brier score: {brier:.4f}  (vs. {brier_climatology:.4f} for a no-skill constant-rate forecast "
          f"-- lower is better, skill score = {1 - brier/brier_climatology:.3f})")

    # calibration slope/intercept: logistic regression of the REAL outcome on
    # the predicted logit. slope=1,intercept=0 is perfect calibration;
    # slope<1 means predicted probabilities are too extreme (overconfident);
    # intercept != 0 means a systematic over/under-shoot across the board.
    logit_p = np.log(p_pred / (1 - p_pred))
    cal = smf.logit("outcome ~ logit_p", data=pd.DataFrame({"outcome": outcome, "logit_p": logit_p})).fit(disp=0)
    print(f"\nCalibration regression (outcome ~ logit(p_pred)): "
          f"intercept={cal.params['Intercept']:.3f} (se={cal.bse['Intercept']:.3f}), "
          f"slope={cal.params['logit_p']:.3f} (se={cal.bse['logit_p']:.3f})  "
          f"[perfect = intercept 0, slope 1]")

    print(f"\n--- Reliability table ({N_BINS} bins, sorted by predicted probability) ---")
    tbl = reliability_table(p_pred, outcome, w_eval, N_BINS)
    print(tbl.to_string(index=False))
