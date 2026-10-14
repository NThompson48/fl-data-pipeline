"""
fl_pwar_calibration.py -- is the leaderboard's P(WAR>0) calibrated?

Fully out-of-sample 2025 -> 2026 on the BROAD population (every pitcher with >= 10 IP in
2026, including debuts), with the CURRENT production design:
  * K%/BB% inputs: cross-fitted projections (fl_ra9_input_robustness.oof_kbb_projections) or,
    for pitchers with no 2025 FL record, the other-league bridge / league mean;
  * RA9 baseline: park-neutral K% + BB% + IP/appearance, K-fold cross-fitted on the BF>=100
    sample (full fit for everyone else);
  * innings: IP_proj = 19.31 + 0.6278 x IP_2025 (24 for gap returners, 18 for debuts);
  * SD = sqrt(185.16 / IP_proj); replacement = the 2026 league constant 6.963.
Compares the predicted P(RA9 < replacement) with whether the pitcher's ACTUAL 2026 RA9 was below
replacement, by predicted-probability bin and by workload; reports the Brier score against the
base-rate guess.

Usage:  python fl_pwar_calibration.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from sklearn.model_selection import KFold
from sqlalchemy import text

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_ra9_input_robustness as rob
import fl_ra9_uncertainty_model as um
import fl_ra9_trustmark_projection as ra9
from fl_supabase import read_sql_df

SPEC = ra9.RA9_SPEC_PROD
RETURN_ALL = False


def main():
    raw = fp.load_fl_pitches_cached()
    war = read_sql_df(text("select player, year, bf, ip, so, bb, ra, ra9, rar, ppf, g from war_pitchers where game_type='reg_season'"))
    b = um.build_forecasts(raw, war).reset_index(drop=True)
    w25 = (bt.START_2025, bt.END_2025)
    ids = pd.DataFrame({"pitcher_id": b["pitcher_id"], "K_oof": 0.0, "BB_oof": 0.0})
    fit = rob.build_frame(war, raw, ids, 2025, 2026, w25)
    fit["K_pred"], fit["BB_pred"] = fit["K_raw"], fit["BB_raw"]
    lo, hi = fit["ip_per_g_prior"].min(), fit["ip_per_g_prior"].max()
    b["ipg"] = b["ip_per_g_prior"].clip(lo, hi)
    fit["ip_per_g_prior"] = fit["ip_per_g_prior"]

    def fm(d):
        return smf.wls(SPEC, data=d, weights=d["w"]).fit()
    b["pred"] = np.nan
    full = fm(fit)
    out = ~b["pitcher_id"].isin(fit["pitcher_id"])
    bb = b.copy(); bb["ip_per_g_prior"] = b["ipg"]
    b.loc[out, "pred"] = full.predict(bb[out]).values
    for tri, tei in KFold(10, shuffle=True, random_state=0).split(fit):
        m = fm(fit.iloc[tri]); ids_te = fit.iloc[tei]["pitcher_id"]
        sel = b["pitcher_id"].isin(ids_te)
        b.loc[sel, "pred"] = m.predict(bb[sel]).values

    # projected innings (what production would have had): 2025 IP -> 2026
    ip25 = war[war["year"] == 2025].groupby("player")["ip"].sum()
    has25 = b["player"].map(ip25).fillna(0)
    b["ip_proj"] = np.where(has25 > 0, 19.31 + 0.6278 * has25, 18.0).clip(5)
    b["sd"] = np.sqrt(185.16 / b["ip_proj"])
    repl = ra9.LEAGUE_RA9_REPLACEMENT
    b["P"] = norm.cdf((repl - b["pred"]) / b["sd"])
    b["y"] = (b["target"] < repl).astype(float)
    if RETURN_ALL:
        return b, war
    d = b[b["ip"] >= 10].copy()
    print(f"broad sample n={len(d)} (IP_2026>=10); base rate beating replacement {d['y'].mean():.3f}; mean predicted {d['P'].mean():.3f}")

    d["bin"] = pd.cut(d["P"], [0, .4, .5, .6, .7, .8, .9, 1.0])
    t = d.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("P", "mean"), actual=("y", "mean"),
                                            actual_IPw=("y", lambda s: np.average(s, weights=d.loc[s.index, "ip"]))).round(3)
    print("\nBY PREDICTED-PROBABILITY BIN"); print(t.to_string())
    d["ipb"] = pd.cut(d["ip"], [0, 25, 50, 90, 1000], labels=["10-25 IP", "25-50", "50-90", "90+"])
    t2 = d.groupby("ipb", observed=True).agg(n=("y", "size"), predicted=("P", "mean"), actual=("y", "mean")).round(3)
    print("\nBY 2026 WORKLOAD"); print(t2.to_string())
    brier = np.mean((d["P"] - d["y"]) ** 2); base = d["y"].mean(); brier0 = np.mean((base - d["y"]) ** 2)
    print(f"\nBrier score {brier:.4f} vs base-rate guess {brier0:.4f} (skill {1 - brier/brier0:+.1%}); "
          f"AUC-like: mean P among beaters {d.loc[d.y==1,'P'].mean():.3f} vs non-beaters {d.loc[d.y==0,'P'].mean():.3f}")
    # slope of actual on predicted (logit-scale calibration)
    d["logitP"] = np.log(d["P"].clip(1e-3, 1 - 1e-3) / (1 - d["P"].clip(1e-3, 1 - 1e-3)))
    m = smf.logit("y ~ logitP", data=d).fit(disp=0)
    print(f"logistic recalibration: actual-logit = {m.params['Intercept']:+.2f} + {m.params['logitP']:.2f} x predicted-logit "
          f"(1.0 slope / 0 intercept = perfectly calibrated); slope se {m.bse['logitP']:.2f}")
    return d, war


if __name__ == "__main__":
    main()
