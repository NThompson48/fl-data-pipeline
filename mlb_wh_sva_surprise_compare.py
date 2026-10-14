"""
mlb_wh_sva_surprise_compare.py -- does the early-season wh_sva "surprise" signal
(mlb_arsenal_depth_wh_sva_aging.py, Part 2c) sharpen if the baseline expectation
is a Marcel-style 5/4/3 weighted 3-year wh_sva history instead of just last
season's raw level?

Usage:
    python mlb_wh_sva_surprise_compare.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_kbb_model import PANEL_PATH

pd.set_option("display.width", 160, "display.max_columns", 40)


def marcel_weight_series(pivot_val: pd.DataFrame, pivot_n: pd.DataFrame, target_year: int) -> pd.Series:
    """Marcel-style 5/4/3-weighted average of a per-season MEAN quantity (not a
    rate built from counts) -- weight each trailing year's mean by BOTH the
    Marcel recency weight and that year's own pitch count (so a thin-sample
    trailing year doesn't count as much as a well-established one)."""
    trailing = [target_year - 1, target_year - 2, target_year - 3]
    rec_weights = [mm.MARCEL_WEIGHTS[lag] for lag in (0, 1, 2)]
    num = pd.Series(0.0, index=pivot_val.index)
    den = pd.Series(0.0, index=pivot_val.index)
    for rw, yr in zip(rec_weights, trailing):
        if yr not in pivot_val.columns:
            continue
        v = pivot_val[yr].fillna(0)
        n = pivot_n[yr].fillna(0) if yr in pivot_n.columns else 0
        w = rw * n
        num = num + w * v
        den = den + w
    out = num / den.replace(0, np.nan)
    return out


if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)
    panel = pd.read_parquet(PANEL_PATH)

    print("Refitting pooled overall wh_sva per-pitch (full history)...")
    wh_pitch = mf.pooled_wh_sva_per_pitch(raw)
    season_sva = wh_pitch.groupby(["pitcher", "game_year"]).agg(
        wh_sva_season=("sva", "mean"), n_sva=("sva", "size")).reset_index()

    pivot_val = season_sva.pivot(index="pitcher", columns="game_year", values="wh_sva_season")
    pivot_n = season_sva.pivot(index="pitcher", columns="game_year", values="n_sva")

    # Marcel-weighted wh_sva baseline, computed for EVERY target year at once
    rows = []
    for target_year in sorted(panel["game_year"].unique()):
        s = marcel_weight_series(pivot_val, pivot_n, target_year)
        rows.append(pd.DataFrame({"pitcher": s.index, "game_year": target_year, "wh_sva_marcel": s.values}))
    wh_sva_marcel = pd.concat(rows, ignore_index=True).dropna()

    # first-half K_rate / wh_sva split (identical construction to the previous script)
    term = raw[raw["is_terminal_pitch"]].copy()
    term["is_k"] = term["events"].astype(str).isin(mf.K_EVENTS)
    term = term.sort_values(["pitcher", "game_year", "game_date", "at_bat_number", "pitch_number"])
    grp = term.groupby(["pitcher", "game_year"])
    rank = grp.cumcount()
    counts = grp["is_k"].transform("size")
    term["_half"] = np.where(rank < counts / 2, "first", "second")
    half_k = term.groupby(["pitcher", "game_year", "_half"]).agg(PA=("is_k", "size"), K=("is_k", "sum")).reset_index()
    half_k["K_rate_half"] = half_k["K"] / half_k["PA"]
    piv = half_k.pivot(index=["pitcher", "game_year"], columns="_half", values=["K_rate_half", "PA"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv = piv[(piv["PA_first"] >= 50) & (piv["PA_second"] >= 50)]

    wh_first = wh_pitch[wh_pitch.groupby(["pitcher", "game_year"]).cumcount() <
                         wh_pitch.groupby(["pitcher", "game_year"])["sva"].transform("size") / 2]
    wh_first_agg = wh_first.groupby(["pitcher", "game_year"]).agg(
        wh_sva_first=("sva", "mean"), n_first=("sva", "size")).reset_index()
    wh_first_agg = wh_first_agg[wh_first_agg["n_first"] >= 100]

    d2 = piv.merge(wh_first_agg, on=["pitcher", "game_year"], how="inner")
    d2 = d2.merge(panel[["pitcher", "game_year", "Marcel_K_rate", "age_c", "age_decline"]],
                  on=["pitcher", "game_year"], how="left")

    # baseline A: last year only (previous script's version)
    last_year = season_sva[["pitcher", "game_year", "wh_sva_season"]].copy()
    last_year["game_year"] += 1
    last_year = last_year.rename(columns={"wh_sva_season": "wh_sva_last_year"})
    d2 = d2.merge(last_year, on=["pitcher", "game_year"], how="left")

    # baseline B: Marcel 5/4/3-weighted 3-year wh_sva
    d2 = d2.merge(wh_sva_marcel, on=["pitcher", "game_year"], how="left")

    d2 = d2.dropna(subset=["Marcel_K_rate", "K_rate_half_second"])
    d2_last = d2.dropna(subset=["wh_sva_last_year"]).copy()
    d2_last["surprise_lastyr"] = d2_last["wh_sva_first"] - d2_last["wh_sva_last_year"]
    d2_marcel = d2.dropna(subset=["wh_sva_marcel"]).copy()
    d2_marcel["surprise_marcel"] = d2_marcel["wh_sva_first"] - d2_marcel["wh_sva_marcel"]

    # apples-to-apples: restrict BOTH to the same rows (have both baselines available)
    common = d2_last.merge(d2_marcel[["pitcher", "game_year", "surprise_marcel", "wh_sva_marcel"]],
                            on=["pitcher", "game_year"], how="inner")
    print(f"\ncommon n = {len(common)} (pitcher-seasons with both a last-year AND a Marcel-weighted wh_sva baseline)")

    def wrmse(fit, data, ycol="K_rate_half_second", wcol="PA_second"):
        return np.sqrt(np.average((data[ycol] * 100 - fit.fittedvalues) ** 2, weights=data[wcol]))

    m_base = smf.wls("I(K_rate_half_second*100) ~ I(Marcel_K_rate*100) + age_c + age_decline",
                      data=common, weights=common["PA_second"]).fit()
    m_lastyr = smf.wls("I(K_rate_half_second*100) ~ I(Marcel_K_rate*100) + age_c + age_decline + surprise_lastyr",
                        data=common, weights=common["PA_second"]).fit()
    m_marcel = smf.wls("I(K_rate_half_second*100) ~ I(Marcel_K_rate*100) + age_c + age_decline + surprise_marcel",
                        data=common, weights=common["PA_second"]).fit()
    m_both = smf.wls("I(K_rate_half_second*100) ~ I(Marcel_K_rate*100) + age_c + age_decline "
                      "+ surprise_lastyr + surprise_marcel", data=common, weights=common["PA_second"]).fit()

    print(f"\npredicting the SAME season's second-half K_rate (n={len(common)}, apples-to-apples sample):")
    print(f"  preseason (Marcel+age) only:            wRMSE={wrmse(m_base, common):.3f}pp  R2={m_base.rsquared:.4f}")
    print(f"  + surprise vs. LAST YEAR only:           wRMSE={wrmse(m_lastyr, common):.3f}pp  R2={m_lastyr.rsquared:.4f}  "
          f"(coef={m_lastyr.params['surprise_lastyr']:.2f}, p={m_lastyr.pvalues['surprise_lastyr']:.4f})")
    print(f"  + surprise vs. MARCEL 5/4/3 3yr wh_sva:  wRMSE={wrmse(m_marcel, common):.3f}pp  R2={m_marcel.rsquared:.4f}  "
          f"(coef={m_marcel.params['surprise_marcel']:.2f}, p={m_marcel.pvalues['surprise_marcel']:.4f})")
    print(f"  + BOTH surprise terms:                  wRMSE={wrmse(m_both, common):.3f}pp  R2={m_both.rsquared:.4f}")
    print(f"      surprise_lastyr coef={m_both.params['surprise_lastyr']:.2f}, p={m_both.pvalues['surprise_lastyr']:.4f}")
    print(f"      surprise_marcel coef={m_both.params['surprise_marcel']:.2f}, p={m_both.pvalues['surprise_marcel']:.4f}")

    # correlation between the two surprise definitions themselves
    print(f"\ncorr(surprise_lastyr, surprise_marcel) = {common['surprise_lastyr'].corr(common['surprise_marcel']):.3f}")
    print(f"n_pitcher-seasons with a full 3-year wh_sva history available for the Marcel baseline: "
          f"{d2_marcel['pitcher'].nunique()} pitchers, {len(d2_marcel)} rows "
          f"(vs. {len(d2_last)} rows / any single trailing year for the last-year-only baseline)")
