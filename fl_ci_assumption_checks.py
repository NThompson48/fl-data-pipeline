"""
fl_ci_assumption_checks.py -- audits the actual statistical assumptions
behind the K% prediction interval, rather than just checking aggregate
coverage (already done in fl_low_pa_ci_calibration.py). Three checks:

1. Does Var(residual) really scale as 1/w (w=PA/(PA+n0)), the assumption
   baked into WLS's get_prediction().se_obs? This is THE assumption behind
   the whole low-PA-widens-the-interval behavior -- if real variance scales
   differently, se_obs is systematically wrong in a PA-dependent way, not
   just off by a flat constant.
2. Are the (weighted) residuals actually close to normal? The logit delta-
   method CI implicitly assumes something like normality on the logit
   scale; heavy tails or skew would mean the STATED coverage is right on
   average (already confirmed) while being wrong in a way that matters
   (e.g. one tail too fat, the other too thin).
3. Does the flat K_SE_CALIBRATION=1.25 correction actually hold uniformly
   across PA levels, or does the TRUE needed correction vary by PA (meaning
   a single constant over- or under-corrects different players)?

Reuses fl_low_pa_ci_calibration.py's pooled early-cutoff dataset (2,747 real
thin-sample-snapshot -> actual-rest-of-season-outcome pairs) for power.

Usage:
    python fl_ci_assumption_checks.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_production_projection as fprod
import mlb_marcel as mm
from fl_low_pa_ci_calibration import build_fold, CUTOFF_WEEKS, Z_90, SPEC, logit_ci

if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    d_tr_k = df_tr.dropna(subset=["K_rate_actual"] + SPEC["K"])
    w_tr_k = mm.reliability_weight(d_tr_k["PA_test"], n0_k).values
    m_k = bt.fit_weighted(d_tr_k, "K_rate_actual", SPEC["K"], w_tr_k)
    print(f"WLS residual scale (m_k.scale): {m_k.scale:.6f}")

    full_pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025),
                           fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True)
    term = full_pool[full_pool["is_terminal"]]
    lg_k = (term["k_or_bb"] == "strikeout").mean()
    lg_bb = (term["k_or_bb"] == "walk").mean()

    SEASONS = {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}
    print("Building pooled early-cutoff dataset...")
    frames = []
    for year, (season_start, season_end) in SEASONS.items():
        prior_windows = [(bt.START_2025, bt.END_2025)] if year == 2026 else []
        for wk in CUTOFF_WEEKS:
            cutoff = (pd.Timestamp(season_start) + pd.Timedelta(weeks=wk)).strftime("%Y-%m-%d")
            day_after = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.Timestamp(day_after) >= pd.Timestamp(season_end) - pd.Timedelta(weeks=2):
                continue
            surprise_windows = prior_windows + [(season_start, cutoff)]
            fold = build_fold(raw, season_start, cutoff, surprise_windows, day_after, season_end,
                               n0_k, n0_bb, lg_k, lg_bb)
            frames.append(fold)
    pooled = pd.concat(frames, ignore_index=True)
    d = pooled.dropna(subset=["K_rate_actual"] + SPEC["K"]).copy()
    w = mm.reliability_weight(d["PA"], n0_k).values
    d["w"] = w
    d["pred_K"] = m_k.predict(d)
    d["resid"] = d["K_rate_actual"] - d["pred_K"]
    d["se_K_raw"] = m_k.get_prediction(d, weights=w).se_obs  # WITHOUT the 1.25 calibration, for the raw-scaling check

    print(f"\nn={len(d)} pooled pairs")

    # ------------------------------------------------------------------
    # CHECK 1: does empirical Var(residual) scale as 1/w?
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CHECK 1: Var(residual) vs. 1/w scaling assumption")
    print("=" * 90)
    d["pa_bucket"] = pd.cut(d["PA"], [0, 5, 10, 20, 40, 80, 99999], labels=["1-5", "6-10", "11-20", "21-40", "41-80", "80+"])
    g = d.groupby("pa_bucket", observed=True).agg(
        n=("resid", "size"), mean_w=("w", "mean"), empirical_var=("resid", "var"),
    ).reset_index()
    g["implied_var_from_w"] = m_k.scale / g["mean_w"]  # the model's OWN assumption: Var = scale/w
    g["ratio_empirical_to_implied"] = g["empirical_var"] / g["implied_var_from_w"]
    print(g.round(5).to_string(index=False))
    print("\nratio should be ~1.0 everywhere if the 1/w scaling assumption is correct;")
    print("a ratio that trends up/down across PA buckets means real variance scales")
    print("differently than 1/w implies, not just by a flat constant.")

    # ------------------------------------------------------------------
    # CHECK 2: residual normality (on the weighted/standardized scale)
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CHECK 2: standardized residual shape (z = resid * sqrt(w) / sqrt(scale))")
    print("=" * 90)
    z_resid = d["resid"] * np.sqrt(d["w"]) / np.sqrt(m_k.scale)
    print(f"mean: {z_resid.mean():.3f} (expect ~0)")
    print(f"std:  {z_resid.std():.3f} (expect ~1)")
    print(f"skew: {stats.skew(z_resid):.3f} (expect ~0; >0 = right tail fatter)")
    print(f"excess kurtosis: {stats.kurtosis(z_resid):.3f} (expect ~0; >0 = heavier tails than normal)")
    sw_stat, sw_p = stats.shapiro(z_resid.sample(min(5000, len(z_resid)), random_state=42))
    print(f"Shapiro-Wilk normality test: stat={sw_stat:.4f}, p={sw_p:.2e} (p<0.05 rejects normality)")
    print("\npercentile comparison (empirical vs standard normal):")
    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        emp = z_resid.quantile(q)
        norm_q = stats.norm.ppf(q)
        print(f"  {q:.0%}: empirical={emp:+.3f}  normal={norm_q:+.3f}  diff={emp-norm_q:+.3f}")

    # ------------------------------------------------------------------
    # CHECK 3: does the needed calibration factor vary by PA?
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CHECK 3: needed SE-multiplier for exactly 90% coverage, by PA bucket")
    print("=" * 90)
    print("(current production uses a FLAT 1.25x on top of raw se_obs)")
    for bucket_name, sub in d.groupby("pa_bucket", observed=True):
        if len(sub) < 30:
            continue
        # binary search the multiplier that gives ~90% coverage in this bucket
        best_mult, best_gap = None, 999
        for mult in np.arange(0.5, 3.01, 0.05):
            se_adj = sub["se_K_raw"] * mult
            lo, hi = logit_ci(sub["pred_K"].values, se_adj.values, Z_90)
            cov = ((sub["K_rate_actual"].values >= lo) & (sub["K_rate_actual"].values <= hi)).mean()
            gap = abs(cov - 0.90)
            if gap < best_gap:
                best_gap, best_mult, best_cov = gap, mult, cov
        print(f"  {bucket_name:>6} (n={len(sub):4d}): needed multiplier ~{best_mult:.2f}x "
              f"(achieves {best_cov:.1%} coverage)")
