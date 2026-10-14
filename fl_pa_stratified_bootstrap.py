"""
fl_pa_stratified_bootstrap.py -- replaces the parametric logit-delta-method
CI (which fl_ci_assumption_checks.py showed rests on two violated
assumptions: residuals aren't normal -- skew=+0.40, excess kurtosis=+2.03 --
and the flat K_SE_CALIBRATION=1.25 isn't uniform across PA -- needed
multiplier ranges 0.90x-1.30x by bucket) with a PA-stratified empirical
bootstrap:

  1. Case-resampling + refit (parameter/coefficient uncertainty) -- same as
     fl_bootstrap_pi_test.py, unchanged.
  2. Outcome noise drawn from TRAINING residuals in the SAME PA bucket as
     the target (not one global pool rescaled by an assumed 1/w formula) --
     this lets the simulated noise inherit whatever shape AND PA-scaling
     the real data actually has, instead of assuming either.

Validated the same way as fl_ci_assumption_checks.py's Check 3: coverage by
PA bucket, on the large pooled early-cutoff dataset (fl_low_pa_ci_calibration.py).

Usage:
    python fl_pa_stratified_bootstrap.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
from fl_low_pa_ci_calibration import build_fold, CUTOFF_WEEKS, SPEC

N_BOOT = 400
RNG = np.random.default_rng(42)
PA_EDGES = [0, 5, 10, 20, 40, 80, 99999]
PA_LABELS = ["1-5", "6-10", "11-20", "21-40", "41-80", "80+"]


def assign_bucket(pa_series):
    return pd.cut(pa_series, PA_EDGES, labels=PA_LABELS)


def pa_stratified_bootstrap_pi(d_tr_full, d_te, target_col, spec, n0, donor_pool_raw):
    """Returns (n_boot, n_test) simulated outcome array.

    Two SEPARATE resampled populations, deliberately: `d_tr_full` (the real
    production training transition) drives the refit coefficients each
    iteration (parameter uncertainty, exactly what production actually
    fits on) -- `donor_pool_raw` (the much richer, PA-diverse early-cutoff
    pool) supplies the PA-matched residual donors (outcome noise). The
    production training set is almost entirely high-PA players (only 1-2
    pitchers in the lowest PA buckets -- confirmed directly), so using it
    as the donor source starves the low-PA buckets of any real variability
    to draw from; the donor pool doesn't have that problem by construction
    (early-season snapshots span the full PA range)."""
    d_tr_full = d_tr_full.copy()
    d_te = d_te.copy()
    d_te["pa_bucket"] = assign_bucket(d_te["PA"])
    donor_pool_raw = donor_pool_raw.copy()
    donor_pool_raw["pa_bucket"] = assign_bucket(donor_pool_raw["PA"])

    n_tr = len(d_tr_full)
    n_donor = len(donor_pool_raw)
    n_te = len(d_te)
    sims = np.full((N_BOOT, n_te), np.nan)

    for b in range(N_BOOT):
        idx = RNG.integers(0, n_tr, size=n_tr)
        boot_tr = d_tr_full.iloc[idx]
        w_tr_b = mm.reliability_weight(boot_tr["PA_test"] if "PA_test" in boot_tr else boot_tr["PA"], n0).values
        try:
            m_b = bt.fit_weighted(boot_tr, target_col, spec, w_tr_b)
        except Exception:
            continue

        didx = RNG.integers(0, n_donor, size=n_donor)
        boot_donor = donor_pool_raw.iloc[didx].copy()
        boot_donor["resid"] = boot_donor[target_col].values - m_b.predict(boot_donor).values

        pred_te_b = np.clip(m_b.predict(d_te).values, 1e-6, 1 - 1e-6)

        # PA-matched donor draw: for each test target, sample a residual from
        # DONOR-POOL cases in the SAME pa_bucket
        resid_sim = np.empty(n_te)
        for bucket in PA_LABELS:
            mask = (d_te["pa_bucket"] == bucket).values
            if not mask.any():
                continue
            donors = boot_donor.loc[boot_donor["pa_bucket"] == bucket, "resid"]
            if donors.empty:
                donors = boot_donor["resid"]
            resid_sim[mask] = RNG.choice(donors.values, size=mask.sum(), replace=True)

        sims[b, :] = np.clip(pred_te_b + resid_sim, 0.0, 1.0)

    return sims


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    d_tr_k = df_tr.dropna(subset=["K_rate_actual"] + SPEC["K"]).reset_index(drop=True)
    print(f"production training set (coefficient fit): n={len(d_tr_k)}")

    full_pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025),
                           fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True)
    term = full_pool[full_pool["is_terminal"]]
    lg_k = (term["k_or_bb"] == "strikeout").mean()
    lg_bb = (term["k_or_bb"] == "walk").mean()

    SEASONS = {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}
    print("Building early-cutoff folds, split by YEAR (2025=donor pool, 2026=held-out test)...")
    frames_2025, frames_2026 = [], []
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
            (frames_2025 if year == 2025 else frames_2026).append(fold)
    donor_pool_raw = pd.concat(frames_2025, ignore_index=True).dropna(subset=["K_rate_actual"] + SPEC["K"])
    d_te = pd.concat(frames_2026, ignore_index=True).dropna(subset=["K_rate_actual"] + SPEC["K"]).reset_index(drop=True)
    print(f"donor pool (2025 early-cutoffs): n={len(donor_pool_raw)}  "
          f"held-out test (2026 early-cutoffs): n={len(d_te)}")
    print("donor pool size by PA bucket:")
    print(assign_bucket(donor_pool_raw["PA"]).value_counts().sort_index())

    print(f"\nRunning {N_BOOT}-iteration PA-stratified bootstrap "
          f"(coefficients resampled from production training set, residual noise from the richer 2025 donor pool)...")
    sims = pa_stratified_bootstrap_pi(d_tr_k, d_te, "K_rate_actual", SPEC["K"], n0_k, donor_pool_raw)

    lo = np.nanpercentile(sims, 5, axis=0)
    hi = np.nanpercentile(sims, 95, axis=0)
    actual = d_te["K_rate_actual"].values
    covered = (actual >= lo) & (actual <= hi)
    width = hi - lo

    d_te["pa_bucket"] = assign_bucket(d_te["PA"])
    d_te["covered"] = covered
    d_te["width"] = width

    print(f"\noverall coverage: {covered.mean():.1%} (nominal 90%)")
    print("\ncoverage + width by PA bucket (PA-stratified bootstrap):")
    print(d_te.groupby("pa_bucket", observed=True).agg(
        n=("covered", "size"), coverage=("covered", "mean"), mean_width=("width", "mean")).round(3))
