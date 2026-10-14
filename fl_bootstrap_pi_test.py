"""
fl_bootstrap_pi_test.py -- calibration check for the dashboard's logit-delta-
method prediction intervals (fl_production_projection.py's proj_K_rate_se/
proj_BB_rate_se), via a case-resampling bootstrap that does NOT rely on the
WLS closed-form formula's distributional assumptions.

Two things the analytic interval currently treats as known, fixed truth that
this bootstrap instead re-estimates on every resample:
  - n0 (the empirical-Bayes shrinkage constant, fit once via method-of-
    moments and then used as if exact everywhere else in this project)
  - the outer K%/BB% regression's own coefficients

Scope limitation, stated not hidden: this does NOT refit the other-league
bridge or wh_sva/horiz_var/vert_var on each resample (those stay fixed at
their already-computed values) -- refitting the bridge's own name-matching +
regression and the wh_sva XGBoost model hundreds of times would be far more
expensive for a marginal gain, since those are themselves lower-variance,
population-level fits on larger samples than the outer regression's ~130-260
transition pairs. So this is a PARTIAL bootstrap (captures n0 + outer-
regression parameter uncertainty), not a full one.

For each resample: resample TRAIN pitchers with replacement, re-estimate n0,
refit the outer WLS model, predict on the (unresampled) TEST set, then
simulate INDIVIDUAL outcome noise via a binomial draw at each test pitcher's
own PA (K_sim ~ Binomial(PA_test, bootstrap_predicted_rate)/PA_test) --
avoids any normal/logit-normal residual assumption entirely, and is
automatically bounded in [0,1] by construction.

Usage:
    python fl_bootstrap_pi_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

N_BOOT = 500
RNG = np.random.default_rng(42)
SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]


def logit_ci(p, se, z):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    se_logit = se / (p * (1 - p))
    lp = np.log(p / (1 - p))
    lo = 1 / (1 + np.exp(-(lp - z * se_logit)))
    hi = 1 / (1 + np.exp(-(lp + z * se_logit)))
    return lo, hi


def bootstrap_pi(df_tr, df_te, target_col, spec, rate_col_for_n0, min_n0):
    """Returns an (n_boot, n_test) array of simulated outcomes for each test
    pitcher, combining refit-parameter uncertainty with individual outcome
    noise drawn EMPIRICALLY from the bootstrap's own train-set residuals
    (weighted-rescaled to the target pitcher's own reliability), not a
    theoretical Binomial(PA, p) draw -- confirmed directly that real K%/BB%
    residuals are overdispersed relative to pure binomial sampling noise by
    ~1.24-1.48x (real year-to-year true-rate drift on top of sampling noise,
    e.g. role/health/matchup changes), so a pure-binomial simulation
    understates individual-level noise and must not be used here."""
    d_tr_full = df_tr.dropna(subset=[target_col] + spec).reset_index(drop=True)
    d_te = df_te.dropna(subset=[target_col] + spec).reset_index(drop=True)
    n_tr = len(d_tr_full)
    n_te = len(d_te)
    sims = np.full((N_BOOT, n_te), np.nan)

    for b in range(N_BOOT):
        idx = RNG.integers(0, n_tr, size=n_tr)
        boot_tr = d_tr_full.iloc[idx]

        n0_b, _ = mm.estimate_n0(boot_tr[rate_col_for_n0], boot_tr["PA_test"], min_n=min_n0)
        w_tr_b = mm.reliability_weight(boot_tr["PA_test"], n0_b).values
        try:
            m_b = bt.fit_weighted(boot_tr, target_col, spec, w_tr_b)
        except Exception:
            continue

        # empirical residuals from THIS bootstrap's own train fit, standardized
        # by their own weight (Var(resid) ~ 1/w under the WLS assumption) so
        # they can be rescaled to each target pitcher's own reliability
        resid_b = boot_tr[target_col].values - m_b.predict(boot_tr).values
        z_b = resid_b * np.sqrt(w_tr_b)  # ~unit-variance "z-scores"

        pred_b = np.clip(m_b.predict(d_te).values, 1e-6, 1 - 1e-6)
        w_te_b = mm.reliability_weight(d_te["PA_test"], n0_b).values
        donor_z = RNG.choice(z_b, size=n_te, replace=True)
        resid_sim = donor_z / np.sqrt(w_te_b)
        sims[b, :] = np.clip(pred_b + resid_sim, 0.0, 1.0)

    return sims, d_te


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)],
                                      [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    print(f"\nRunning {N_BOOT}-iteration bootstrap for K%...")
    sims_k, d_te_k = bootstrap_pi(df_tr, df_te, "K_rate_actual", SPEC["K"], "K_rate_actual", bt.MIN_PA_N0)
    print(f"Running {N_BOOT}-iteration bootstrap for BB%...")
    sims_bb, d_te_bb = bootstrap_pi(df_tr, df_te, "BB_rate_actual", SPEC["BB"], "BB_rate_actual", bt.MIN_PA_N0)

    # current analytic (logit-delta-method) interval, for comparison -- same
    # fit the production dashboard actually uses
    w_tr_k = mm.reliability_weight(df_tr.dropna(subset=["K_rate_actual"] + SPEC["K"])["PA_test"], n0_k).values
    d_tr_k = df_tr.dropna(subset=["K_rate_actual"] + SPEC["K"])
    m_k = bt.fit_weighted(d_tr_k, "K_rate_actual", SPEC["K"], w_tr_k)
    w_te_k = mm.reliability_weight(d_te_k["PA_test"], n0_k).values
    pred_k = m_k.predict(d_te_k).values
    se_k = m_k.get_prediction(d_te_k, weights=w_te_k).se_obs

    d_tr_bb = df_tr.dropna(subset=["BB_rate_actual"] + SPEC["BB"])
    w_tr_bb = mm.reliability_weight(d_tr_bb["PA_test"], n0_bb).values
    m_bb = bt.fit_weighted(d_tr_bb, "BB_rate_actual", SPEC["BB"], w_tr_bb)
    w_te_bb = mm.reliability_weight(d_te_bb["PA_test"], n0_bb).values
    pred_bb = m_bb.predict(d_te_bb).values
    se_bb = m_bb.get_prediction(d_te_bb, weights=w_te_bb).se_obs

    print("\n" + "=" * 90)
    print("Comparing analytic (logit-delta-method) 90% PI vs bootstrap empirical 90% PI")
    print("=" * 90)
    for label, sims, pred, se, pa in [
        ("K%", sims_k, pred_k, se_k, d_te_k["PA_test"].values),
        ("BB%", sims_bb, pred_bb, se_bb, d_te_bb["PA_test"].values),
    ]:
        boot_lo = np.nanpercentile(sims, 5, axis=0)
        boot_hi = np.nanpercentile(sims, 95, axis=0)
        boot_width = boot_hi - boot_lo

        an_lo, an_hi = logit_ci(pred, se, 1.6449)
        an_width = an_hi - an_lo

        print(f"\n--- {label} (n={len(pred)}) ---")
        print(f"  Analytic median width:  {np.median(an_width):.1%}")
        print(f"  Bootstrap median width: {np.median(boot_width):.1%}")
        print(f"  Median (bootstrap - analytic) width diff: {np.median(boot_width - an_width):+.1%}")

        # coverage check: does the ANALYTIC interval contain the bootstrap median?
        # (a rough calibration signal -- are the two methods centered consistently)
        covered = (boot_lo >= an_lo - 1e-9) & (boot_hi <= an_hi + 1e-9)
        print(f"  Bootstrap interval fully INSIDE analytic interval: {covered.mean():.1%} of pitchers")
        wider = boot_width > an_width
        print(f"  Bootstrap WIDER than analytic: {wider.mean():.1%} of pitchers")

        # by PA tercile
        d = pd.DataFrame({"pa": pa, "an_width": an_width, "boot_width": boot_width})
        d["pa_bin"] = pd.qcut(d["pa"], 3, labels=["low PA", "mid PA", "high PA"])
        print("  By PA tercile (median width):")
        for b, g in d.groupby("pa_bin", observed=True):
            print(f"    {b}: analytic={g['an_width'].median():.1%}  bootstrap={g['boot_width'].median():.1%}")
