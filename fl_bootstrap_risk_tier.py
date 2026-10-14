"""
fl_bootstrap_risk_tier.py -- production-ready PA-stratified bootstrap +
Medium/High risk tier, replacing the parametric logit-delta-method K-BB% CI.

Validated this session (see DECISION_LOG.md "PA-stratified bootstrap
replaces the parametric K% CI"):
  - The parametric CI rested on two violated assumptions: residuals aren't
    normal (skew=+0.40, excess kurtosis=+2.03) and the flat
    K_SE_CALIBRATION=1.25 correction isn't uniform across PA (needed
    multiplier ranged 0.90x-1.30x by bucket).
  - A PA-stratified empirical bootstrap (case-resample + refit for parameter
    uncertainty; PA-matched donor residuals from a richer early-cutoff pool
    for outcome noise) fixed both: coverage lands within a few points of
    nominal at every PA bucket AND every confidence level tested
    (50/67/80/90/95%).
  - Reporting a numeric interval was itself found impractical: even at 90%
    confidence the interval is ~3x the real population SD of K-BB% wide,
    i.e. it swallows most of the roster's real talent spread in one
    interval. Settled on a Medium/High risk LABEL instead (nothing in this
    population is "low risk" -- even the best-sampled players' intervals
    rarely drop below ~1x population SD), derived from the ACTUAL K-BB%
    bootstrap width (running K% and BB% bootstraps together, correlated via
    shared resampling draws, rather than K%'s width alone, which was the
    wrong metric for a K-BB%-denominated threshold).

Usage:
    import fl_bootstrap_risk_tier as brt
    tiers = brt.compute_risk_tiers(raw, m_k, m_bb, d_tr_k, d_tr_bb, targets, n0_k, n0_bb)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

N_BOOT = 400
RNG_SEED = 42
PA_EDGES = [0, 5, 10, 20, 40, 80, 99999]
PA_LABELS = ["1-5", "6-10", "11-20", "21-40", "41-80", "80+"]
CUTOFF_WEEKS = [1, 2, 3, 5, 8, 12]
CI_LEVEL = (25, 75)  # 50% interval -- see DECISION_LOG.md for why 50%, not 90%
POP_STD_KBB = 0.0715  # real K-BB% population std at PA>=100 (fl_pitcher_features population, 2025+2026) --
# NOT used as the risk-tier threshold (see below): checked directly once K%
# and BB% uncertainty are combined correctly (K_sim - BB_sim, not K alone),
# every roster player's width exceeds this -- a true but non-differentiating
# finding. The tier threshold uses the roster's own median width instead.


def assign_bucket(pa_series):
    return pd.cut(pa_series, PA_EDGES, labels=PA_LABELS)


def _naive_from_window(raw, start, end, n0_k, n0_bb, lg_k, lg_bb):
    kbb = fp.build_period_kbb(fp.filter_period(raw, start, end), min_pa=1)
    kbb["naive_K_rate"] = (kbb["K"] + n0_k * lg_k) / (kbb["PA"] + n0_k)
    kbb["naive_BB_rate"] = (kbb["BB"] + n0_bb * lg_bb) / (kbb["PA"] + n0_bb)
    return kbb[["pitcher_id", "PA", "naive_K_rate", "naive_BB_rate"]]


def _build_fold(raw, early_start, early_end, surprise_windows, label_start, label_end,
                 n0_k, n0_bb, lg_k, lg_bb):
    naive = _naive_from_window(raw, early_start, early_end, n0_k, n0_bb, lg_k, lg_bb)
    br = bt.bridge_features(naive["pitcher_id"], raw, current_year=int(early_end[:4]))
    df = naive.merge(br, on="pitcher_id", how="left")
    df["fl_k_rate_bridge"] = df["fl_k_rate_bridge"].fillna(lg_k)
    df["fl_bb_rate_bridge"] = df["fl_bb_rate_bridge"].fillna(lg_bb)

    surprise_pool = pd.concat([fp.filter_period(raw, s, e) for s, e in surprise_windows], ignore_index=True)
    wh_level, _ = bt.wh_sva_level_and_changepoint(surprise_pool)
    loc_var = bt.loc_var_levels(surprise_pool)
    stuff = bt.stuff_levels(surprise_pool)
    df = df.merge(wh_level, on="pitcher_id", how="left").merge(loc_var, on="pitcher_id", how="left")
    df = df.merge(stuff, on="pitcher_id", how="left")
    for c in ["wh_sva_level", "horiz_var", "vert_var", "command_score", "spin_z_norm", "extension"]:
        df[c] = df[c].fillna(df[c].mean())

    actual = bt.actual_rate(raw, [(label_start, label_end)])
    df = df.merge(actual, on="pitcher_id", how="inner")
    return df


def build_donor_pool(raw, n0_k, n0_bb) -> pd.DataFrame:
    """Richer, PA-diverse early-2025-cutoff pool used as the residual donor
    source (the real production training transition is almost entirely
    high-PA players -- confirmed directly: only 1-2 donors in the lowest PA
    buckets -- so it can't supply the low-PA tail with real variability)."""
    full_pool = pd.concat([fp.filter_period(raw, bt.START_2025, bt.END_2025),
                           fp.filter_period(raw, bt.START_2026, bt.END_2026)], ignore_index=True)
    term = full_pool[full_pool["is_terminal"]]
    lg_k = (term["k_or_bb"] == "strikeout").mean()
    lg_bb = (term["k_or_bb"] == "walk").mean()

    frames = []
    for wk in CUTOFF_WEEKS:
        cutoff = (pd.Timestamp(bt.START_2025) + pd.Timedelta(weeks=wk)).strftime("%Y-%m-%d")
        day_after = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if pd.Timestamp(day_after) >= pd.Timestamp(bt.END_2025) - pd.Timedelta(weeks=2):
            continue
        fold = _build_fold(raw, bt.START_2025, cutoff, [(bt.START_2025, cutoff)], day_after, bt.END_2025,
                            n0_k, n0_bb, lg_k, lg_bb)
        frames.append(fold)
    pool = pd.concat(frames, ignore_index=True)
    pool["pa_bucket"] = assign_bucket(pool["PA"])
    return pool


def _bootstrap_sims(d_tr, targets, target_col, spec, n0, donor_pool, rng):
    n_tr, n_donor, n_te = len(d_tr), len(donor_pool), len(targets)
    sims = np.full((N_BOOT, n_te), np.nan)
    tr_idx_draws = np.empty((N_BOOT, n_tr), dtype=int)
    donor_idx_draws = np.empty((N_BOOT, n_donor), dtype=int)

    for b in range(N_BOOT):
        tr_idx_draws[b] = rng.integers(0, n_tr, size=n_tr)
        donor_idx_draws[b] = rng.integers(0, n_donor, size=n_donor)
        boot_tr = d_tr.iloc[tr_idx_draws[b]]
        w_tr_b = mm.reliability_weight(boot_tr["PA_test"], n0).values
        try:
            m_b = bt.fit_weighted(boot_tr, target_col, spec, w_tr_b)
        except Exception:
            continue

        boot_donor = donor_pool.iloc[donor_idx_draws[b]].copy()
        boot_donor["resid"] = boot_donor[target_col].values - m_b.predict(boot_donor).values

        pred_te_b = np.clip(m_b.predict(targets).values, 1e-6, 1 - 1e-6)
        resid_sim = np.empty(n_te)
        for bucket in PA_LABELS:
            mask = (targets["pa_bucket"] == bucket).values
            if not mask.any():
                continue
            donors = boot_donor.loc[boot_donor["pa_bucket"] == bucket, "resid"]
            if donors.empty:
                donors = boot_donor["resid"]
            resid_sim[mask] = rng.choice(donors.values, size=mask.sum(), replace=True)
        sims[b, :] = np.clip(pred_te_b + resid_sim, 0.0, 1.0)
    return sims


def compute_risk_tiers(raw, d_tr_k, d_tr_bb, targets, n0_k, n0_bb,
                        k_spec, bb_spec) -> pd.DataFrame:
    """Runs K% and BB% PA-stratified bootstraps TOGETHER (same RNG stream,
    so each iteration's case/donor draws align across the two), derives
    per-iteration K-BB% = K_sim - BB_sim, and returns a risk tier per
    pitcher_id based on the ACTUAL K-BB% interval width -- not K%'s width
    alone, which is the wrong metric for a K-BB%-denominated threshold."""
    targets = targets.copy()
    targets["pa_bucket"] = assign_bucket(targets["PA"])

    donor_pool = build_donor_pool(raw, n0_k, n0_bb)

    rng_k = np.random.default_rng(RNG_SEED)
    rng_bb = np.random.default_rng(RNG_SEED)  # SAME seed -> same draw sequence -> correlated iterations
    sims_k = _bootstrap_sims(d_tr_k, targets, "K_rate_actual", k_spec, n0_k, donor_pool, rng_k)
    sims_bb = _bootstrap_sims(d_tr_bb, targets, "BB_rate_actual", bb_spec, n0_bb, donor_pool, rng_bb)
    sims_kbb = sims_k - sims_bb

    lo, hi = np.nanpercentile(sims_kbb, CI_LEVEL, axis=0)
    width = hi - lo

    # per-metric widths too -- lets a caller see WHICH metric is driving a
    # player's combined risk (K% and BB% uncertainty don't always move
    # together -- see DECISION_LOG.md)
    k_lo, k_hi = np.nanpercentile(sims_k, CI_LEVEL, axis=0)
    bb_lo, bb_hi = np.nanpercentile(sims_bb, CI_LEVEL, axis=0)
    k_width = k_hi - k_lo
    bb_width = bb_hi - bb_lo

    # threshold: the roster's OWN median width, not the absolute population-SD
    # benchmark (POP_STD_KBB) -- checked directly: once K% and BB% uncertainty
    # are combined correctly, every player's width exceeds that absolute
    # benchmark (min width 0.078 > 0.0715), so an absolute cutoff can't
    # differentiate anyone. A relative (median) split is the only one that
    # produces a meaningful High/Medium distinction within this population.
    threshold = np.nanmedian(width)

    # NaN width (a target missing a required regressor, e.g. a 2025-only
    # pitcher with no 2026 naive rate -- no projection exists at all for
    # them) must stay NaN here, not silently default to "Medium" the way
    # np.where(width >= threshold, ...) would (NaN comparisons are False)
    tier = np.where(np.isnan(width), None, np.where(width >= threshold, "High", "Medium"))

    out = pd.DataFrame({
        "pitcher_id": targets["pitcher_id"].values,
        "kbb_ci_width_50": width,
        "kbb_ci_lo_50": lo,
        "kbb_ci_hi_50": hi,
        "k_ci_width_50": k_width,
        "bb_ci_width_50": bb_width,
        "risk_tier": tier,
    })
    return out
