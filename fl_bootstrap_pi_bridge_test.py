"""
fl_bootstrap_pi_bridge_test.py -- extends fl_bootstrap_pi_test.py's bootstrap
to ALSO resample and refit the college bridge on each iteration (previously
scoped out as "too expensive for a marginal gain" -- revisited because the
PA x regressor-value calibration grid showed the flat 1.25x K% correction
over-corrects high-PA pitchers while still under-correcting low-PA ones,
exactly the pattern you'd expect if bridge-estimation uncertainty (which
matters MORE for low-PA pitchers, who lean on the bridge since their own
naive prior carries little weight) is the missing piece.

Per iteration: (1) resample the college bridge's own n=152 real transition
pairs, refit k_model_b/bb_model_b; (2) apply that refit bridge to every
train/test pitcher's own (fixed) k_pct/bb_pct_multi to get a bootstrapped
fl_k_rate_bridge/fl_bb_rate_bridge; (3) resample the outer-regression
training pitchers, re-estimate n0, refit the outer WLS model using the
bootstrapped bridge values; (4) predict on the (unresampled) test set using
the SAME bootstrapped bridge values; (5) add empirical residual noise
(weighted-rescaled, per fl_bootstrap_pi_test.py's fix for binomial
undersimulation). This now has THREE resampled stages instead of one.

Usage:
    python fl_bootstrap_pi_bridge_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_college_bridge as bridge
import mlb_marcel as mm

N_BOOT = 300
RNG = np.random.default_rng(42)
SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]


def logit(p):
    return np.log(p / (1 - p))


def logit_ci(p, se, z):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    se_logit = se / (p * (1 - p))
    lp = logit(p)
    return 1 / (1 + np.exp(-(lp - z * se_logit))), 1 / (1 + np.exp(-(lp + z * se_logit)))


def build_bridge_fitting_data():
    """Same construction fl_college_bridge.fit_bridge uses internally, but
    returning the FULL merged frame (pitcher_id, k_pct, bb_pct_multi,
    k_rate, bb_rate) so it can be resampled here -- fit_bridge itself only
    returns the fitted models, not this intermediate table."""
    d = pd.read_parquet(bridge.BRIDGE_PATH).drop_duplicates("pitcher_id").copy()
    college_years = bridge.load_all_college_years()
    rows = []
    for _, r in d.iterrows():
        feats = bridge._latest_and_multi(college_years, r["name_clean"], as_of_season=r["college_year"])
        if feats is None:
            continue
        rows.append({"pitcher_id": r["pitcher_id"], "bb_pct_multi": feats["bb_pct_multi"]})
    feat_df = pd.DataFrame(rows)
    d = d.merge(feat_df, on="pitcher_id", how="inner")
    return d[["pitcher_id", "k_pct", "bb_pct_multi", "k_rate", "bb_rate"]]


def attach_college_inputs(df, bridge_fit_data):
    """Attach each pitcher's OWN (fixed) k_pct/bb_pct_multi, for pitchers who
    are in the bridge's fitting sample -- lets us re-predict their bridge
    rate under any refit bridge model without re-deriving the college match
    itself each iteration (that part doesn't change, only the bridge
    model's COEFFICIENTS do)."""
    return df.merge(bridge_fit_data[["pitcher_id", "k_pct", "bb_pct_multi"]], on="pitcher_id", how="left")


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)],
                                      [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    print("Building college bridge fitting data (n=152 real transition pairs)...")
    bridge_data = build_bridge_fitting_data()
    n_bridge = len(bridge_data)
    print(f"  n={n_bridge}")

    lg_k_rate = df_tr["fl_k_rate_bridge"].mode().iloc[0] if False else None  # unused, kept for clarity

    results = {}
    for metric, target_col, spec, n0_metric, bridge_out_col, bridge_in_col in [
        ("K%", "K_rate_actual", SPEC["K"], n0_k, "fl_k_rate_bridge", "k_pct"),
        ("BB%", "BB_rate_actual", SPEC["BB"], n0_bb, "fl_bb_rate_bridge", "bb_pct_multi"),
    ]:
        d_tr_full = df_tr.dropna(subset=[target_col] + spec).reset_index(drop=True)
        d_te = df_te.dropna(subset=[target_col] + spec).reset_index(drop=True)
        d_tr_full = attach_college_inputs(d_tr_full, bridge_data)
        d_te = attach_college_inputs(d_te, bridge_data)
        n_tr = len(d_tr_full)
        n_te = len(d_te)
        sims = np.full((N_BOOT, n_te), np.nan)
        print(f"\nRunning {N_BOOT}-iteration bridge+outer bootstrap for {metric}...")

        bridge_formula = f"{target_col.replace('_actual','')} ~ {bridge_in_col}" if False else None
        bridge_y_col = "k_rate" if bridge_in_col == "k_pct" else "bb_rate"

        for b in range(N_BOOT):
            # stage 1: resample + refit the COLLEGE BRIDGE
            bidx = RNG.integers(0, n_bridge, size=n_bridge)
            boot_bridge = bridge_data.iloc[bidx]
            try:
                bmodel = smf.ols(f"{bridge_y_col} ~ {bridge_in_col}", data=boot_bridge).fit()
            except Exception:
                continue

            tr = d_tr_full.copy()
            te = d_te.copy()
            for frame in (tr, te):
                has_match = frame[bridge_in_col].notna()
                pred_bridge = bmodel.predict(frame.loc[has_match])
                frame.loc[has_match, bridge_out_col] = pred_bridge.values
                # unmatched pitchers keep their existing fallback value (league
                # average) unaffected by the bridge refit, same as production

            # stage 2: resample + refit n0 + OUTER REGRESSION
            tidx = RNG.integers(0, n_tr, size=n_tr)
            boot_tr = tr.iloc[tidx]
            n0_b, _ = mm.estimate_n0(boot_tr[target_col], boot_tr["PA_test"], min_n=bt.MIN_PA_N0)
            w_tr_b = mm.reliability_weight(boot_tr["PA_test"], n0_b).values
            try:
                m_b = bt.fit_weighted(boot_tr, target_col, spec, w_tr_b)
            except Exception:
                continue

            # stage 3: empirical residual noise (weighted-rescaled), per the
            # overdispersion fix validated in fl_bootstrap_pi_test.py
            resid_b = boot_tr[target_col].values - m_b.predict(boot_tr).values
            z_b = resid_b * np.sqrt(w_tr_b)

            pred_te_b = np.clip(m_b.predict(te).values, 1e-6, 1 - 1e-6)
            w_te_b = mm.reliability_weight(te["PA_test"], n0_b).values
            donor_z = RNG.choice(z_b, size=n_te, replace=True)
            resid_sim = donor_z / np.sqrt(w_te_b)
            sims[b, :] = np.clip(pred_te_b + resid_sim, 0.0, 1.0)

        # current production analytic interval, for comparison
        w_tr_orig = mm.reliability_weight(d_tr_full[target_col].notna() * 0 + d_tr_full["PA_test"], n0_metric).values
        m_orig = bt.fit_weighted(d_tr_full, target_col, spec, w_tr_orig)
        w_te_orig = mm.reliability_weight(d_te["PA_test"], n0_metric).values
        pred_orig = np.clip(m_orig.predict(d_te).values, 1e-6, 1 - 1e-6)
        se_orig = m_orig.get_prediction(d_te, weights=w_te_orig).se_obs
        se_cal = se_orig * (1.25 if metric == "K%" else 1.0)

        actual = d_te[target_col].clip(1e-6, 1 - 1e-6).values
        boot_lo = np.nanpercentile(sims, 5, axis=0)
        boot_hi = np.nanpercentile(sims, 95, axis=0)
        boot_cov = ((actual >= boot_lo) & (actual <= boot_hi)).mean()
        an_lo, an_hi = logit_ci(pred_orig, se_cal, 1.6449)
        an_cov = ((actual >= an_lo) & (actual <= an_hi)).mean()

        print(f"\n--- {metric} (n={n_te}) ---")
        print(f"  Analytic (current production, incl. any flat calibration) 90% coverage: {an_cov:.1%}")
        print(f"  Bridge+outer bootstrap 90% coverage: {boot_cov:.1%}")
        print(f"  Analytic median width: {np.median(an_hi - an_lo):.1%}")
        print(f"  Bootstrap median width: {np.median(boot_hi - boot_lo):.1%}")

        # by PA tercile
        d = pd.DataFrame({"pa": d_te["PA_test"].values, "boot_lo": boot_lo, "boot_hi": boot_hi,
                           "an_lo": an_lo, "an_hi": an_hi, "actual": actual})
        d["covered_boot"] = (d["actual"] >= d["boot_lo"]) & (d["actual"] <= d["boot_hi"])
        d["covered_an"] = (d["actual"] >= d["an_lo"]) & (d["actual"] <= d["an_hi"])
        d["pa_bin"] = pd.qcut(d["pa"], 2, labels=["low PA", "high PA"])
        print("  By PA half (coverage, analytic vs bridge-bootstrap):")
        for bn, g in d.groupby("pa_bin", observed=True):
            print(f"    {bn}: n={len(g)}  analytic={g['covered_an'].mean():.1%}  "
                  f"bridge-bootstrap={g['covered_boot'].mean():.1%}")
