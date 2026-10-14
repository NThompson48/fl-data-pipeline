"""fl_arsenal_interaction_test.py -- does how a pitcher's pitch TYPES play off
each other (not just their individual quality) explain K%/BB%, beyond the
existing model? Two concrete, well-motivated mechanisms tested, both
genuinely held out via the same Experiment 2 harness every other feature in
this project has been judged against:

1. Velocity differential (fastball mean effective_velo - offspeed/breaking
   mean effective_velo) -- the standard "tunneling"/deception-via-speed-
   change mechanism. A DIFFERENT mechanism from the already-tested release-
   point consistency-across-arsenal ("Release point as a K% regressor" in
   DECISION_LOG.md), which found no signal for a location-based deception
   proxy -- this tests the speed-based version instead of assuming the same
   null result carries over.
2. Arsenal depth (# distinct pitch types with >=5% usage and >=25 pitches
   in the window) -- pitch-mix diversity/unpredictability, mirroring the
   MLB-side definition in mlb_arsenal_depth_wh_sva_aging.py (that script's
   own finding was about an AGING interaction, never tested here as a
   direct FL K%/BB% regressor).

Usage:
    python fl_arsenal_interaction_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

MIN_USAGE_PCT = 0.05
MIN_USAGE_N = 25


def compute_arsenal_features(period_df: pd.DataFrame) -> pd.DataFrame:
    d = period_df.dropna(subset=["pitch_type", "effective_velo"]).copy()
    d = d[~d["pitch_type"].isin(["", "not_him", "other"])]
    d["is_fb"] = d["pitch_type"].isin(fp.FASTBALL_TYPES)

    counts = d.groupby(["pitcher_id", "pitch_type"]).agg(
        n=("effective_velo", "size"), velo=("effective_velo", "mean")).reset_index()
    totals = counts.groupby("pitcher_id")["n"].sum().rename("total").reset_index()
    counts = counts.merge(totals, on="pitcher_id")
    counts["pct"] = counts["n"] / counts["total"]
    counts["qualifies"] = (counts["pct"] >= MIN_USAGE_PCT) & (counts["n"] >= MIN_USAGE_N)
    counts["is_fb"] = counts["pitch_type"].isin(fp.FASTBALL_TYPES)

    q = counts[counts["qualifies"]]
    depth = q.groupby("pitcher_id").size().rename("arsenal_depth")

    fb_velo = (q[q["is_fb"]].groupby("pitcher_id")
               .apply(lambda g: np.average(g["velo"], weights=g["n"]), include_groups=False)
               .rename("fb_velo"))
    off = q[~q["is_fb"]]
    off_velo = (off.groupby("pitcher_id")
                .apply(lambda g: np.average(g["velo"], weights=g["n"]), include_groups=False)
                .rename("off_velo"))

    out = pd.concat([depth, fb_velo, off_velo], axis=1).reset_index()
    out["velo_diff"] = out["fb_velo"] - out["off_velo"]
    return out[["pitcher_id", "arsenal_depth", "velo_diff"]]


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print("Computing arsenal features over the TRAIN window (full 2025)...")
    train_period = fp.filter_period(raw, bt.START_2025, bt.END_2025)
    arsenal = compute_arsenal_features(train_period)
    arsenal["pitcher_id"] = arsenal["pitcher_id"].astype(str)
    print(f"  {arsenal['velo_diff'].notna().sum()} pitchers with both FB and offspeed velo; "
          f"depth distribution:\n{arsenal['arsenal_depth'].value_counts().sort_index()}")
    print(f"  velo_diff describe:\n{arsenal['velo_diff'].describe()}")

    print("\nAssembling Experiment-2 train/test panels...")
    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    lg_depth = arsenal["arsenal_depth"].median()
    lg_velo_diff = arsenal["velo_diff"].median()
    for df in (df_tr, df_te):
        df["pitcher_id"] = df["pitcher_id"].astype(str)
        df["arsenal_depth"] = df["pitcher_id"].map(arsenal.set_index("pitcher_id")["arsenal_depth"]).fillna(lg_depth)
        df["velo_diff"] = df["pitcher_id"].map(arsenal.set_index("pitcher_id")["velo_diff"]).fillna(lg_velo_diff)

    SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]
    results = []
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"], df_te, "K_rate_actual", n0_k, "K% baseline"))
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["velo_diff"], df_te, "K_rate_actual", n0_k,
                                   "K% + velo_diff"))
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["arsenal_depth"], df_te, "K_rate_actual", n0_k,
                                   "K% + arsenal_depth"))
    results.append(bt.run_heldout(df_tr, "K_rate_actual", SPEC["K"] + ["velo_diff", "arsenal_depth"], df_te,
                                   "K_rate_actual", n0_k, "K% + both"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"], df_te, "BB_rate_actual", n0_bb, "BB% baseline"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"] + ["velo_diff"], df_te, "BB_rate_actual", n0_bb,
                                   "BB% + velo_diff"))
    results.append(bt.run_heldout(df_tr, "BB_rate_actual", SPEC["BB"] + ["arsenal_depth"], df_te, "BB_rate_actual",
                                   n0_bb, "BB% + arsenal_depth"))
    print()
    print(pd.DataFrame(results).to_string(index=False))

    w_k = mm.reliability_weight(df_tr["PA"], n0_k)
    m_k = smf.wls("K_rate_actual ~ " + " + ".join(SPEC["K"] + ["velo_diff", "arsenal_depth"]), data=df_tr,
                   weights=w_k).fit()
    print(f"\nK% combined model: velo_diff coef={m_k.params['velo_diff']:.4f} p={m_k.pvalues['velo_diff']:.4f}  "
          f"arsenal_depth coef={m_k.params['arsenal_depth']:.4f} p={m_k.pvalues['arsenal_depth']:.4f}")

    w_bb = mm.reliability_weight(df_tr["PA"], n0_bb)
    m_bb = smf.wls("BB_rate_actual ~ " + " + ".join(SPEC["BB"] + ["velo_diff", "arsenal_depth"]), data=df_tr,
                    weights=w_bb).fit()
    print(f"BB% combined model: velo_diff coef={m_bb.params['velo_diff']:.4f} p={m_bb.pvalues['velo_diff']:.4f}  "
          f"arsenal_depth coef={m_bb.params['arsenal_depth']:.4f} p={m_bb.pvalues['arsenal_depth']:.4f}")
