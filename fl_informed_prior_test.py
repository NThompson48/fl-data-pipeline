"""
fl_informed_prior_test.py -- tests shrinking the naive prior toward a
wh_sva/horiz_dist/vert_dist-INFORMED target instead of the flat league average.

Standard empirical-Bayes naive prior:
    naive_rate = (K + n0 * league_avg_rate) / (PA + n0)

This version instead shrinks toward a per-pitcher expected rate predicted from
their own surprise variables:
    informed_target = a + b * wh_sva_level                    (K%)
    informed_target = a + b1 * horiz_dist + b2 * vert_dist     (BB%)
    naive_rate_v2 = (K + n0 * informed_target) / (PA + n0)

This automatically gives the surprise variable MORE influence for a low-PA
pitcher (shrinkage weight n0/(PA+n0) is large) and LESS for a well-established
one (their own raw rate dominates) -- the same property an explicit interaction
term was trying to achieve (see DECISION_LOG.md "Surprise-variable x
unreliability interaction"), but built into the shrinkage mechanism itself
rather than added as a second-order regressor.

`informed_k_model`/`informed_bb_model` are fit ONCE on the training window only
(never the test window), then applied frozen to the test window's own surprise
variables -- same no-leakage discipline as everywhere else in this project.

Compares three outer-model variants via the same held-out Experiment-2 check:
  A: naive(flat) + bridge + wh_sva separately        (current production design)
  B: naive_v2(informed) + bridge                     (wh_sva folded into the prior, not separate)
  C: naive_v2(informed) + bridge + wh_sva separately  (both -- any residual value left over?)

Usage:
    python fl_informed_prior_test.py
"""
from __future__ import annotations

import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

TRAIN_WINDOWS = [(bt.START_2025, bt.END_2025)]
TEST_WINDOWS = [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)]


def raw_kbb_for_window(raw, windows):
    pool = pd.concat([fp.filter_period(raw, s, e) for s, e in windows], ignore_index=True)
    return fp.build_period_kbb(pool, min_pa=1)[["pitcher_id", "PA", "K", "BB"]]


def build_informed_naive(df_tr, df_te, raw, n0_k, n0_bb):
    raw_tr = raw_kbb_for_window(raw, TRAIN_WINDOWS)
    raw_te = raw_kbb_for_window(raw, TEST_WINDOWS)
    df_tr = df_tr.merge(raw_tr, on="pitcher_id", how="left", suffixes=("", "_raw"))
    df_te = df_te.merge(raw_te, on="pitcher_id", how="left", suffixes=("", "_raw"))

    # fit the informed-target regressions on TRAIN only
    d_k = df_tr.dropna(subset=["K_rate_actual", "wh_sva_level"])
    w_k = mm.reliability_weight(d_k["PA_test"], n0_k).values
    informed_k_model = bt.fit_weighted(d_k, "K_rate_actual", ["wh_sva_level"], w_k)

    d_bb = df_tr.dropna(subset=["BB_rate_actual", "horiz_dist", "vert_dist"])
    w_bb = mm.reliability_weight(d_bb["PA_test"], n0_bb).values
    informed_bb_model = bt.fit_weighted(d_bb, "BB_rate_actual", ["horiz_dist", "vert_dist"], w_bb)

    print("\nInformed-target models (fit on train window only):")
    print("K%  informed_target = ", informed_k_model.params.to_dict())
    print("BB% informed_target = ", informed_bb_model.params.to_dict())

    for df in (df_tr, df_te):
        df["informed_k_target"] = informed_k_model.predict(df)
        df["informed_bb_target"] = informed_bb_model.predict(df)
        df["naive_K_rate_v2"] = (df["K"] + n0_k * df["informed_k_target"]) / (df["PA"] + n0_k)
        df["naive_BB_rate_v2"] = (df["BB"] + n0_bb * df["informed_bb_target"]) / (df["PA"] + n0_bb)
    return df_tr, df_te


SPECS = {
    "A: naive(flat) + bridge + wh_sva": {
        "K": ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level"],
        "BB": ["naive_BB_rate", "fl_bb_rate_bridge", "horiz_dist", "vert_dist"]},
    "B: naive_v2(informed) + bridge": {
        "K": ["naive_K_rate_v2", "fl_k_rate_bridge"],
        "BB": ["naive_BB_rate_v2", "fl_bb_rate_bridge"]},
    "C: naive_v2(informed) + bridge + wh_sva": {
        "K": ["naive_K_rate_v2", "fl_k_rate_bridge", "wh_sva_level"],
        "BB": ["naive_BB_rate_v2", "fl_bb_rate_bridge", "horiz_dist", "vert_dist"]},
}

if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print("\nAssembling Experiment-2 train/test panels...")
    df_tr, n0_k, n0_bb = bt.assemble(raw, TRAIN_WINDOWS, [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, TEST_WINDOWS, [(bt._DAY_AFTER_2026_CUT, bt.END_2026)],
                               with_changepoints=False)

    df_tr, df_te = build_informed_naive(df_tr, df_te, raw, n0_k, n0_bb)

    print("\n" + "=" * 70)
    print("Held-out test: flat-mean prior vs. wh_sva/dist-informed prior")
    print("=" * 70)
    results = []
    for name, spec in SPECS.items():
        k_res = bt.run_heldout(df_tr, "K_rate_actual", spec["K"], df_te, "K_rate_actual",
                                n0_k, f"{name} [K%]")
        bb_res = bt.run_heldout(df_tr, "BB_rate_actual", spec["BB"], df_te, "BB_rate_actual",
                                 n0_bb, f"{name} [BB%]")
        results += [k_res, bb_res]
    print(pd.DataFrame(results).to_string(index=False))
