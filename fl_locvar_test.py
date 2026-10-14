"""
fl_locvar_test.py -- tests horiz_var/vert_var (location VARIANCE, command
consistency -- fl_pitcher_features.compute_loc_var) as a REPLACEMENT for
horiz_dist/vert_dist (location DISTANCE-FROM-CENTER, command accuracy) in the
BB% model, both as a plain regressor swap and as the informed-prior shrinkage
target (same mechanism as fl_informed_prior_test.py, which tried this with
horiz_dist/vert_dist and LOST: BB% A=0.3274 vs B=0.3186/C=0.3160).

Motivation (see DECISION_LOG.md "Location variance as a BB% regressor/
informed-prior input"): horiz_var/vert_var beat horiz_dist/vert_dist on every
axis checked (held-out corr, stickiness) while being highly correlated with
them (r=0.85/0.87) -- reading as a cleaner measurement of the same command
trait rather than an independent one. Worth re-testing the informed-prior
mechanism with the stronger variable, since the earlier BB% informed-prior
loss was reasoned to be partly about horiz_dist/vert_dist being too noisy
relative to what the outer regression could already extract directly.

Four variants, all BB% only (K% is untouched -- these variables were never
candidates there):
  A: naive(flat) + bridge + horiz_dist + vert_dist        (current production)
  B: naive(flat) + bridge + horiz_var + vert_var           (plain replacement)
  C: naive_v2(informed via horiz_var/vert_var) + bridge    (informed prior only)
  D: naive_v2(informed via horiz_var/vert_var) + bridge + horiz_var + vert_var

Usage:
    python fl_locvar_test.py
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


def with_loc_var(df: pd.DataFrame, raw: pd.DataFrame, windows: list[tuple[str, str]]) -> pd.DataFrame:
    pool = pd.concat([fp.filter_period(raw, s, e) for s, e in windows], ignore_index=True)
    lv = fp.compute_loc_var(pool)[["pitcher_id", "horiz_var", "vert_var"]]
    out = df.merge(lv, on="pitcher_id", how="left")
    # same neutral-imputation convention bt.assemble_features already uses for
    # wh_sva_level/horiz_dist/vert_dist -- missing = training population's own mean
    out["horiz_var"] = out["horiz_var"].fillna(out["horiz_var"].mean())
    out["vert_var"] = out["vert_var"].fillna(out["vert_var"].mean())
    return out


def build_informed_naive_bb(df_tr, df_te, raw, n0_bb):
    raw_tr = raw_kbb_for_window(raw, TRAIN_WINDOWS)
    raw_te = raw_kbb_for_window(raw, TEST_WINDOWS)
    df_tr = df_tr.merge(raw_tr, on="pitcher_id", how="left", suffixes=("", "_raw"))
    df_te = df_te.merge(raw_te, on="pitcher_id", how="left", suffixes=("", "_raw"))

    # fit the informed-target regression on TRAIN only
    d_bb = df_tr.dropna(subset=["BB_rate_actual", "horiz_var", "vert_var"])
    w_bb = mm.reliability_weight(d_bb["PA_test"], n0_bb).values
    informed_bb_model = bt.fit_weighted(d_bb, "BB_rate_actual", ["horiz_var", "vert_var"], w_bb)
    print("\nInformed-target model (fit on train window only):")
    print("BB% informed_target =", informed_bb_model.params.to_dict())

    for df in (df_tr, df_te):
        df["informed_bb_target"] = informed_bb_model.predict(df)
        df["naive_BB_rate_v2"] = (df["BB"] + n0_bb * df["informed_bb_target"]) / (df["PA"] + n0_bb)
    return df_tr, df_te


SPECS = {
    "A: naive(flat) + bridge + horiz_dist + vert_dist (current production)": {
        "BB": ["naive_BB_rate", "fl_bb_rate_bridge", "horiz_dist", "vert_dist"]},
    "B: naive(flat) + bridge + horiz_var + vert_var (replacement)": {
        "BB": ["naive_BB_rate", "fl_bb_rate_bridge", "horiz_var", "vert_var"]},
    "C: naive_v2(informed via var) + bridge": {
        "BB": ["naive_BB_rate_v2", "fl_bb_rate_bridge"]},
    "D: naive_v2(informed via var) + bridge + horiz_var + vert_var": {
        "BB": ["naive_BB_rate_v2", "fl_bb_rate_bridge", "horiz_var", "vert_var"]},
}

if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    print("\nAssembling Experiment-2 train/test panels...")
    df_tr, n0_k, n0_bb = bt.assemble(raw, TRAIN_WINDOWS, [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, TEST_WINDOWS, [(bt._DAY_AFTER_2026_CUT, bt.END_2026)],
                               with_changepoints=False)

    df_tr = with_loc_var(df_tr, raw, [(bt.START_2025, bt.END_2025)])
    df_te = with_loc_var(df_te, raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)])

    df_tr, df_te = build_informed_naive_bb(df_tr, df_te, raw, n0_bb)

    print("\n" + "=" * 70)
    print("Held-out test: horiz_dist/vert_dist vs. horiz_var/vert_var (plain swap & informed prior)")
    print("=" * 70)
    results = []
    for name, spec in SPECS.items():
        bb_res = bt.run_heldout(df_tr, "BB_rate_actual", spec["BB"], df_te, "BB_rate_actual",
                                 n0_bb, f"{name} [BB%]")
        results.append(bb_res)
    print(pd.DataFrame(results).to_string(index=False))

    print("\n--- Significance check: variant B (plain replacement), fit on train window ---")
    d = df_tr.dropna(subset=["BB_rate_actual"] + SPECS["B: naive(flat) + bridge + horiz_var + vert_var (replacement)"]["BB"])
    w = mm.reliability_weight(d["PA_test"], n0_bb).values
    m = bt.fit_weighted(d, "BB_rate_actual", SPECS["B: naive(flat) + bridge + horiz_var + vert_var (replacement)"]["BB"], w)
    print(m.summary().tables[1])
