"""
fl_projection_diff_report.py -- for a given model-spec change, reports the
biggest INDIVIDUAL projection differences on the real 2027 projection build
(full 2025+2026 features, 2026-only naive prior/PA -- matches
build_2027_projections exactly, including its override of the naive prior
and PA to the 2026-only window, a previously-validated finding -- see
DECISION_LOG.md "Flat-pool window size"). Meant to be rerun for each
iteration that changes the production spec, split three ways:
  1. overall biggest movers (K% and BB% separately)
  2. LOW-HISTORY players (PA < that metric's own n0 -- the naive prior gets
     less than 50% weight here, so these are the players most exposed to
     whatever regressors changed)
  3. ESTABLISHED players (PA >= n0 -- their own track record already
     dominates, so a real mover here is more surprising/notable)

Usage:
    python fl_projection_diff_report.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

# Edit these two for whatever iteration is being compared.
OLD_SPEC = {"K": ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level"],
            "BB": bt.MODEL_SPECS["naive+other_league+surprise_prior"]["BB"]}
NEW_SPEC = bt.MODEL_SPECS["naive+other_league+surprise_prior"]


def build_2027_feat(raw):
    """Mirrors fl_production_projection.build_2027_projections' feature
    construction exactly: full 2025+2026 for bridge/wh_sva/spin/extension,
    but naive_K_rate/naive_BB_rate/PA overridden to the 2026-ONLY window."""
    feat, n0_k, n0_bb = bt.assemble_features(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.END_2026)],
                                              with_changepoints=False)
    naive_2026, n0_k, n0_bb = bt.naive_prior(raw, [(bt.START_2026, bt.END_2026)])
    feat = feat.drop(columns=["naive_K_rate", "naive_BB_rate", "PA"]).merge(
        naive_2026[["pitcher_id", "PA", "naive_K_rate", "naive_BB_rate"]], on="pitcher_id", how="left")
    names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    feat = feat.merge(names, on="pitcher_id", how="left")
    return feat, n0_k, n0_bb


def fit_and_predict(df_tr, feat_2027, target_col, cols, n0):
    d_tr = df_tr.dropna(subset=[target_col] + cols)
    w_tr = mm.reliability_weight(d_tr["PA_test"], n0).values
    m = bt.fit_weighted(d_tr, target_col, cols, w_tr)
    d = feat_2027.dropna(subset=cols).copy()
    d["pred"] = m.predict(d)
    return d[["pitcher_id", "pred"]]


def report_segment(d, label, diff_col="diff_K", pred_old="proj_K_old", pred_new="proj_K_new",
                    extra_cols=()):
    if len(d) == 0:
        print(f"  ({label}: no players in this segment)")
        return
    print(f"\n--- {label} (n={len(d)}) ---")
    print(f"  mean diff: {d[diff_col].mean():+.4f}   mean |diff|: {d[diff_col].abs().mean():.4f}   "
          f"median |diff|: {d[diff_col].abs().median():.4f}   max |diff|: {d[diff_col].abs().max():.4f}")
    cols = ["pitcher", "PA", pred_old, pred_new, diff_col] + list(extra_cols)
    top = d.reindex(d[diff_col].abs().sort_values(ascending=False).index).head(8)
    fmt = {pred_old: "{:.3f}".format, pred_new: "{:.3f}".format, diff_col: "{:+.3f}".format}
    for c in extra_cols:
        fmt[c] = "{:.2f}".format
    print(top[cols].to_string(index=False, formatters=fmt))


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    feat_2027, n0_k_live, n0_bb_live = build_2027_feat(raw)

    old_k = fit_and_predict(df_tr, feat_2027, "K_rate_actual", OLD_SPEC["K"], n0_k).rename(columns={"pred": "proj_K_old"})
    new_k = fit_and_predict(df_tr, feat_2027, "K_rate_actual", NEW_SPEC["K"], n0_k).rename(columns={"pred": "proj_K_new"})
    old_bb = fit_and_predict(df_tr, feat_2027, "BB_rate_actual", OLD_SPEC["BB"], n0_bb).rename(columns={"pred": "proj_BB_old"})
    new_bb = fit_and_predict(df_tr, feat_2027, "BB_rate_actual", NEW_SPEC["BB"], n0_bb).rename(columns={"pred": "proj_BB_new"})

    d = feat_2027[["pitcher_id", "pitcher", "PA", "spin_z_norm", "extension", "wh_sva_level",
                   "horiz_var", "vert_var", "has_college_match"]].copy()
    d = d.merge(old_k, on="pitcher_id", how="inner").merge(new_k, on="pitcher_id", how="inner")
    d = d.merge(old_bb, on="pitcher_id", how="inner").merge(new_bb, on="pitcher_id", how="inner")
    d["diff_K"] = d["proj_K_new"] - d["proj_K_old"]
    d["diff_BB"] = d["proj_BB_new"] - d["proj_BB_old"]

    print(f"\nn={len(d)} pitchers scored (n0_k={n0_k:.1f}, n0_bb={n0_bb:.1f}, live-window PA used for the split)")

    print("\n" + "=" * 100)
    print(f"K% -- OVERALL biggest movers")
    print("=" * 100)
    report_segment(d, "all pitchers", "diff_K", "proj_K_old", "proj_K_new",
                    extra_cols=("spin_z_norm", "extension"))

    print("\n" + "=" * 100)
    print(f"K% -- LOW-HISTORY (PA < n0_k={n0_k:.1f}) vs ESTABLISHED (PA >= n0_k)")
    print("=" * 100)
    report_segment(d[d["PA"] < n0_k], "low-history", "diff_K", "proj_K_old", "proj_K_new",
                    extra_cols=("spin_z_norm", "extension"))
    report_segment(d[d["PA"] >= n0_k], "established", "diff_K", "proj_K_old", "proj_K_new",
                    extra_cols=("spin_z_norm", "extension"))

    print("\n" + "=" * 100)
    print(f"BB% -- OVERALL biggest movers")
    print("=" * 100)
    report_segment(d, "all pitchers", "diff_BB", "proj_BB_old", "proj_BB_new",
                    extra_cols=("horiz_var", "vert_var"))

    print("\n" + "=" * 100)
    print(f"BB% -- LOW-HISTORY (PA < n0_bb={n0_bb:.1f}) vs ESTABLISHED (PA >= n0_bb)")
    print("=" * 100)
    report_segment(d[d["PA"] < n0_bb], "low-history", "diff_BB", "proj_BB_old", "proj_BB_new",
                    extra_cols=("horiz_var", "vert_var"))
    report_segment(d[d["PA"] >= n0_bb], "established", "diff_BB", "proj_BB_old", "proj_BB_new",
                    extra_cols=("horiz_var", "vert_var"))

    print("\n--- segment sizes ---")
    print(f"low-history (K, PA<{n0_k:.0f}): {(d['PA'] < n0_k).sum()}   established: {(d['PA'] >= n0_k).sum()}")
    print(f"low-history (BB, PA<{n0_bb:.0f}): {(d['PA'] < n0_bb).sum()}   established: {(d['PA'] >= n0_bb).sum()}")
