"""
fl_release_point_test.py -- tests release point as a K% regressor, in two
distinct forms since they capture different mechanisms:
  (1) raw LEVEL (mean release_pos_z / rel_side_as per pitcher) -- like the
      spin_rate/extension tests, a simple per-pitcher average, somewhat
      pitch-mix confounded (release point varies modestly by pitch type:
      release_pos_z spans ~5.5-5.9ft, rel_side_as ~1.6-2.2ft across types).
  (2) DECEPTION/tunneling proxy: how CONSISTENT a pitcher's release point is
      ACROSS their different pitch types (usage-weighted variance of each
      pitch type's own mean release point). A hitter picking up pitch type
      from arm slot/release is the opposite of deception -- so a pitcher
      whose fastball and breaking ball come from nearly the same release
      point is harder to read. This is a BETWEEN-type measure, distinct
      from the WITHIN-pitch-type release-point variance already tested
      earlier this session (found weak for BB%, a command/consistency
      question, not a deception one).

Validated the same way as every other candidate: fit fold 1 (full-2025 ->
actual 1H2026), freeze, score fold 2 (2025+1H2026 -> actual 2H2026), layered
on top of the already-validated wh_sva_level + spin_z_norm + extension spec.

Usage:
    python fl_release_point_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

MIN_TYPE_PITCHES = 10  # a pitch type needs at least this many thrown to count toward "consistency"


def build_release_features(raw: pd.DataFrame, windows: list[tuple[str, str]],
                            spin_baseline_pool: pd.DataFrame) -> pd.DataFrame:
    pool = pd.concat([fp.filter_period(raw, s, e) for s, e in windows], ignore_index=True)

    # (1) raw level
    level = pool.groupby("pitcher_id").agg(
        release_pos_z_raw=("release_pos_z", "mean"),
        rel_side_as_raw=("rel_side_as", "mean"),
    ).reset_index()

    # (2) between-pitch-type release consistency: usage-weighted variance of
    # each pitch TYPE's own mean release point, across a pitcher's arsenal
    type_means = pool.groupby(["pitcher_id", "pitch_type"]).agg(
        n=("pitch_type", "size"), mz=("release_pos_z", "mean"), mx=("rel_side_as", "mean"),
    ).reset_index()
    type_means = type_means[type_means["n"] >= MIN_TYPE_PITCHES]

    def consistency(g):
        if len(g) < 2:
            return pd.Series({"release_consist_z": np.nan, "release_consist_x": np.nan, "n_types": len(g)})
        w = g["n"].values
        vz = np.average((g["mz"].values - np.average(g["mz"].values, weights=w)) ** 2, weights=w)
        vx = np.average((g["mx"].values - np.average(g["mx"].values, weights=w)) ** 2, weights=w)
        return pd.Series({"release_consist_z": np.sqrt(vz), "release_consist_x": np.sqrt(vx), "n_types": len(g)})

    consist = type_means.groupby("pitcher_id").apply(consistency, include_groups=False).reset_index()

    # also need spin_z_norm + extension (the already-validated spec) to layer on top of
    bt_stats = spin_baseline_pool.groupby("pitch_type")["spin_rate"].agg(["mean", "std"]).rename(
        columns={"mean": "type_mean", "std": "type_std"})
    pool2 = pool.merge(bt_stats, on="pitch_type", how="left")
    pool2["spin_z"] = (pool2["spin_rate"] - pool2["type_mean"]) / pool2["type_std"]
    spin_ext = pool2.groupby("pitcher_id").agg(spin_z_norm=("spin_z", "mean"), extension=("extension", "mean")).reset_index()

    out = level.merge(consist, on="pitcher_id", how="left").merge(spin_ext, on="pitcher_id", how="left")
    return out


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    spin_baseline_pool = fp.filter_period(raw, bt.START_2025, bt.END_2025)
    traits_tr = build_release_features(raw, [(bt.START_2025, bt.END_2025)], spin_baseline_pool)
    traits_te = build_release_features(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)], spin_baseline_pool)
    df_tr = df_tr.merge(traits_tr, on="pitcher_id", how="left")
    df_te = df_te.merge(traits_te, on="pitcher_id", how="left")

    FEAT_COLS = ["release_pos_z_raw", "rel_side_as_raw", "release_consist_z", "release_consist_x",
                 "spin_z_norm", "extension"]
    for col in FEAT_COLS:
        fill = df_tr[col].mean()
        df_tr[col] = df_tr[col].fillna(fill)
        df_te[col] = df_te[col].fillna(fill)

    BASE = ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level"]
    BEST_SO_FAR = BASE + ["spin_z_norm", "extension"]
    SPECS = {
        "current best (spin_z + extension)": BEST_SO_FAR,
        "+release_pos_z (raw level)": BEST_SO_FAR + ["release_pos_z_raw"],
        "+rel_side_as (raw level)": BEST_SO_FAR + ["rel_side_as_raw"],
        "+release consistency (vertical, deception)": BEST_SO_FAR + ["release_consist_z"],
        "+release consistency (horizontal, deception)": BEST_SO_FAR + ["release_consist_x"],
        "+both release consistency dims": BEST_SO_FAR + ["release_consist_z", "release_consist_x"],
    }

    print("\n" + "=" * 100)
    print("K% held-out: fit fold 1 (full-2025->1H2026), frozen, scored fold 2 (->2H2026)")
    print("=" * 100)
    for name, k_cols in SPECS.items():
        d_tr = df_tr.dropna(subset=["K_rate_actual"] + k_cols)
        w_tr = mm.reliability_weight(d_tr["PA_test"], n0_k).values
        m_k = bt.fit_weighted(d_tr, "K_rate_actual", k_cols, w_tr)

        d_te = df_te.dropna(subset=["K_rate_actual"] + k_cols).copy()
        d_te["pred"] = m_k.predict(d_te)
        w_te = mm.reliability_weight(d_te["PA_test"], n0_k).values
        r2 = mm.weighted_r2(d_te["K_rate_actual"].values, d_te["pred"].values, w_te)
        rmse = np.sqrt(np.average((d_te["K_rate_actual"] - d_te["pred"]) ** 2, weights=w_te))

        print(f"\n--- {name} ---")
        print(f"  held-out K% R2_w={r2:.4f}  RMSE_w={rmse:.4f}")
        new_terms = [c for c in k_cols if c not in BASE]
        print(m_k.summary2().tables[1].loc[new_terms if new_terms else slice(None), ["Coef.", "Std.Err.", "P>|t|"]].round(5).to_string())
