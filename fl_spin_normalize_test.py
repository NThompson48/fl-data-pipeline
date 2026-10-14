"""
fl_spin_normalize_test.py -- the raw per-pitcher mean spin_rate used in
fl_trait_regressor_test.py blends spin rates across pitch types that differ
by over 1000 rpm (splitter ~1170 vs sweeper ~2420) -- a pitcher's average is
therefore driven heavily by THEIR PITCH MIX, not just how much spin they
generate on a given pitch. Tests a pitch-type-normalized alternative: for
each pitch, a z-score relative to that SAME pitch type's own league
baseline (computed from the TRAINING window only, to avoid leakage), then
averaged per pitcher -- isolates "spins the ball more than typical for the
pitch types he throws" from "throws high-spin pitch types more often."

Validated the same way as every other candidate this session: fit on fold 1
(full-2025 -> actual 1H2026), freeze, score on fold 2 (2025+1H2026 -> actual
2H2026).

Usage:
    python fl_spin_normalize_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm


def build_spin_features(raw: pd.DataFrame, windows: list[tuple[str, str]], baseline_pool: pd.DataFrame) -> pd.DataFrame:
    """baseline_pool defines the pitch-type mean/std (training-window only,
    shared across both the raw-mean and normalized variants so the
    comparison isn't confounded by different baselines)."""
    pool = pd.concat([fp.filter_period(raw, s, e) for s, e in windows], ignore_index=True)
    bt_stats = baseline_pool.groupby("pitch_type")["spin_rate"].agg(["mean", "std"]).rename(
        columns={"mean": "type_mean", "std": "type_std"})
    pool = pool.merge(bt_stats, on="pitch_type", how="left")
    pool["spin_z"] = (pool["spin_rate"] - pool["type_mean"]) / pool["type_std"]

    agg = pool.groupby("pitcher_id").agg(
        spin_rate_raw=("spin_rate", "mean"),
        spin_z_norm=("spin_z", "mean"),
        extension=("extension", "mean"),
    ).reset_index()
    return agg


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    # pitch-type baselines from the FOLD-1 TRAINING window only (full 2025) --
    # used for BOTH folds' normalization, same discipline as n0/naive priors
    # being fit once and reused, never re-derived from the test window
    baseline_pool = fp.filter_period(raw, bt.START_2025, bt.END_2025)

    traits_tr = build_spin_features(raw, [(bt.START_2025, bt.END_2025)], baseline_pool)
    traits_te = build_spin_features(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)], baseline_pool)
    df_tr = df_tr.merge(traits_tr, on="pitcher_id", how="left")
    df_te = df_te.merge(traits_te, on="pitcher_id", how="left")
    for col in ["spin_rate_raw", "spin_z_norm", "extension"]:
        fill = df_tr[col].mean()
        df_tr[col] = df_tr[col].fillna(fill)
        df_te[col] = df_te[col].fillna(fill)

    SPECS = {
        "baseline (no spin/extension)": ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level"],
        "+raw mean spin_rate +extension": ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level", "spin_rate_raw", "extension"],
        "+pitch-type-normalized spin_z +extension": ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level", "spin_z_norm", "extension"],
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
        print(m_k.summary2().tables[1][["Coef.", "Std.Err.", "P>|t|"]].round(4).to_string())
