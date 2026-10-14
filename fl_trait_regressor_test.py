"""
fl_trait_regressor_test.py -- tests median_pitches_per_game (role/workload
proxy), spin_rate, and extension as DIRECT additional regressors in the K%
equation, individually and jointly, validated with the same discipline that
caught the milb/none dummy correction failing to generalize: fit on fold 1
(full-2025 -> actual 1H2026), freeze, score on fold 2 (2025+1H2026 -> actual
2H2026) -- never fit and score on the same fold that motivated the test.

Each candidate trait is computed per-pitcher over EACH fold's own proper
feature window (not a pooled 2025+2026 window, which would leak fold 2's
window into fold 1's features) -- same discipline every other feature in
this harness already follows.

Usage:
    python fl_trait_regressor_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm

SPEC_BASE = bt.MODEL_SPECS["naive+other_league+surprise_prior"]


def build_role_stuff_traits(raw: pd.DataFrame, windows: list[tuple[str, str]]) -> pd.DataFrame:
    pool = pd.concat([fp.filter_period(raw, s, e) for s, e in windows], ignore_index=True)
    agg = pool.groupby("pitcher_id").agg(
        spin_rate=("spin_rate", "mean"), extension=("extension", "mean"),
    ).reset_index()
    per_game = pool.groupby(["pitcher_id", "date"]).size().reset_index(name="pitches_in_game")
    role = per_game.groupby("pitcher_id")["pitches_in_game"].median().reset_index(name="median_pitches_per_game")
    return agg.merge(role, on="pitcher_id", how="left")


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()

    df_tr, n0_k, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)],
                                      with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)],
                               [(bt._DAY_AFTER_2026_CUT, bt.END_2026)], with_changepoints=False)

    # fold-specific feature windows -- same windows each fold's OTHER
    # features (naive_K_rate, wh_sva_level, etc.) are already built from
    traits_tr = build_role_stuff_traits(raw, [(bt.START_2025, bt.END_2025)])
    traits_te = build_role_stuff_traits(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)])
    df_tr = df_tr.merge(traits_tr, on="pitcher_id", how="left")
    df_te = df_te.merge(traits_te, on="pitcher_id", how="left")

    # impute missing (no pitches in window -- shouldn't happen for anyone
    # with a naive_K_rate, but guard anyway) with the training population's
    # own mean, same neutral-imputation convention used elsewhere
    for col in ["spin_rate", "extension", "median_pitches_per_game"]:
        fill = df_tr[col].mean()
        df_tr[col] = df_tr[col].fillna(fill)
        df_te[col] = df_te[col].fillna(fill)

    SPECS = {
        "baseline": SPEC_BASE["K"],
        "+role (median_pitches_per_game)": SPEC_BASE["K"] + ["median_pitches_per_game"],
        "+spin_rate": SPEC_BASE["K"] + ["spin_rate"],
        "+extension": SPEC_BASE["K"] + ["extension"],
        "+all three": SPEC_BASE["K"] + ["median_pitches_per_game", "spin_rate", "extension"],
    }

    print("\n" + "=" * 100)
    print("K% held-out performance: fit on fold 1 (full-2025->1H2026), frozen, scored on fold 2 (->2H2026)")
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
        print("  coefficients (fit on fold 1):")
        summary = m_k.summary2().tables[1][["Coef.", "Std.Err.", "P>|t|"]]
        print(summary.round(4).to_string())
