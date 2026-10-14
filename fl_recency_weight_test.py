"""
fl_recency_weight_test.py -- a more thorough follow-up to two earlier findings
(DECISION_LOG.md "Marcel-weighted-with-reliability FL prior" and "Flat-pool
window size"):
  - Flat (equal-weight) pooling of just the single most recent trailing year
    beat pooling 2-3 years, and beat classic Marcel 5/4/3 recency weighting,
    on BOTH checks tried (val2025, test2026).
  - Root cause: only 14% of matched players had all 3 trailing years, and for
    that minority, downweighting older seasons threw away real signal.
  - The entry's own "Revisit if" flagged testing a GENTLER recency taper than
    full classic Marcel, and using a larger evaluation sample -- this script
    does both:
      1. Sweeps a continuous decay rate (not just flat-vs-full-Marcel) on the
         SAME pooled-counts mechanism already shown to beat weighted-average-
         of-rates (mlb_marcel.marcel_reliability_prior's mechanism), isolating
         "how much should age matter" from "pool counts vs average rates."
      2. Adds a THIRD full-season evaluation fold (predict StatsCrew's own
         2024 from StatsCrew 2022-2023) on top of the original two
         (val2025, test2026) -- all three use the SAME StatsCrew+Supabase
         data already pulled, no new scraping, but roughly 50% more fold-level
         evaluation sample than either earlier script used alone.

decay=1.0 reproduces the current flat-pool baseline exactly; decay=0.0
reproduces the already-shipped "most recent year only" design. Everything
between is the gentler-taper space the earlier entry flagged as untested.

Usage:
    python fl_recency_weight_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_production_projection as prod
import fl_marcel_backtest as fmb
import mlb_marcel as mm

DECAYS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
MAX_WINDOW = 3


def decayed_pool(season_rates: dict[int, tuple[float, float]], target_year: int,
                  decay: float, n0: float, lg_rate: float, max_window: int = MAX_WINDOW) -> float:
    """Generalizes fl_flatpool_tuning.flat_pool_windowed with a per-year-of-lag
    multiplicative decay on the POOLED-COUNTS weight (not a weighted average of
    season rates -- that mechanism already lost, see module docstring).
    decay=1.0 -> flat pool (lag doesn't matter). decay=0.0 -> only the single
    most recent year counts (weight 0**0=1 at lag=1, 0 at lag>=2)."""
    tot_k, tot_pa = 0.0, 0.0
    for year, (rate, pa) in season_rates.items():
        lag = target_year - year
        if lag < 1 or lag > max_window:
            continue
        if pa <= 0 or np.isnan(rate):
            continue
        w = decay ** (lag - 1)
        tot_k += w * rate * pa
        tot_pa += w * pa
    if tot_pa == 0:
        return np.nan
    return (tot_k + n0 * lg_rate) / (tot_pa + n0)


def score_decay(season_rates_by_pitcher: dict, actual_df: pd.DataFrame, target_year: int,
                 decay: float, n0: float, lg_rate: float, metric: str, fold_label: str) -> dict:
    rows = []
    for _, r in actual_df.iterrows():
        sr = season_rates_by_pitcher.get(r["name_clean"])
        if not sr:
            continue
        pred = decayed_pool(sr, target_year, decay, n0, lg_rate)
        rows.append({"PA": r["PA"], "actual": r[metric], "pred": pred})
    d = pd.DataFrame(rows).dropna(subset=["pred", "actual"])
    if len(d) < 15:
        return {"fold": fold_label, "decay": decay, "n": len(d), "R2_w": np.nan, "RMSE_w_pp": np.nan}
    w = mm.reliability_weight(d["PA"], n0).values
    return {"fold": fold_label, "decay": decay, "n": len(d),
            "R2_w": round(mm.weighted_r2(d["actual"], d["pred"], w), 4),
            "RMSE_w_pp": round(mm.weighted_rmse(d["actual"], d["pred"], w) * 100, 3),
            "total_w": round(float(w.sum()), 1)}


if __name__ == "__main__":
    print("Loading cached FL pitches + StatsCrew panel...")
    raw = fp.load_fl_pitches_cached()
    id_map = prod.build_canonical_id_map(raw)
    raw["pitcher_id"] = raw["pitcher_id"].replace(id_map)
    sc = pd.read_parquet(fmb.STATSCREW_PATH)

    act_2024 = sc[sc["year"] == 2024][["name_clean", "k_pct", "bb_pct", "bf"]].rename(
        columns={"k_pct": "K_rate", "bb_pct": "BB_rate", "bf": "PA"})
    act_2025 = fmb.fl_season_actual(raw, bt.START_2025, bt.END_2025)
    act_2026 = fmb.fl_season_actual(raw, bt.START_2026, bt.END_2026)

    def build_rates(metric_col, extra_dfs=()):
        """extra_dfs: list of (df, year, metric_col_in_df) to layer on top of the
        StatsCrew 2022-2024 base -- e.g. Supabase 2025 actual, treated as one
        more year of trailing history for the 2026 fold."""
        out = {}
        for _, r in sc.iterrows():
            out.setdefault(r["name_clean"], {})[int(r["year"])] = (r[metric_col], r["bf"])
        for df, yr, col in extra_dfs:
            for _, r in df.iterrows():
                out.setdefault(r["name_clean"], {})[yr] = (r[col], r["PA"])
        return out

    k_rates = build_rates("k_pct", [(act_2025, 2025, "K_rate")])
    bb_rates = build_rates("bb_pct", [(act_2025, 2025, "BB_rate")])

    # n0/lg_rate fit ONCE on the full available cross-section (StatsCrew 2022-2024
    # + Supabase 2025 + Supabase 2026) -- a fixed population hyperparameter, same
    # convention as every other reliability constant in this project, not re-tuned
    # per fold or per decay value.
    pool_k = pd.concat([sc[["k_pct", "bf"]].rename(columns={"k_pct": "rate", "bf": "pa"}),
                        act_2025[["K_rate", "PA"]].rename(columns={"K_rate": "rate", "PA": "pa"}),
                        act_2026[["K_rate", "PA"]].rename(columns={"K_rate": "rate", "PA": "pa"})])
    pool_bb = pd.concat([sc[["bb_pct", "bf"]].rename(columns={"bb_pct": "rate", "bf": "pa"}),
                         act_2025[["BB_rate", "PA"]].rename(columns={"BB_rate": "rate", "PA": "pa"}),
                         act_2026[["BB_rate", "PA"]].rename(columns={"BB_rate": "rate", "PA": "pa"})])
    n0_k, info_k = mm.estimate_n0(pool_k["rate"], pool_k["pa"], min_n=20)
    n0_bb, info_bb = mm.estimate_n0(pool_bb["rate"], pool_bb["pa"], min_n=20)
    lg_k = np.average(pool_k["rate"], weights=pool_k["pa"])
    lg_bb = np.average(pool_bb["rate"], weights=pool_bb["pa"])
    print(f"n0_k={n0_k:.1f}  n0_bb={n0_bb:.1f}  lg_k={lg_k:.4f}  lg_bb={lg_bb:.4f}")

    FOLDS = [
        ("2024 (StatsCrew 2022-23 -> StatsCrew 2024)", k_rates, bb_rates, act_2024, 2024),
        ("2025 (StatsCrew 2022-24 -> Supabase 2025)", k_rates, bb_rates, act_2025, 2025),
        ("2026 (StatsCrew 2023-24+Supabase25 -> Supabase 2026)",
         build_rates("k_pct", [(act_2025, 2025, "K_rate")]),
         build_rates("bb_pct", [(act_2025, 2025, "BB_rate")]), act_2026, 2026),
    ]

    print("\n" + "=" * 90)
    print(f"Recency-decay sweep: decay in {DECAYS} (1.0=flat pool, 0.0=most-recent-year-only)")
    print("=" * 90)
    k_results, bb_results = [], []
    for fold_label, k_r, bb_r, act, target_year in FOLDS:
        for decay in DECAYS:
            k_results.append(score_decay(k_r, act, target_year, decay, n0_k, lg_k, "K_rate", fold_label))
            bb_results.append(score_decay(bb_r, act, target_year, decay, n0_bb, lg_bb, "BB_rate", fold_label))

    print("\n--- K% ---")
    k_df = pd.DataFrame(k_results)
    print(k_df.to_string(index=False))
    print("\n--- BB% ---")
    bb_df = pd.DataFrame(bb_results)
    print(bb_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("Aggregate: mean R2_w across all 3 folds per decay value (simple average, each fold equal weight)")
    print("=" * 90)
    k_agg = k_df.groupby("decay")["R2_w"].mean().round(4)
    bb_agg = bb_df.groupby("decay")["R2_w"].mean().round(4)
    print("K%  mean R2_w by decay:\n", k_agg.to_string())
    print("\nBB% mean R2_w by decay:\n", bb_agg.to_string())
