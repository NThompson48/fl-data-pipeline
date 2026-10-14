"""
fl_flatpool_tuning.py -- two follow-up tests on the flat-pool naive prior (see
DECISION_LOG.md "Marcel-weighted-with-reliability FL prior... does NOT beat flat
pooling"):
  1. How many trailing years should the pool include? (currently always "every
     available year up to 3" -- never swept)
  2. Does folding the college-bridge-translated rate into the SAME pool (as pseudo
     K/PA counts) beat keeping it as a separate regressor (current production
     design)?

Reuses fl_marcel_backtest.py's StatsCrew-matched data/scoring scaffolding and
fl_college_bridge.py's already-validated bridge model -- no new data pulls.

Usage:
    python fl_flatpool_tuning.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_production_projection as prod
import fl_college_bridge as bridge
import fl_marcel_backtest as fmb
import mlb_marcel as mm


def flat_pool_windowed(season_rates: dict[int, tuple[float, float]], target_year: int,
                        window: int, n0: float, lg_rate: float) -> float:
    """Same flat-pool arithmetic as fmb.flat_pool_naive, restricted to the most
    recent `window` trailing years (years in [target_year-window, target_year-1])."""
    tot_k, tot_pa = 0.0, 0.0
    for year, (rate, pa) in season_rates.items():
        if year < target_year - window or year >= target_year:
            continue
        if pa <= 0 or np.isnan(rate):
            continue
        tot_k += rate * pa
        tot_pa += pa
    if tot_pa == 0:
        return np.nan
    return (tot_k + n0 * lg_rate) / (tot_pa + n0)


def score_window(season_rates_by_pitcher: dict, actual_df: pd.DataFrame, target_year: int,
                  window: int, n0: float, lg_rate: float, metric: str, label: str) -> dict:
    rows = []
    for _, r in actual_df.iterrows():
        sr = season_rates_by_pitcher.get(r["name_clean"])
        if not sr:
            continue
        pred = flat_pool_windowed(sr, target_year, window, n0, lg_rate)
        rows.append({"PA": r["PA"], "actual": r[metric], "pred": pred})
    d = pd.DataFrame(rows).dropna(subset=["pred", "actual"])
    w = mm.reliability_weight(d["PA"], n0).values
    return {"label": label, "window_years": window, "n": len(d),
            "R2_w": round(mm.weighted_r2(d["actual"], d["pred"], w), 4),
            "RMSE_w_pp": round(mm.weighted_rmse(d["actual"], d["pred"], w) * 100, 3)}


def flat_pool_with_college(season_rates: dict[int, tuple[float, float]], target_year: int,
                            window: int, n0: float, lg_rate: float,
                            college_pseudo: tuple[float, float] | None) -> float:
    """Same as flat_pool_windowed, but ALSO folds in a college-bridge pseudo-season
    (pseudo_rate, pseudo_PA) as one more set of counts in the SAME pool, rather
    than as a separate additive regressor (the current production design)."""
    tot_k, tot_pa = 0.0, 0.0
    for year, (rate, pa) in season_rates.items():
        if year < target_year - window or year >= target_year:
            continue
        if pa <= 0 or np.isnan(rate):
            continue
        tot_k += rate * pa
        tot_pa += pa
    if college_pseudo is not None:
        c_rate, c_pa = college_pseudo
        if c_pa > 0 and not np.isnan(c_rate):
            tot_k += c_rate * c_pa
            tot_pa += c_pa
    if tot_pa == 0:
        return np.nan
    return (tot_k + n0 * lg_rate) / (tot_pa + n0)


def score_with_college(season_rates_by_pitcher: dict, actual_df: pd.DataFrame, target_year: int,
                        window: int, n0: float, lg_rate: float, metric: str,
                        college_pseudo_by_name: dict, label: str) -> dict:
    rows = []
    for _, r in actual_df.iterrows():
        sr = season_rates_by_pitcher.get(r["name_clean"], {})
        pseudo = college_pseudo_by_name.get(r["name_clean"])
        pred = flat_pool_with_college(sr, target_year, window, n0, lg_rate, pseudo)
        rows.append({"PA": r["PA"], "actual": r[metric], "pred": pred, "has_college": pseudo is not None})
    d = pd.DataFrame(rows).dropna(subset=["pred", "actual"])
    w = mm.reliability_weight(d["PA"], n0).values
    overall = {"label": label, "n": len(d),
               "R2_w": round(mm.weighted_r2(d["actual"], d["pred"], w), 4),
               "RMSE_w_pp": round(mm.weighted_rmse(d["actual"], d["pred"], w) * 100, 3)}
    sub = d[d["has_college"]]
    if len(sub) >= 15:
        w_sub = mm.reliability_weight(sub["PA"], n0).values
        overall["n_college_matched"] = len(sub)
        overall["R2_w_college_subset"] = round(mm.weighted_r2(sub["actual"], sub["pred"], w_sub), 4)
    return overall


if __name__ == "__main__":
    print("Loading cached FL pitches + StatsCrew panel...")
    raw = fp.load_fl_pitches_cached()
    id_map = prod.build_canonical_id_map(raw)
    raw["pitcher_id"] = raw["pitcher_id"].replace(id_map)
    sc = pd.read_parquet(fmb.STATSCREW_PATH)

    act_2025 = fmb.fl_season_actual(raw, bt.START_2025, bt.END_2025)
    act_2026 = fmb.fl_season_actual(raw, bt.START_2026, bt.END_2026)

    def build_rates(metric_col, extra_df=None, extra_year=None, extra_metric=None):
        out = {}
        for _, r in sc.iterrows():
            out.setdefault(r["name_clean"], {})[int(r["year"])] = (r[metric_col], r["bf"])
        if extra_df is not None:
            for _, r in extra_df.iterrows():
                out.setdefault(r["name_clean"], {})[extra_year] = (r[extra_metric], r["PA"])
        return out

    k_pre2025 = build_rates("k_pct")
    bb_pre2025 = build_rates("bb_pct")
    k_incl2025 = build_rates("k_pct", act_2025, 2025, "K_rate")
    bb_incl2025 = build_rates("bb_pct", act_2025, 2025, "BB_rate")

    pool_k = pd.concat([sc[["k_pct", "bf"]].rename(columns={"k_pct": "rate", "bf": "pa"}),
                        act_2025[["K_rate", "PA"]].rename(columns={"K_rate": "rate", "PA": "pa"})])
    pool_bb = pd.concat([sc[["bb_pct", "bf"]].rename(columns={"bb_pct": "rate", "bf": "pa"}),
                         act_2025[["BB_rate", "PA"]].rename(columns={"BB_rate": "rate", "PA": "pa"})])
    n0_k, _ = mm.estimate_n0(pool_k["rate"], pool_k["pa"], min_n=20)
    n0_bb, _ = mm.estimate_n0(pool_bb["rate"], pool_bb["pa"], min_n=20)
    lg_k = np.average(pool_k["rate"], weights=pool_k["pa"])
    lg_bb = np.average(pool_bb["rate"], weights=pool_bb["pa"])

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: how many trailing years should the flat pool include?")
    print("=" * 70)
    results = []
    for window in [1, 2, 3]:
        results.append(score_window(k_pre2025, act_2025, 2025, window, n0_k, lg_k, "K_rate", "K% val2025"))
        results.append(score_window(bb_pre2025, act_2025, 2025, window, n0_bb, lg_bb, "BB_rate", "BB% val2025"))
        results.append(score_window(k_incl2025, act_2026, 2026, window, n0_k, lg_k, "K_rate", "K% test2026"))
        results.append(score_window(bb_incl2025, act_2026, 2026, window, n0_bb, lg_bb, "BB_rate", "BB% test2026"))
    print(pd.DataFrame(results).to_string(index=False))

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: folding the college-bridge rate into the SAME pool")
    print("=" * 70)
    bridge_model = bridge.fit_bridge()
    pseudo_pa_k, pseudo_pa_bb = bridge.bridge_pseudo_pa(bridge_model, n0_k, n0_bb)
    print(f"college pseudo_PA: K={pseudo_pa_k:.1f}, BB={pseudo_pa_bb:.1f}")

    def build_college_pseudo(target_year, rate_col, model_key, pseudo_pa):
        out = {}
        cutoff = target_year - 1
        for name_clean in set(bridge_model["college_years"]["name_clean"]):
            feats = bridge.predict_from_name(bridge_model, name_clean, as_of_season=cutoff)
            if feats is None:
                continue
            out[name_clean] = (feats[rate_col], pseudo_pa)
        return out

    college_k_2025 = build_college_pseudo(2025, "fl_k_rate_bridge", "k_model", pseudo_pa_k)
    college_bb_2025 = build_college_pseudo(2025, "fl_bb_rate_bridge", "bb_model", pseudo_pa_bb)
    college_k_2026 = build_college_pseudo(2026, "fl_k_rate_bridge", "k_model", pseudo_pa_k)
    college_bb_2026 = build_college_pseudo(2026, "fl_bb_rate_bridge", "bb_model", pseudo_pa_bb)

    best_window = 3  # compare against the already-established 3-year baseline
    print(f"\n--- baseline (FL years only, window={best_window}) vs +college pseudo-season ---")
    r1 = score_window(k_pre2025, act_2025, 2025, best_window, n0_k, lg_k, "K_rate", "K% val2025 FL-only")
    r2 = score_with_college(k_pre2025, act_2025, 2025, best_window, n0_k, lg_k, "K_rate",
                             college_k_2025, "K% val2025 +college")
    r3 = score_window(bb_pre2025, act_2025, 2025, best_window, n0_bb, lg_bb, "BB_rate", "BB% val2025 FL-only")
    r4 = score_with_college(bb_pre2025, act_2025, 2025, best_window, n0_bb, lg_bb, "BB_rate",
                             college_bb_2025, "BB% val2025 +college")
    r5 = score_window(k_incl2025, act_2026, 2026, best_window, n0_k, lg_k, "K_rate", "K% test2026 FL-only")
    r6 = score_with_college(k_incl2025, act_2026, 2026, best_window, n0_k, lg_k, "K_rate",
                             college_k_2026, "K% test2026 +college")
    r7 = score_window(bb_incl2025, act_2026, 2026, best_window, n0_bb, lg_bb, "BB_rate", "BB% test2026 FL-only")
    r8 = score_with_college(bb_incl2025, act_2026, 2026, best_window, n0_bb, lg_bb, "BB_rate",
                             college_bb_2026, "BB% test2026 +college")
    print(pd.DataFrame([r1, r2, r3, r4, r5, r6, r7, r8]).to_string(index=False))
