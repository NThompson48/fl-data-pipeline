"""
fl_marcel_backtest.py -- validates the new Marcel-weighted-with-reliability FL prior
(mlb_marcel.marcel_reliability_prior) against the CURRENT production naive_prior
method, using StatsCrew.com's 2022-2024 box-score history (fl_statscrew_pull.py) to
supply the trailing years Supabase Trackman data doesn't cover.

Design (approved plan):
  - Validate on 2025: trailing years 2022/2023/2024 (all StatsCrew) -> actual 2025
    (Supabase, exact pitch-level).
  - Test on 2026 (the real held-out check): trailing years 2023/2024 (StatsCrew) +
    2025 (Supabase actual) -> actual 2026 (Supabase).
  - Old vs new uses the IDENTICAL underlying trailing data for both, so the
    comparison isolates the WEIGHTING SCHEME (flat equal-pool+shrink vs.
    recency x reliability) rather than "more data helps."
  - n0/reg fit ONCE via mm.estimate_n0 on the full pooled cross-section (all
    season-rate observations, 2022-2025), not re-tuned per target year.

Usage:
    python fl_marcel_backtest.py
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_production_projection as prod
import mlb_marcel as mm

STATSCREW_PATH = "output/fl_statscrew_pitching_2022_2024.parquet"


def clean_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fl_name_to_statscrew_clean(pitcher_name: str) -> str:
    s = str(pitcher_name)
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return clean_name(s)


def fl_season_actual(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Full-season K%/BB%/PA per FL pitcher (consolidated ids), with a matching
    StatsCrew-style name_clean column attached."""
    kbb = fp.build_period_kbb(fp.filter_period(raw, start, end), min_pa=1)
    names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    kbb = kbb.merge(names, on="pitcher_id", how="left")
    kbb["name_clean"] = kbb["pitcher"].apply(fl_name_to_statscrew_clean)
    return kbb[["pitcher_id", "pitcher", "name_clean", "PA", "K_rate", "BB_rate"]]


def flat_pool_naive(season_rates: dict[int, tuple[float, float]], n0: float, lg_rate: float) -> float:
    """The CURRENT production method (fl_halfseason_backtest.naive_prior's formula):
    pool K/PA equally across whatever years are given (no recency, no per-season
    reliability), single shrinkage. Reusing the exact arithmetic here (not calling
    naive_prior directly, since that function pools RAW PITCH DATA, not pre-
    computed rates) so the "old" baseline is a faithful match."""
    tot_k, tot_pa = 0.0, 0.0
    for year, (rate, pa) in season_rates.items():
        if pa <= 0 or np.isnan(rate):
            continue
        tot_k += rate * pa
        tot_pa += pa
    if tot_pa == 0:
        return np.nan
    return (tot_k + n0 * lg_rate) / (tot_pa + n0)


def run_comparison(label: str, season_rates_by_pitcher: dict, actual_df: pd.DataFrame,
                    target_year: int, n0: float, reg: float, lg_rate: float, metric: str) -> pd.DataFrame:
    rows = []
    for _, r in actual_df.iterrows():
        sr = season_rates_by_pitcher.get(r["name_clean"])
        if not sr:
            continue
        old_pred = flat_pool_naive(sr, n0, lg_rate)
        new_pred = mm.marcel_reliability_prior(sr, target_year, n0, reg, lg_rate)
        rows.append({"name_clean": r["name_clean"], "PA": r["PA"], "actual": r[metric],
                      "old_pred": old_pred, "new_pred": new_pred})
    d = pd.DataFrame(rows).dropna(subset=["old_pred", "new_pred", "actual"])
    w = mm.reliability_weight(d["PA"], n0).values
    old_score = {"label": f"{label} OLD (flat-pool)", "n": len(d),
                 "R2_w": round(mm.weighted_r2(d["actual"], d["old_pred"], w), 4),
                 "RMSE_w_pp": round(mm.weighted_rmse(d["actual"], d["old_pred"], w) * 100, 3)}
    new_score = {"label": f"{label} NEW (marcel+reliability)", "n": len(d),
                 "R2_w": round(mm.weighted_r2(d["actual"], d["new_pred"], w), 4),
                 "RMSE_w_pp": round(mm.weighted_rmse(d["actual"], d["new_pred"], w) * 100, 3)}
    print(pd.DataFrame([old_score, new_score]).to_string(index=False))
    return d


if __name__ == "__main__":
    print("Loading cached FL pitches...")
    raw = fp.load_fl_pitches_cached()
    id_map = prod.build_canonical_id_map(raw)
    raw["pitcher_id"] = raw["pitcher_id"].replace(id_map)

    print("Loading StatsCrew 2022-2024 panel...")
    sc = pd.read_parquet(STATSCREW_PATH)

    print("Building FL Supabase full-season actuals (2025, 2026)...")
    act_2025 = fl_season_actual(raw, bt.START_2025, bt.END_2025)
    act_2026 = fl_season_actual(raw, bt.START_2026, bt.END_2026)

    # season_rates_by_pitcher[name_clean] = {year: (rate, PA)} -- built per metric
    def build_season_rates(metric_col: str, extra_year_df: pd.DataFrame | None = None,
                            extra_year: int | None = None, extra_metric: str | None = None) -> dict:
        out = {}
        for _, r in sc.iterrows():
            out.setdefault(r["name_clean"], {})[int(r["year"])] = (r[metric_col], r["bf"])
        if extra_year_df is not None:
            for _, r in extra_year_df.iterrows():
                out.setdefault(r["name_clean"], {})[extra_year] = (r[extra_metric], r["PA"])
        return out

    k_rates_pre2025 = build_season_rates("k_pct")
    bb_rates_pre2025 = build_season_rates("bb_pct")
    k_rates_incl2025 = build_season_rates("k_pct", act_2025, 2025, "K_rate")
    bb_rates_incl2025 = build_season_rates("bb_pct", act_2025, 2025, "BB_rate")

    # n0/reg fit ONCE on the full pooled cross-section of season-rate observations
    pool_k = pd.concat([sc[["k_pct", "bf"]].rename(columns={"k_pct": "rate", "bf": "pa"}),
                        act_2025[["K_rate", "PA"]].rename(columns={"K_rate": "rate", "PA": "pa"})])
    pool_bb = pd.concat([sc[["bb_pct", "bf"]].rename(columns={"bb_pct": "rate", "bf": "pa"}),
                         act_2025[["BB_rate", "PA"]].rename(columns={"BB_rate": "rate", "PA": "pa"})])
    n0_k, info_k = mm.estimate_n0(pool_k["rate"], pool_k["pa"], min_n=20)
    n0_bb, info_bb = mm.estimate_n0(pool_bb["rate"], pool_bb["pa"], min_n=20)
    lg_k = np.average(pool_k["rate"], weights=pool_k["pa"])
    lg_bb = np.average(pool_bb["rate"], weights=pool_bb["pa"])
    print(f"\nn0_k={n0_k:.1f} {info_k}  lg_k={lg_k:.4f}")
    print(f"n0_bb={n0_bb:.1f} {info_bb}  lg_bb={lg_bb:.4f}")

    print("\n" + "=" * 70)
    print("VALIDATION: trailing 2022-2024 (StatsCrew) -> actual 2025")
    print("=" * 70)
    run_comparison("K% val2025", k_rates_pre2025, act_2025, 2025, n0_k, n0_k, lg_k, "K_rate")
    run_comparison("BB% val2025", bb_rates_pre2025, act_2025, 2025, n0_bb, n0_bb, lg_bb, "BB_rate")

    print("\n" + "=" * 70)
    print("TEST (held out): trailing 2023-2024 (StatsCrew) + 2025 (Supabase) -> actual 2026")
    print("=" * 70)
    run_comparison("K% test2026", k_rates_incl2025, act_2026, 2026, n0_k, n0_k, lg_k, "K_rate")
    run_comparison("BB% test2026", bb_rates_incl2025, act_2026, 2026, n0_bb, n0_bb, lg_bb, "BB_rate")
