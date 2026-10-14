"""
mlb_zone_based_command.py -- the video's actual approach: PRE-DEFINE zones as
Good/Great from pooled league data, then just count how often each pitcher's
pitches land there, instead of a per-pitch ML model (mlb_count_leverage_command.py).

Uses Statcast's own `zone` column (1-9 = the in-zone 3x3 grid, 11-14 = the four
out-of-zone "chase" quadrants) -- a ready-made, standard zone grid, exactly the
kind of pre-emptive zone assignment the video uses (no custom binning needed).

Method:
  1. For each (family, count-leverage, zone) cell, compute the pooled
     swing-conditioned whiff rate across ALL pitchers -- 3 families x 3
     leverage buckets x ~13 zones = ~117 cells, each averaged over tens of
     thousands of pitches from 1000+ pitchers, so any one pitcher's own
     contribution to "what counts as Good" is negligible (unlike the batter-
     matchup leave-one-out case, this doesn't need an explicit holdout -- the
     lookup table has too few degrees of freedom relative to the pooled N for
     leakage to matter).
  2. Classify zones as Great (top ~2 by whiff rate), Good (next ~3), Other,
     within each (family, leverage) cell -- mirroring the video's two-tier
     Good/Great system.
  3. Aggregate to pitcher-season: GoodZone%/GreatZone% per family, blended
     overall -- a purely descriptive, zero-model-fitting metric.
  4. Compare against cmd_value (the two-stage XGBoost score from
     mlb_count_leverage_command.py): correlation between the two, and how each
     does on K_rate/BB_rate/stickiness -- does the flexible model actually buy
     anything over "count how often you hit the pre-defined good zone"?

Usage:
    python mlb_zone_based_command.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlb_marcel as mm
import mlb_pitcher_features as mf
from mlb_kbb_model import MIN_PA, PANEL_PATH
from mlb_count_leverage_command import leverage_bucket

pd.set_option("display.width", 160, "display.max_columns", 40)

FAMILIES = ["FB", "BRK", "OFF"]
N_GREAT = 2  # top N zones classified "Great"
N_GOOD = 3   # next N zones classified "Good" (on top of Great)


def build_zone_table(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[raw["is_swing"] & raw["pitch_family"].isin(FAMILIES) & raw["zone"].notna()].copy()
    d["leverage"] = leverage_bucket(d["balls"].values, d["strikes"].values)
    tbl = d.groupby(["pitch_family", "leverage", "zone"]).agg(
        n=("is_whiff", "size"), whiff_rate=("is_whiff", "mean")).reset_index()
    tbl = tbl[tbl["n"] >= 500]  # need real sample size per cell to trust the rate

    tbl["tier"] = "other"
    for (fam, lev), g in tbl.groupby(["pitch_family", "leverage"]):
        order = g.sort_values("whiff_rate", ascending=False).index
        tbl.loc[order[:N_GREAT], "tier"] = "great"
        tbl.loc[order[N_GREAT:N_GREAT + N_GOOD], "tier"] = "good"
    return tbl


if __name__ == "__main__":
    print("Loading pitch-level data...")
    raw = mf.load_statcast_all()
    raw = mf.derive_flags(raw)

    print("\nBuilding pre-defined zone table (pooled across all pitchers)...")
    zone_tbl = build_zone_table(raw)
    print(zone_tbl.sort_values(["pitch_family", "leverage", "whiff_rate"], ascending=[True, True, False])
          .to_string(index=False))

    raw["leverage"] = leverage_bucket(raw["balls"].values, raw["strikes"].values)
    tagged = raw[raw["is_swing"] & raw["pitch_family"].isin(FAMILIES) & raw["zone"].notna()].merge(
        zone_tbl[["pitch_family", "leverage", "zone", "tier"]], on=["pitch_family", "leverage", "zone"], how="left")
    tagged["tier"] = tagged["tier"].fillna("other")
    tagged["is_great"] = tagged["tier"] == "great"
    tagged["is_good_or_great"] = tagged["tier"].isin(["good", "great"])

    print("\nAggregating to pitcher-season zone-command scores...")
    season_zone = tagged.groupby(["pitcher", "game_year", "pitch_family"]).agg(
        n=("is_whiff", "size"), good_pct=("is_good_or_great", "mean"), great_pct=("is_great", "mean")
    ).reset_index()
    season_zone = season_zone[season_zone["n"] >= mf.MIN_N_FAMILY]

    wide = season_zone.pivot(index=["pitcher", "game_year"], columns="pitch_family",
                              values=["good_pct", "great_pct", "n"])
    wide.columns = [f"{a}_{b.lower()}" for a, b in wide.columns]
    wide = wide.reset_index()
    good_cols = [f"good_pct_{f.lower()}" for f in FAMILIES if f"good_pct_{f.lower()}" in wide.columns]
    wide["good_pct_overall"] = wide[good_cols].mean(axis=1, skipna=True)

    # ---- compare against the ML-based cmd_value ---- #
    cmd = pd.read_parquet("output/count_leverage_command.parquet")
    panel = pd.read_parquet(PANEL_PATH)
    d = panel.merge(wide, on=["pitcher", "game_year"], how="inner").merge(
        cmd[["pitcher", "game_year", "cmd_value_overall"]], on=["pitcher", "game_year"], how="left")
    elig = d[(d["PA"] >= MIN_PA) & d["good_pct_overall"].notna()]
    print(f"\nn eligible pitcher-seasons = {len(elig)}")

    print(f"\ncorr(good_pct_overall [zone-based], cmd_value_overall [ML-based]) = "
          f"{elig['good_pct_overall'].corr(elig['cmd_value_overall']):.4f}")

    print("\n=== Relationship with K_rate and BB_rate (concurrent) ===")
    for col in good_cols + ["good_pct_overall"]:
        ck = elig[col].corr(elig["K_rate"])
        cb = elig[col].corr(elig["BB_rate"])
        print(f"  {col:20s}  corr(K_rate)={ck:+.4f}   corr(BB_rate)={cb:+.4f}")
    print(f"  {'cmd_value_overall (ML)':20s}  corr(K_rate)={elig['cmd_value_overall'].corr(elig['K_rate']):+.4f}   "
          f"corr(BB_rate)={elig['cmd_value_overall'].corr(elig['BB_rate']):+.4f}")

    print("\n=== Stickiness: year-over-year correlation ===")
    b = elig[["pitcher", "game_year", "good_pct_overall"]].copy()
    b["game_year"] -= 1
    cp = elig.merge(b, on=["pitcher", "game_year"], suffixes=("1", "2"))
    print(f"  n couplets = {len(cp)}")
    print(f"  raw corr(good_pct_overall_Y, good_pct_overall_Y+1) = "
          f"{cp['good_pct_overall1'].corr(cp['good_pct_overall2']):.4f}  "
          f"(cmd_value_overall's was 0.4728)")

    print("\n=== Stability: n0 ===")
    n_col_sum = wide[[f"n_{f.lower()}" for f in FAMILIES if f"n_{f.lower()}" in wide.columns]].fillna(0).sum(axis=1)
    elig2 = elig.copy()
    elig2["n_total"] = n_col_sum.reindex(elig2.index)
    n0_mm, info = mm.estimate_n0(elig2["good_pct_overall"], elig2["n_total"], min_n=30)
    print(f"  n0 (method-of-moments) = {n0_mm:.1f}  {info}  (cmd_value_overall's was 495.7)")
