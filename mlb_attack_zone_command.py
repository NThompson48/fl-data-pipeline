"""
mlb_attack_zone_command.py -- rebuild the pre-defined Good/Great zone tiers
using Statcast's public "Attack Zones" (Heart/Shadow/Chase/Waste) instead of
the raw 13-cell zone grid in mlb_zone_based_command.py.

Attack Zones use a FIXED reference strike zone (not each batter's own
sz_top/sz_bot -- that's what the 1-9/11-14 `zone` field does; Attack Zones are
an absolute physical region), per the standard, widely-reproduced public
definition:
  Heart:  |plate_x| <= 0.558   and  1.833 <= plate_z <= 3.166
  Shadow: |plate_x| <= 1.108   and  1.166 <= plate_z <= 3.833   (minus Heart)
  Chase:  |plate_x| <= 1.667   and  0.500 <= plate_z <= 4.500   (minus Shadow/Heart)
  Waste:  everything else

Same pipeline as mlb_zone_based_command.py otherwise: pool swing-conditioned
whiff rate by (family, count-leverage, attack_zone), rank tiers Good/Great,
aggregate to pitcher-season %, then compare against BOTH the 13-zone version
and the ML-based cmd_value on K_rate/BB_rate/stickiness/stability.

Usage:
    python mlb_attack_zone_command.py
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
N_GREAT = 1  # only 4 tiers total now, so keep the Good/Great split proportionally narrower
N_GOOD = 1


def classify_attack_zone(plate_x, plate_z) -> np.ndarray:
    # coerce to plain float64 ndarrays first -- pandas nullable Float32 arrays (with
    # NaNs present) produce nullable BooleanArray masks on comparison, which numpy
    # fancy-indexing rejects; plain float64 + NaN just compares False, as intended.
    plate_x = np.asarray(plate_x, dtype=float)
    plate_z = np.asarray(plate_z, dtype=float)
    ax = np.abs(plate_x)
    heart = (ax <= 0.558) & (plate_z >= 1.833) & (plate_z <= 3.166)
    shadow = (ax <= 1.108) & (plate_z >= 1.166) & (plate_z <= 3.833) & ~heart
    chase = (ax <= 1.667) & (plate_z >= 0.500) & (plate_z <= 4.500) & ~heart & ~shadow
    out = np.full(plate_x.shape, "waste", dtype=object)
    out[chase] = "chase"
    out[shadow] = "shadow"
    out[heart] = "heart"
    return out


def build_zone_table(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[raw["pitch_family"].isin(FAMILIES) & raw["plate_x"].notna() & raw["plate_z"].notna()].copy()
    d["leverage"] = leverage_bucket(d["balls"].values, d["strikes"].values)
    d["attack_zone"] = classify_attack_zone(d["plate_x"].values, d["plate_z"].values)

    # descriptive: swing rate + called-strike rate over ALL pitches (not just swings) --
    # the fuller picture the video's framework and the prior swing-only view both miss
    d["is_called_strike"] = d["description"] == "called_strike"
    desc_all = d.groupby(["pitch_family", "leverage", "attack_zone"]).agg(
        n=("is_whiff", "size"), swing_rate=("is_swing", "mean"), called_strike_rate=("is_called_strike", "mean"))
    print("Descriptive: swing rate & called-strike rate by attack zone (pooled, all pitches):")
    print(desc_all.round(4).to_string())

    sw = d[d["is_swing"]]
    tbl = sw.groupby(["pitch_family", "leverage", "attack_zone"]).agg(
        n=("is_whiff", "size"), whiff_rate=("is_whiff", "mean")).reset_index()
    tbl = tbl[tbl["n"] >= 300]

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

    print("\nBuilding attack-zone table (pooled across all pitchers)...")
    zone_tbl = build_zone_table(raw)
    print("\nGood/Great tiers by (family, leverage):")
    print(zone_tbl.sort_values(["pitch_family", "leverage", "whiff_rate"], ascending=[True, True, False])
          .to_string(index=False))

    raw["leverage"] = leverage_bucket(raw["balls"].values, raw["strikes"].values)
    raw["attack_zone"] = classify_attack_zone(raw["plate_x"].values, raw["plate_z"].values)
    tagged = raw[raw["is_swing"] & raw["pitch_family"].isin(FAMILIES)].merge(
        zone_tbl[["pitch_family", "leverage", "attack_zone", "tier"]],
        on=["pitch_family", "leverage", "attack_zone"], how="left")
    tagged["tier"] = tagged["tier"].fillna("other")
    tagged["is_great"] = tagged["tier"] == "great"
    tagged["is_good_or_great"] = tagged["tier"].isin(["good", "great"])

    print("\nAggregating to pitcher-season attack-zone command scores...")
    season_zone = tagged.groupby(["pitcher", "game_year", "pitch_family"]).agg(
        n=("is_whiff", "size"), good_pct=("is_good_or_great", "mean")).reset_index()
    season_zone = season_zone[season_zone["n"] >= mf.MIN_N_FAMILY]

    wide = season_zone.pivot(index=["pitcher", "game_year"], columns="pitch_family",
                              values=["good_pct", "n"])
    wide.columns = [f"{a}_{b.lower()}" for a, b in wide.columns]
    wide = wide.reset_index()
    good_cols = [f"good_pct_{f.lower()}" for f in FAMILIES if f"good_pct_{f.lower()}" in wide.columns]
    wide["attack_good_pct_overall"] = wide[good_cols].mean(axis=1, skipna=True)

    # ---- compare against BOTH the 13-zone version and the ML cmd_value ---- #
    cmd = pd.read_parquet("output/count_leverage_command.parquet")
    panel = pd.read_parquet(PANEL_PATH)
    d = panel.merge(wide, on=["pitcher", "game_year"], how="inner").merge(
        cmd[["pitcher", "game_year", "cmd_value_overall"]], on=["pitcher", "game_year"], how="left")
    elig = d[(d["PA"] >= MIN_PA) & d["attack_good_pct_overall"].notna()]
    print(f"\nn eligible pitcher-seasons = {len(elig)}")

    print(f"\ncorr(attack_good_pct_overall, cmd_value_overall [ML]) = "
          f"{elig['attack_good_pct_overall'].corr(elig['cmd_value_overall']):.4f}")

    print("\n=== Relationship with K_rate and BB_rate (concurrent) ===")
    for col in good_cols + ["attack_good_pct_overall"]:
        ck = elig[col].corr(elig["K_rate"])
        cb = elig[col].corr(elig["BB_rate"])
        print(f"  {col:26s}  corr(K_rate)={ck:+.4f}   corr(BB_rate)={cb:+.4f}")
    print(f"  {'cmd_value_overall (ML)':26s}  corr(K_rate)={elig['cmd_value_overall'].corr(elig['K_rate']):+.4f}   "
          f"corr(BB_rate)={elig['cmd_value_overall'].corr(elig['BB_rate']):+.4f}")
    print("  (13-zone good_pct_overall from last time: corr(K_rate)=+0.1533, corr(BB_rate)=-0.1403)")

    print("\n=== Stickiness: year-over-year correlation ===")
    b = elig[["pitcher", "game_year", "attack_good_pct_overall"]].copy()
    b["game_year"] -= 1
    cp = elig.merge(b, on=["pitcher", "game_year"], suffixes=("1", "2"))
    print(f"  n couplets = {len(cp)}")
    print(f"  raw corr(attack_good_pct_overall_Y, _Y+1) = "
          f"{cp['attack_good_pct_overall1'].corr(cp['attack_good_pct_overall2']):.4f}  "
          f"(13-zone was 0.5376, ML cmd_value was 0.4728)")

    print("\n=== Stability: n0 ===")
    n_col_sum = wide[[f"n_{f.lower()}" for f in FAMILIES if f"n_{f.lower()}" in wide.columns]].fillna(0).sum(axis=1)
    elig2 = elig.copy()
    elig2["n_total"] = n_col_sum.reindex(elig2.index)
    n0_mm, info = mm.estimate_n0(elig2["attack_good_pct_overall"], elig2["n_total"], min_n=30)
    print(f"  n0 (method-of-moments) = {n0_mm:.1f}  {info}  (13-zone was 59.7, ML cmd_value was 495.7)")
