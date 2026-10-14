"""
fl_position_player_flag.py -- flag position players who appear in the pitcher tables (mop-up innings), so they do not get a K% / BB% / RA9 projection or crowd the pitcher leaderboards.

Two independent rules (a pitcher is flagged when EITHER fires; both are recorded):
  A  hitter record:  >= 100 PA as a hitter in 2025-26 (fl_hitter_panel, matched on cleaned name) AND <= 60 batters faced as a pitcher.
  V  pitch characteristics:  fastball-family (four-seam / sinker / cutter) velocity an OUTLIER LOW: >= 5 tracked fastballs and a robust z <= -5 against pitchers who are not rule-A
     (robust z = (velo - median) / (1.4826 MAD); league median 88.5 mph, robust SD 2.47 -> cut ~76 mph).  Position players throw 48-71 mph; the slowest real pitchers 77-81 mph
     (the gap between the groups is 71.3 -> 77.0 mph, so the cut is not knife-edge).  Catches position players the hitter rule misses (>60 BF, or <100 hitter PA).
Velocity arbitrates rule A: with >= 5 tracked fastballs and a NORMAL velocity (robust z > -3) the hitter-record match is not trusted (a real two-way player, or a pitcher who shares a
name with a hitter: e.g. 87-93 mph) and the pitcher is NOT flagged (recorded as `possible_two_way`).  Rule A stands when velocity is unverifiable (< 5 tracked fastballs) or in the low tail (z <= -3).
Output: output/position_player_pitchers.parquet (copied to the dashboard data folder).
"""
import numpy as np
import pandas as pd

import fl_innings_projection as ip
import fl_pitcher_features as fp

HIT_PA_MIN, BF_MAX, MIN_FB, Z_CUT = 100, 60, 5, -5.0
Z_NORMAL = -3.0


def build():
    raw = fp.load_fl_pitches_cached(); raw = raw[raw["year"] >= 2025].copy(); raw["nc"] = raw["pitcher"].apply(ip.lastfirst_to_clean)
    g = raw.groupby("pitcher_id")
    S = pd.DataFrame({"pitcher": g["pitcher"].first(), "name_clean": g["nc"].first(), "pitches": g.size(), "bf": g["is_terminal"].sum(), "tracked_frac": g["effective_velo"].count() / g.size()})
    fb = raw[raw["pitch_type"].isin(["four_seam", "sinker", "cutter"]) & raw["effective_velo"].notna()]
    S = S.join(fb.groupby("pitcher_id")["effective_velo"].agg(fb_velo="mean", fb_n="size"))
    h = pd.read_parquet("output/fl_hitter_panel.parquet"); S["hit_pa"] = S["name_clean"].map(h[h["year"] >= 2025].groupby("name_clean")["pa"].sum()).fillna(0)
    S["rule_hitter_pa"] = (S["hit_pa"] >= HIT_PA_MIN) & (S["bf"] <= BF_MAX)
    ref = S[(S["fb_n"] >= 30) & ~S["rule_hitter_pa"]]["fb_velo"]; med = ref.median(); mad = 1.4826 * (ref - med).abs().median()
    S["velo_z"] = (S["fb_velo"] - med) / mad
    S["rule_low_velo"] = (S["fb_n"] >= MIN_FB) & (S["velo_z"] <= Z_CUT)
    normal_velo = (S["fb_n"].fillna(0) >= MIN_FB) & (S["velo_z"] > Z_NORMAL)
    S["possible_two_way"] = S["rule_hitter_pa"] & normal_velo & ~S["rule_low_velo"]
    S["is_position_player"] = S["rule_low_velo"] | (S["rule_hitter_pa"] & ~normal_velo)
    S["reason"] = np.select([S["rule_hitter_pa"] & S["rule_low_velo"], S["rule_low_velo"], S["rule_hitter_pa"] & ~normal_velo & (S["fb_n"].fillna(0) >= MIN_FB), S["rule_hitter_pa"] & ~normal_velo],
                            ["hitter record + low velocity", "low velocity only", "hitter record + below-normal velocity (z <= -3)", "hitter record (velocity unverifiable: < 5 tracked fastballs)"], "")
    out = S[S["is_position_player"] | S["possible_two_way"]].reset_index(); out["reason"] = np.where(out["possible_two_way"], "NOT flagged: hitter record but normal velocity (two-way or name match)", out["reason"])
    out.to_parquet("output/position_player_pitchers.parquet", index=False)
    print(f"league fastball velocity median {med:.1f} mph, robust SD {mad:.2f}; cut z <= {Z_CUT} (~{med + Z_CUT * mad:.1f} mph)")
    print(f"flagged {int(out['is_position_player'].sum())} of {len(S)} pitchers; not flagged but hitter record + normal velocity: {int(out['possible_two_way'].sum())}")
    print(out["reason"].value_counts().to_string())
    print(out[out["is_position_player"] & ~out["rule_hitter_pa"] | out["reason"].str.contains("below-normal")][["pitcher", "bf", "hit_pa", "fb_n", "fb_velo", "velo_z", "reason"]].round(1).to_string(index=False))
    return out


if __name__ == "__main__":
    build()


def exclude(raw, path="output/position_player_pitchers.parquet"):
    """Drop flagged position players (is_position_player) from a raw pitch frame before any pitcher model is fit or projected.  `possible_two_way` rows are kept."""
    f = pd.read_parquet(path); ids = set(f.loc[f["is_position_player"], "pitcher_id"])
    out = raw[~raw["pitcher_id"].isin(ids)]
    print(f"  excluded {len(ids)} flagged position players ({len(raw) - len(out):,} pitches) from the pitch data")
    return out
