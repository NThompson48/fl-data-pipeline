"""
fl_hitter_whiff.py -- hitter Whiff% (whiffs per swing) for the K% projection: pooled over 2025-26 pitches, shrunk toward the league level by n0 = 52.5 swings (random-half split reliability).
Cached pitch table: output/hitter_pitches_2025_2026.parquet (rebuilt from Supabase `pitches` when missing).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sqlalchemy import text

import fl_innings_projection as ip
from fl_supabase import read_sql_df

PITCHES = "output/hitter_pitches_2025_2026.parquet"
N0_WHIFF = 52.5
SWING = {"strike_swinging", "foul_not_fieldable", "foul_fieldable", "foul_ball", "in_play"}


def load(refresh: bool = False) -> pd.DataFrame:
    if os.path.exists(PITCHES) and not refresh:
        return pd.read_parquet(PITCHES)
    d = read_sql_df(text("""select batter, year, game_id, pa_id, pitch_no, pitch_call, k_or_bb, plate_x, plate_z, sz_top, sz_bot, outside_zone, outside_zone_adjusted, batter_side, balls, strikes
                            from pitches where game_type='reg_season' and year in (2025,2026) and pitch_call is not null"""))
    d.to_parquet(PITCHES, index=False)
    return d


def whiff_table(years=(2025, 2026), refresh: bool = False):
    d = load(refresh)
    d = d[d["year"].isin(years)].copy()
    d["name_clean"] = d["batter"].apply(ip.lastfirst_to_clean)
    sw = d["pitch_call"].isin(SWING); wh = d["pitch_call"].eq("strike_swinging")
    t = pd.DataFrame({"name_clean": d["name_clean"], "swings": sw.astype(int), "whiffs": wh.astype(int)}).groupby("name_clean").sum().reset_index()
    lg = float(t["whiffs"].sum() / t["swings"].sum())
    t["whiff_swing_pct"] = t["whiffs"] / t["swings"].replace(0, np.nan)
    t["whiff_dev_shrunk"] = (t["whiff_swing_pct"] - lg) * t["swings"] / (t["swings"] + N0_WHIFF)
    return t, lg
