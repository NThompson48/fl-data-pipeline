"""
fl_hitter_ev.py -- hitter exit-velocity inputs for the projection: EV90 (90th-percentile exit velocity over every tracked batted ball, HR included; > 120 mph dropped as tracking error),
pooled over 2025-26, shrunk by its own reliability toward the league level.

n0 = 14.26 batted balls (random-half split reliability of full seasons, fl_hitter_evpct_test.py); league level = mean EV90 over hitter-seasons with >= 50 tracked balls (2025 100.17, 2026 100.06).
The cached ball table (output/hitter_bbe_2025_2026.parquet) is rebuilt from Supabase `pitches` when missing.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sqlalchemy import text

import fl_innings_projection as ip
from fl_supabase import read_sql_df

BBE = "output/hitter_bbe_2025_2026.parquet"
N0_EV90 = 14.26
MIN_BBE_LEAGUE = 50


def load_bbe(refresh: bool = False) -> pd.DataFrame:
    if os.path.exists(BBE) and not refresh:
        d = pd.read_parquet(BBE)
    else:
        d = read_sql_df(text("""select batter, year, exit_speed, angle, direction, batter_side, barrel, play_result from pitches
                                where game_type='reg_season' and year in (2025,2026)
                                  and play_result in ('single','double','triple','home_run','out','fielders_choice','error') and exit_speed is not null and angle is not null"""))
        d.to_parquet(BBE, index=False)
    d = d[d["exit_speed"] <= 120].copy()
    d["name_clean"] = d["batter"].apply(ip.lastfirst_to_clean)
    return d


def ev90_table(years=(2025, 2026), refresh: bool = False):
    """Per hitter: pooled EV90 and ball count over `years`; plus the league level and the shrunk deviation (mph)."""
    d = load_bbe(refresh)
    d = d[d["year"].isin(years)]
    per_year = d.groupby(["name_clean", "year"])["exit_speed"].agg(n="size", ev90=lambda s: np.quantile(s, 0.9)).reset_index()
    lg = float(per_year[per_year["n"] >= MIN_BBE_LEAGUE]["ev90"].mean())
    t = d.groupby("name_clean")["exit_speed"].agg(bbe="size", ev90=lambda s: np.quantile(s, 0.9)).reset_index()
    t["ev90_dev_shrunk"] = (t["ev90"] - lg) * t["bbe"] / (t["bbe"] + N0_EV90)
    return t, lg
