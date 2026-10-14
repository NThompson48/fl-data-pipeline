"""
fl_hitter_swing.py -- hitter Swing% (swings per pitch seen) for the BB% projection: pooled over 2025-26 pitches, shrunk toward the league level by n0 = 92.1 pitches (random-half split reliability, fl_hitter_bb_study.py).
Same pitch filter as the study that fit the coefficient (fl_hitter_k_study.pa_table): rows with a plate-appearance id, a tracked location and a zone flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_hitter_whiff as hwf
import fl_innings_projection as ip

N0_SWING = 92.1


def swing_table(years=(2025, 2026), refresh: bool = False):
    d = hwf.load(refresh)
    d = d[d["year"].isin(years) & d["pa_id"].notna() & d["plate_x"].notna() & d["outside_zone"].notna()].copy()
    d["name_clean"] = d["batter"].apply(ip.lastfirst_to_clean)
    t = pd.DataFrame({"name_clean": d["name_clean"], "swings_all": d["pitch_call"].isin(hwf.SWING).astype(int), "pitches": 1}).groupby("name_clean").sum().reset_index()
    lg = float(t["swings_all"].sum() / t["pitches"].sum())
    t["swing_pct_pool"] = t["swings_all"] / t["pitches"]
    t["swing_dev_shrunk"] = (t["swing_pct_pool"] - lg) * t["pitches"] / (t["pitches"] + N0_SWING)
    return t, lg
