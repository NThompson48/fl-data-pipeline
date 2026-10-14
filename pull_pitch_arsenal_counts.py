"""
Baseball Savant pull: pitch-type arsenal breadth by pitcher-season, 2021-2025.

Source: Savant "Pitch Arsenals" leaderboard (pitch usage %), one CSV per year:
    https://baseballsavant.mlb.com/leaderboard/pitch-arsenals?year=YYYY&type=n_&min=1&csv=true

A pitch type is counted as part of a pitcher's arsenal if Savant reports any
usage for it that season (cell is non-empty, including 0.0 = throws it but <0.05%).

Output: output/pitch_arsenal_counts_2021_2025.csv
Columns: season, pitcher (MLBAM id), name, n_pitch_types, n_fastball_types,
         n_breaking_types, plus the raw usage% for each pitch type.
"""

import io
import requests
import pandas as pd
from pybaseball import playerid_reverse_lookup

YEARS = range(2021, 2026)
URL = ("https://baseballsavant.mlb.com/leaderboard/pitch-arsenals"
       "?year={year}&type=n_&min=1&csv=true")

# Savant pitch-type codes -> group
FASTBALL = ["ff", "si", "fc"]              # 4-seam, sinker, cutter
BREAKING = ["sl", "cu", "st", "sv"]        # slider, curve, sweeper, slurve
OFFSPEED = ["ch", "fs", "kn"]              # changeup, splitter, knuckleball
ALL_CODES = FASTBALL + BREAKING + OFFSPEED

frames = []
for year in YEARS:
    r = requests.get(URL.format(year=year), timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.rename(columns={"last_name, first_name": "name"})
    df["season"] = year
    frames.append(df)
    print(f"{year}: {len(df)} pitchers")

raw = pd.concat(frames, ignore_index=True)

usage_cols = [f"n_{c}" for c in ALL_CODES]
for col in usage_cols:
    if col not in raw:
        raw[col] = pd.NA
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

has = raw[usage_cols].notna()
out = pd.DataFrame({
    "season": raw["season"],
    "pitcher": raw["pitcher"],
    "name": raw["name"],
    "n_pitch_types": has.sum(axis=1),
    "n_fastball_types": has[[f"n_{c}" for c in FASTBALL]].sum(axis=1),
    "n_breaking_types": has[[f"n_{c}" for c in BREAKING]].sum(axis=1),
})
out = pd.concat([out, raw[usage_cols]], axis=1)

# MLBAM id -> FanGraphs PlayerId (for merging with FanGraphs data, e.g. Stuff+)
xwalk = playerid_reverse_lookup(sorted(out["pitcher"].unique()), key_type="mlbam")
xwalk = xwalk[["key_mlbam", "key_fangraphs"]].rename(
    columns={"key_mlbam": "pitcher", "key_fangraphs": "PlayerId"})
xwalk["PlayerId"] = xwalk["PlayerId"].astype("Int64")
out = out.merge(xwalk, on="pitcher", how="left")
out.insert(2, "PlayerId", out.pop("PlayerId"))

out = out.sort_values(["season", "name"]).reset_index(drop=True)

dest = "output/pitch_arsenal_counts_2021_2025.csv"
out.to_csv(dest, index=False)
print(f"\nwrote {len(out)} pitcher-seasons -> {dest}")
print(out.head(10).to_string(index=False))
