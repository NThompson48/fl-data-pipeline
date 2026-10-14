"""
Builds a pitcher MLBAM-id -> name crosswalk from the Statcast parquet files
already in this repo. `player_name` is the PITCHER's own name on every row of a
raw Baseball Savant / pybaseball export (confirmed) -- no extra pull needed.

Usage:
    python build_pitcher_id_name_map.py
"""
import glob

import pandas as pd

OUT_PATH = "output/pitcher_id_name_map.parquet"

if __name__ == "__main__":
    frames = []
    for path in sorted(glob.glob("statcast_*.parquet")):
        d = pd.read_parquet(path, columns=["pitcher", "player_name"])
        frames.append(d)
    all_rows = pd.concat(frames, ignore_index=True)
    name_map = (all_rows.dropna(subset=["pitcher"])
                .groupby("pitcher")["player_name"].agg(lambda s: s.mode().iloc[0])
                .reset_index().rename(columns={"pitcher": "pitcher_id", "player_name": "name"}))
    name_map["pitcher_id"] = name_map["pitcher_id"].astype(int)
    print(f"{len(name_map)} distinct pitcher ids mapped")
    name_map.to_parquet(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
