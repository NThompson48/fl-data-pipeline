"""
Pulls MLB player height/weight from the public MLB Stats API (no auth needed)
for every batter_id in output/mlb_bat_speed_pipeline_final.csv. Batches
requests (~100 IDs per call) and caches to output/mlb_player_heights.csv.

Usage:
    python pull_mlb_heights.py
"""

import re
import time

import pandas as pd
import requests

OUT_PATH = "output/mlb_player_heights.csv"


def parse_height_to_inches(height_str):
    if not height_str:
        return None
    m = re.match(r"(\d+)'\s*(\d+)", height_str)
    if not m:
        return None
    feet, inches = int(m.group(1)), int(m.group(2))
    return feet * 12 + inches


def fetch_heights(batter_ids, batch_size=100):
    rows = []
    ids = list(batter_ids)
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(str(b) for b in batch)}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        people = r.json().get("people", [])
        for p in people:
            rows.append({
                "batter": p["id"],
                "full_name": p.get("fullName"),
                "height_str": p.get("height"),
                "height_in": parse_height_to_inches(p.get("height")),
                "weight_lb": p.get("weight"),
            })
        print(f"  fetched {i + len(batch)}/{len(ids)}")
        time.sleep(0.2)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mlb_est = pd.read_csv("output/mlb_bat_speed_pipeline_final.csv")
    batter_ids = mlb_est["batter"].unique()
    print(f"Fetching heights for {len(batter_ids)} MLB batters...")
    heights = fetch_heights(batter_ids)
    print(f"Got {heights['height_in'].notna().sum()} / {len(heights)} with parseable height")
    heights.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
