"""
Pulls MLB pitcher birthdates from the public MLB Stats API (no auth needed) for
every distinct pitcher id found across the Statcast parquet files in this repo.
Batches requests (~100 ids per call) and caches to output/pitcher_bio.parquet.

Mirrors pull_mlb_heights.py's exact pattern, extended to bio (birthDate,
mlbDebutDate) rather than height/weight, and pitcher-focused rather than
batter-focused.

Usage:
    python pull_pitcher_bio.py
"""
import glob
import time

import pandas as pd
import requests

OUT_PATH = "output/pitcher_bio.parquet"


def collect_pitcher_ids():
    ids = set()
    for path in glob.glob("statcast_*.parquet"):
        try:
            d = pd.read_parquet(path, columns=["pitcher"])
        except Exception:
            continue
        ids.update(d["pitcher"].dropna().unique().tolist())
    return sorted(int(i) for i in ids)


def fetch_bio(pitcher_ids, batch_size=100):
    rows = []
    ids = list(pitcher_ids)
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(str(b) for b in batch)}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        people = r.json().get("people", [])
        for p in people:
            rows.append({
                "pitcher_id": p["id"],
                "full_name": p.get("fullName"),
                "birth_date": p.get("birthDate"),
                "mlb_debut_date": p.get("mlbDebutDate"),
                "pitch_hand": (p.get("pitchHand") or {}).get("code"),
            })
        print(f"  fetched {i + len(batch)}/{len(ids)}")
        time.sleep(0.2)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pitcher_ids = collect_pitcher_ids()
    print(f"Fetching bio for {len(pitcher_ids)} distinct MLB pitchers...")
    bio = fetch_bio(pitcher_ids)
    bio["birth_date"] = pd.to_datetime(bio["birth_date"], errors="coerce")
    print(f"Got {bio['birth_date'].notna().sum()} / {len(bio)} with parseable birth_date")
    bio.to_parquet(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
