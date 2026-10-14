"""
pull_milb_hitting.py -- season-level minor-league HITTING stats (AAA/AA/High-A/Single-A/Rookie), 2018-2026, from the public MLB Stats API
(same endpoint, level codes and playerPool=ALL fix as pull_milb_pitching.py; player.id is the MLBAM id used elsewhere in the project).  Hitter analogue for the hitter other-league bridge (user 2026-10-09: "need to make a bridge for hitters").
Usage:  python pull_milb_hitting.py
"""
import time

import pandas as pd
import requests

LEVELS = {11: "AAA", 12: "AA", 13: "A+", 14: "A", 16: "ROK"}
YEARS = list(range(2018, 2027))
OUT_PATH = "output/milb_hitting_2018_2026.parquet"
FIELDS = ["plateAppearances", "atBats", "hits", "doubles", "triples", "homeRuns", "baseOnBalls", "intentionalWalks", "hitByPitch", "strikeOuts", "sacFlies", "gamesPlayed", "stolenBases"]


def fetch_level_year(sport_id: int, year: int) -> pd.DataFrame:
    url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&sportId={sport_id}&season={year}&playerPool=ALL&limit=3000"
    r = requests.get(url, timeout=30); r.raise_for_status(); data = r.json()
    if not data.get("stats") or not data["stats"][0].get("splits"):
        return pd.DataFrame()
    rows = []
    for s in data["stats"][0]["splits"]:
        st = s["stat"]
        rows.append({"batter": s["player"]["id"], "player_name": s["player"]["fullName"], "season": year, "level_id": sport_id, "level": LEVELS[sport_id], "team": s.get("team", {}).get("name"), "league": s.get("league", {}).get("name"), **{f: st.get(f) for f in FIELDS}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    frames = []
    for sport_id, name in LEVELS.items():
        for year in YEARS:
            df = fetch_level_year(sport_id, year); print(f"  {name} {year}: {len(df)} hitter-seasons", flush=True); frames.append(df); time.sleep(0.3)
    full = pd.concat(frames, ignore_index=True)
    for c in FIELDS:
        full[c] = pd.to_numeric(full[c], errors="coerce")
    print(f"\ntotal: {len(full):,} rows"); full.to_parquet(OUT_PATH, index=False); print(f"Saved -> {OUT_PATH}")
