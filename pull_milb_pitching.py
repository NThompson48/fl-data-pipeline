"""
pull_milb_pitching.py -- pulls season-level minor-league pitching stats
(AAA/AA/High-A/Single-A/Rookie) from the public MLB Stats API, 2018-2026,
matching the MLB Statcast data's coverage window. Uses playerPool=ALL (the
default endpoint silently restricts to "qualified" pitchers only -- confirmed
empirically: 22 vs. 1,234 AAA pitcher-seasons for 2024 alone).

player.id in this endpoint is the SAME MLBAM id space used throughout this
project's MLB Statcast/bio pulls -- no separate crosswalk needed.

2026 added to enable testing a same-year ("mixed-league season") MiLB->FL
bridge match for players who arrive in FL mid-season -- see DECISION_LOG.md.
Confirmed directly the API already has full 2026 season data (unlike
StatsCrew's Pioneer League pages, which are still empty for 2026 as of this
pull).

Usage:
    python pull_milb_pitching.py
"""
import time

import pandas as pd
import requests

LEVELS = {11: "AAA", 12: "AA", 13: "A+", 14: "A", 16: "ROK"}
YEARS = list(range(2018, 2027))
OUT_PATH = "output/milb_pitching_2018_2026.parquet"

FIELDS = ["strikeOuts", "baseOnBalls", "battersFaced", "inningsPitched",
          "gamesPlayed", "gamesStarted", "era", "whip"]


def fetch_level_year(sport_id: int, year: int) -> pd.DataFrame:
    url = (f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=pitching"
           f"&sportId={sport_id}&season={year}&playerPool=ALL&limit=3000")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("stats") or not data["stats"][0].get("splits"):
        return pd.DataFrame()
    rows = []
    for s in data["stats"][0]["splits"]:
        stat = s["stat"]
        rows.append({
            "pitcher": s["player"]["id"],
            "player_name": s["player"]["fullName"],
            "season": year,
            "level_id": sport_id,
            "level": LEVELS[sport_id],
            "team": s.get("team", {}).get("name"),
            "league": s.get("league", {}).get("name"),
            **{f: stat.get(f) for f in FIELDS},
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    frames = []
    for sport_id, level_name in LEVELS.items():
        for year in YEARS:
            df = fetch_level_year(sport_id, year)
            print(f"  {level_name} {year}: {len(df)} pitcher-seasons")
            frames.append(df)
            time.sleep(0.3)
    full = pd.concat(frames, ignore_index=True)
    full["inningsPitched"] = pd.to_numeric(full["inningsPitched"], errors="coerce")
    full["strikeOuts"] = pd.to_numeric(full["strikeOuts"], errors="coerce")
    full["baseOnBalls"] = pd.to_numeric(full["baseOnBalls"], errors="coerce")
    full["battersFaced"] = pd.to_numeric(full["battersFaced"], errors="coerce")
    full["K_rate"] = full["strikeOuts"] / full["battersFaced"]
    full["BB_rate"] = full["baseOnBalls"] / full["battersFaced"]
    print(f"\ntotal: {len(full):,} rows")
    full.to_parquet(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
