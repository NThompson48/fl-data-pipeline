"""
pull_statscrew_pioneer_batting.py -- Pioneer League BATTING, 2021-2025, from StatsCrew (user 2026-10-14: "Use Statscrew for pre-2026 Pioneer hitting").
2021+ = the independent era (MLB rookie league before).  For each season: the league page statscrew.com/minorbaseball/l-PION/y-{year} lists the team-year pages
(statscrew.com/minorbaseball/stats/t-{team}/y-{year}); each team page carries a pitching table and a batting table (same layout as the FL pages parsed by
pull_statscrew_batting_2022_2024.py).  robots.txt allows these pages with Crawl-Delay 5 -> 5 s between requests; pages cached in output/statscrew_pages_pioneer/.
Output: output/pioneer_batting_2021_2025.parquet (one row per team stint: player, team, year, GP PA AB R H 2B 3B HR RBI SB CS BB SO TB DP HBP SH SF IBB).
"""
from __future__ import annotations

import io
import os
import re
import time

import pandas as pd
import requests

CACHE = "output/statscrew_pages_pioneer"; OUT = "output/pioneer_batting_2021_2025.parquet"
LEAGUE = "https://www.statscrew.com/minorbaseball/l-PION/y-{year}"; TEAM = "https://www.statscrew.com/minorbaseball/stats/t-{team}/y-{year}"
NUM = ["GP", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "TB", "DP", "HBP", "SH", "SF", "IBB"]
_last = [0.0]


def get(url: str, cache_name: str) -> str | None:
    os.makedirs(CACHE, exist_ok=True); f = os.path.join(CACHE, cache_name)
    if os.path.exists(f):
        return open(f, encoding="utf-8", errors="ignore").read()
    for _ in range(3):
        wait = 5.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30); _last[0] = time.time()
            if r.status_code == 200:
                open(f, "w", encoding="utf-8").write(r.text); return r.text
        except Exception:
            _last[0] = time.time()
    return None


def main():
    rows, missing = [], []
    for year in range(2021, 2026):
        html = get(LEAGUE.format(year=year), f"league_{year}.html")
        teams = sorted(set(re.findall(r"/minorbaseball/stats/t-([a-z]+\d+)/y-" + str(year), html or "")))
        print(f"{year}: {len(teams)} teams", flush=True)
        for t in teams:
            page = get(TEAM.format(team=t, year=year), f"{t}_{year}.html")
            if page is None:
                missing.append((t, year)); continue
            tabs = pd.read_html(io.StringIO(page))
            b = next((x for x in tabs if {"Player", "PA", "AB", "HR"} <= set(map(str, x.columns))), None)
            if b is None:
                missing.append((t, year)); continue
            b = b[b["Player"].astype(str).str.strip().ne("") & ~b["Player"].astype(str).str.contains("Total|Team", case=False, na=False)].copy()
            for c in NUM:
                if c in b.columns:
                    b[c] = pd.to_numeric(b[c], errors="coerce")
            m = re.search(r"<title>\s*\d{4}\s+(.*?)\s+minor league", page, re.S)          # page title "2023 Boise Hawks minor league baseball Statistics ..."
            b["team_id"], b["team"], b["year"] = t, (m.group(1).strip() if m else t), year; rows.append(b)
    out = pd.concat(rows, ignore_index=True).rename(columns=lambda c: str(c).lower()).rename(columns={"2b": "x2b", "3b": "x3b"})
    out.to_parquet(OUT, index=False)
    print(f"saved {OUT}: {len(out)} team stints; missing pages: {missing}")
    print(out.groupby("year").agg(teams=("team", "nunique"), rows=("player", "size"), pa=("pa", "sum"), hr=("hr", "sum"), so=("so", "sum"), bb=("bb", "sum"),
                                  hbp=("hbp", "sum"), pa_missing=("pa", lambda s: s.isna().mean())).round(3).to_string())


if __name__ == "__main__":
    main()
