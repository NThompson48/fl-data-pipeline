"""
pull_statscrew_games_2022_2024.py -- adds GAMES (and starts where StatsCrew has them) for the
Frontier League 2022-2024 pitcher-seasons. The earlier StatsCrew pull (fl_statscrew_pull.py)
kept only IP/H/BB/HBP/SO/BF, so role (innings per appearance, starter share) could only be
used for 2025-26 (war_pitchers). Same pages, same team-year list as that pull
(statscrew.com/minorbaseball/stats/t-{team}/y-{year}), 1.2 s between requests.

Output: output/fl_statscrew_pitching_games_2022_2024.parquet
        (team, year, player, g, gs, ip, h, bb, so, bf)   -- one row per team stint

Usage:  python pull_statscrew_games_2022_2024.py
"""
from __future__ import annotations

import io
import os
import time

import pandas as pd
import requests

import fl_statscrew_pull as sc

OUT_PATH = "output/fl_statscrew_pitching_games_2022_2024.parquet"
CACHE = r"C:\Users\215nt\AppData\Local\Temp\claude\C--Users-215nt-PycharmProjects-FLStuff-FLHitting\e96abeb0-fb3e-4d45-adec-9b81c366e683\scratchpad\statscrew_pages"
URL = "https://www.statscrew.com/minorbaseball/stats/t-{team}/y-{year}"


def fetch(team: str, year: int) -> str | None:
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{team}_{year}.html")
    if os.path.exists(f):
        return open(f, encoding="utf-8", errors="ignore").read()
    for attempt in range(3):
        try:
            r = requests.get(URL.format(team=team, year=year), headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if r.status_code == 200:
                open(f, "w", encoding="utf-8").write(r.text)
                time.sleep(1.2)
                return r.text
        except Exception:
            pass
        time.sleep(2.0)
    return None


def main():
    raw = sc.parse_raw()
    combos = sorted(set(zip(raw["team"], raw["year"])))
    print(f"{len(combos)} team-year pages")
    rows, missing = [], []
    for team, year in combos:
        html = fetch(team, int(year))
        if html is None:
            missing.append((team, year)); continue
        tabs = pd.read_html(io.StringIO(html))
        t = next((x for x in tabs if {"Player", "IP", "G"} <= set(map(str, x.columns))), None)
        if t is None:
            missing.append((team, year)); continue
        t = t[t["Player"].astype(str).str.strip().ne("") & ~t["Player"].astype(str).str.contains("Total|Team", case=False, na=False)].copy()
        for c in ("G", "GS", "IP", "H", "BB", "SO", "BF"):
            if c in t.columns:
                t[c] = pd.to_numeric(t[c], errors="coerce")
        t["team"], t["year"] = team, int(year)
        rows.append(t.rename(columns={"Player": "player", "G": "g", "GS": "gs", "IP": "ip", "H": "h", "BB": "bb", "SO": "so", "BF": "bf"})
                    [["team", "year", "player", "g", "gs", "ip", "h", "bb", "so", "bf"]])
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"saved {OUT_PATH}: {len(out)} rows; missing pages: {missing}")
    print("GS available:", out.groupby("year")["gs"].apply(lambda s: s.notna().mean()).round(2).to_dict())
    # reconcile with the earlier pull (IP by team-year)
    a = out.groupby(["team", "year"])["ip"].sum(); b = raw.groupby(["team", "year"])["ip"].sum()
    j = pd.concat([a.rename("new"), b.rename("old")], axis=1).dropna()
    print("IP reconciliation vs earlier pull: max |diff| %.1f, share exactly equal %.2f" % ((j.new - j.old).abs().max(), ((j.new - j.old).abs() < 0.05).mean()))


if __name__ == "__main__":
    main()
