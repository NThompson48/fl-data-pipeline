"""
pull_statscrew_batting_2022_2024.py -- Frontier League BATTING lines for 2022-2024 from the same
StatsCrew team-year pages already cached for the pitching pulls (statscrew.com/minorbaseball/stats/
t-{team}/y-{year}; the page carries a pitching table and a batting table). Supabase Trackman data
starts in 2025, so 2022-24 is where the extra hitter seasons for transition-level validation come from.

Batting table columns: Player GP PA AB R H 2B 3B HR RBI SB CS BB SO BA OBP SLG OPS TB DP HBP SH SF IBB

Output: output/fl_statscrew_batting_2022_2024.parquet  (one row per team stint)
Usage:  python pull_statscrew_batting_2022_2024.py
"""
from __future__ import annotations

import io

import pandas as pd

import fl_statscrew_pull as sc
import pull_statscrew_games_2022_2024 as pg

OUT_PATH = "output/fl_statscrew_batting_2022_2024.parquet"
NUM = ["GP", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "TB", "DP", "HBP", "SH", "SF", "IBB"]


def main():
    raw = sc.parse_raw()
    combos = sorted(set(zip(raw["team"], raw["year"])))
    rows, missing = [], []
    for team, year in combos:
        html = pg.fetch(team, int(year))
        if html is None:
            missing.append((team, year)); continue
        tabs = pd.read_html(io.StringIO(html))
        t = next((x for x in tabs if {"Player", "PA", "AB", "HR"} <= set(map(str, x.columns))), None)
        if t is None:
            missing.append((team, year)); continue
        t = t[t["Player"].astype(str).str.strip().ne("") & ~t["Player"].astype(str).str.contains("Total|Team", case=False, na=False)].copy()
        for c in NUM:
            if c in t.columns:
                t[c] = pd.to_numeric(t[c], errors="coerce")
        t["team"], t["year"] = team, int(year)
        rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={c: c.lower() for c in out.columns}).rename(columns={"2b": "x2b", "3b": "x3b"})
    out.to_parquet(OUT_PATH, index=False)
    print(f"saved {OUT_PATH}: {len(out)} rows; missing pages: {missing}")
    print(out.groupby("year").agg(rows=("player", "size"), pa=("pa", "sum"), hr=("hr", "sum"), so=("so", "sum"), bb=("bb", "sum"), hbp=("hbp", "sum"),
                                  sf=("sf", "sum"), ibb=("ibb", "sum"), pa_missing=("pa", lambda s: s.isna().mean())).round(3).to_string())


if __name__ == "__main__":
    main()
