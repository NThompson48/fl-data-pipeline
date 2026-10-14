"""
fl_pioneer_pull.py -- parses the StatsCrew.com Pioneer League team-year pitching
pages (pulled via browser fetch this session, raw text saved to the scratchpad) into
a per-pitcher-season K%/BB% panel for 2023-2025, covering the modern-era
(fully independent since 2021) Pioneer League teams: Boise Hawks, Glacier Range
Riders, Grand Junction Jackalopes, Great Falls Voyagers, Idaho Falls Chukars,
Missoula Paddleheads, Northern Colorado Owlz, Oakland Ballers (2024+), Ogden
Raptors, Rocky Mountain Vibes.

Same page structure, same BF/HBP year-dependent-availability pattern, and same
IP-in-decimal-thirds encoding already confirmed for the Frontier League pull in
fl_statscrew_pull.py -- this script mirrors that one's parsing/BF-fallback/
mid-season-trade-aggregation logic exactly, pointed at the Pioneer raw dump instead.

Usage:
    python fl_pioneer_pull.py
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

RAW_PATH = (r"C:\Users\215nt\AppData\Local\Temp\claude\C--Users-215nt-PycharmProjects-"
            r"FLStuff-FLHitting\10970283-d3af-4ce4-a21e-e91dde5580c0\scratchpad\pioneer_raw.txt")
OUT_PATH = "output/fl_pioneer_pitching_2023_2025.parquet"


def clean_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_float(x: str) -> float:
    x = x.strip()
    return float(x) if x else np.nan


def parse_raw(path: str = RAW_PATH) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != 9:
                print(f"  skipping malformed line: {line[:80]}")
                continue
            team, year, player, ip, h, bb, hbp, so, bf = parts
            rows.append({
                "team": team, "year": int(year), "player": player,
                "ip": _to_float(ip), "h": _to_float(h), "bb": _to_float(bb),
                "hbp": _to_float(hbp), "so": _to_float(so), "bf_real": _to_float(bf),
            })
    df = pd.DataFrame(rows)
    print(f"parsed {len(df):,} team-stint rows across {df['team'].nunique()} teams, "
          f"years {sorted(df['year'].unique())}")
    return df


def compute_bf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    outs = (df["ip"] * 3).round()
    bf_fallback = outs + df["h"].fillna(0) + df["bb"].fillna(0) + df["hbp"].fillna(0)
    df["bf"] = df["bf_real"].where(df["bf_real"].notna() & (df["bf_real"] > 0), bf_fallback)
    df["bf_is_fallback"] = df["bf_real"].isna() | (df["bf_real"] == 0)
    return df


def build_season_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by (normalized name, year), SUMMING bf/so/bb across team-stints --
    the mid-season-trade handling."""
    df = df.copy()
    df["name_clean"] = df["player"].apply(clean_name)
    agg = df.groupby(["name_clean", "year"]).agg(
        bf=("bf", "sum"), so=("so", "sum"), bb=("bb", "sum"),
        n_teams=("team", "nunique"),
        teams=("team", lambda s: ",".join(sorted(s.unique()))),
        player=("player", "first"),
        any_fallback=("bf_is_fallback", "any"),
    ).reset_index()
    agg["k_pct"] = agg["so"] / agg["bf"]
    agg["bb_pct"] = agg["bb"] / agg["bf"]
    return agg


if __name__ == "__main__":
    raw = parse_raw()
    raw = compute_bf(raw)
    print(f"\nBF source: {(~raw['bf_is_fallback']).sum()} real, {raw['bf_is_fallback'].sum()} fallback-computed")

    panel = build_season_panel(raw)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}, shape={panel.shape}")
    print(panel["year"].value_counts().sort_index())

    print("\n--- Mid-season-trade spot check: players with n_teams > 1 ---")
    traded = panel[panel["n_teams"] > 1].sort_values(["year", "player"])
    print(f"{len(traded)} pitcher-seasons involved a mid-season trade")
    print(traded[["player", "year", "teams", "bf", "so", "bb", "k_pct", "bb_pct"]].head(15).to_string(index=False))
