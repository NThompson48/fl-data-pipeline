"""
fl_statscrew_pull.py -- parses the StatsCrew.com Frontier League team-year pitching
pages (pulled via browser fetch this session, raw text saved to the scratchpad) into
a per-pitcher-season K%/BB% panel for 2022-2024, the years not covered by Supabase
Trackman data (which starts 2025).

Data facts confirmed directly against the live site:
  - Pitching table columns: Player W L W% ERA G GS GF CG SHO SV IP H R ER HR BB IBB
    SO HBP BK WP BF. Names are "First Last".
  - BF availability is year-dependent: 2022 has real BF populated; 2023 has BOTH BF
    and HBP missing; 2024 has HBP but not BF.
  - IP is encoded in DECIMAL THIRDS (not baseball's .1/.2 notation): x.3 = 1/3 inning,
    x.7 = 2/3 inning. outs = round(IP * 3).
  - Fallback BF (used only when the real BF column is missing):
    BF_approx = round(IP*3) + H + BB + HBP ("sum of all [PA-ending] events" per the
    user's own phrasing) -- an approximation, doesn't capture reached-on-error/sac
    edge cases, confirmed directly to not match real BF exactly on a real example.
  - Mid-season trades: a traded pitcher has a separate partial-season row on EACH
    team's page for that year -- aggregated by summing BF/SO/BB across team-stints
    WITHIN a season before computing that season's K%/BB%, per the user's explicit
    instruction.

Usage:
    python fl_statscrew_pull.py
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

RAW_PATH = (r"C:\Users\215nt\AppData\Local\Temp\claude\C--Users-215nt-PycharmProjects-"
            r"FLStuff-FLHitting\10970283-d3af-4ce4-a21e-e91dde5580c0\scratchpad\statscrew_raw.txt")
OUT_PATH = "output/fl_statscrew_pitching_2022_2024.parquet"


# Name aliases (user 2026-10-13, from the 10-08 name / id audit fl_name_id_audit.py): one person recorded under two spellings -> the majority spelling.
NAME_ALIASES = {"teag gollert": "taeg gollert", "jake houtz": "jacob houtz", "james gilleran": "jimmy gilleran"}


def clean_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return NAME_ALIASES.get(s, s)


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

    print("\n--- BF fallback sanity check (2022 real vs. fallback formula, same rows) ---")
    chk = raw[(raw["year"] == 2022) & (~raw["bf_is_fallback"])].copy()
    outs_chk = (chk["ip"] * 3).round()
    chk["bf_fallback_formula"] = outs_chk + chk["h"].fillna(0) + chk["bb"].fillna(0) + chk["hbp"].fillna(0)
    chk["diff"] = chk["bf_fallback_formula"] - chk["bf_real"]
    print(f"n={len(chk)}, mean diff={chk['diff'].mean():.2f}, "
          f"median abs diff={chk['diff'].abs().median():.1f}, "
          f"within 5 of real: {(chk['diff'].abs() <= 5).mean():.1%}")
