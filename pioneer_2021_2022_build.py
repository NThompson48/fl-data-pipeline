"""Parses scratch_pioneer_2021_2022.py's embedded 2021-2022 Pioneer League
pitching tables (fetched live from StatsCrew.com this session) into the same
per-pitcher-season panel shape as the existing 2023-2025 pull
(fl_pioneer_pull.py's build_season_panel), then concats with the existing
output/fl_pioneer_pitching_2023_2025.parquet and saves the combined
2021-2025 panel. Real BF column is populated for every team-year here (same
as already confirmed for 2023-2025), so no fallback-BF path is needed."""
import re
import numpy as np
import pandas as pd

from scratch_pioneer_2021_2022 import TABLES


def clean_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_table(text: str) -> pd.DataFrame:
    rows = []
    for line in text.strip("\n").split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 23:
            print(f"  skipping malformed line ({len(parts)} fields): {line[:60]}")
            continue
        player = parts[0]
        ip, h, bb, so, bf = parts[11], parts[12], parts[16], parts[18], parts[22]
        hbp = parts[19]
        rows.append({"player": player, "ip": float(ip), "h": float(h), "bb": float(bb),
                      "hbp": float(hbp), "so": float(so), "bf_real": float(bf)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    frames = []
    for (team, year), text in TABLES.items():
        d = parse_table(text)
        d["team"] = team
        d["year"] = year
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    print(f"parsed {len(raw):,} team-stint rows across {raw['team'].nunique()} teams, years {sorted(raw['year'].unique())}")

    outs = (raw["ip"] * 3).round()
    bf_fallback = outs + raw["h"].fillna(0) + raw["bb"].fillna(0) + raw["hbp"].fillna(0)
    raw["bf"] = raw["bf_real"].where(raw["bf_real"].notna() & (raw["bf_real"] > 0), bf_fallback)
    raw["bf_is_fallback"] = raw["bf_real"].isna() | (raw["bf_real"] == 0)
    print(f"BF source: {(~raw['bf_is_fallback']).sum()} real, {raw['bf_is_fallback'].sum()} fallback-computed")

    raw["name_clean"] = raw["player"].apply(clean_name)
    new_panel = raw.groupby(["name_clean", "year"]).agg(
        bf=("bf", "sum"), so=("so", "sum"), bb=("bb", "sum"),
        n_teams=("team", "nunique"),
        teams=("team", lambda s: ",".join(sorted(s.unique()))),
        player=("player", "first"),
        any_fallback=("bf_is_fallback", "any"),
    ).reset_index()
    new_panel["k_pct"] = new_panel["so"] / new_panel["bf"]
    new_panel["bb_pct"] = new_panel["bb"] / new_panel["bf"]
    print(f"new 2021-2022 panel: {len(new_panel)} pitcher-seasons")

    existing = pd.read_parquet("output/fl_pioneer_pitching_2023_2025.parquet")
    print(f"existing 2023-2025 panel: {len(existing)} pitcher-seasons")

    combined = pd.concat([new_panel, existing], ignore_index=True)
    dupe = combined.duplicated(["name_clean", "year"]).sum()
    print(f"duplicate (name_clean, year) collisions between old/new: {dupe}")

    out_path = "output/fl_pioneer_pitching_2021_2025.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}, shape={combined.shape}")
    print(combined["year"].value_counts().sort_index())
