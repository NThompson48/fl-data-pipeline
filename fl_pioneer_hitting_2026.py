"""
fl_pioneer_hitting_2026.py -- ingest the user's 2026 PIONEER LEAGUE HITTING (Baseball Reference team tables the user pasted 2026-10-10; the file has all 12 teams: 11 pasted 2026-10-10 (two duplicated pastes removed) + the Boise Hawks pasted 2026-10-13).
(1) parse output/pioneer_2026_raw/pioneer_2026_br_hitting.csv (blocks '#BLOCK n'), (2) CHECK each block against the team totals row that came with the paste, (3) aggregate players who appear on two teams (same BR id), (4) league means (PA-weighted),
(5) save output/pioneer_hitting_2026.parquet: one row per player, counting stats, rates, wOBA proxy (same weights as the MiLB pull), K%, BB%, HR/PA, batting side from the BR marker (* left, # switch).
"""
import re

import numpy as np
import pandas as pd

RAW = "output/pioneer_2026_raw/pioneer_2026_br_hitting.csv"; OUT = "output/pioneer_hitting_2026.parquet"
# totals rows from the pastes (players, PA, AB, H, HR, SB, BB, SO) in block order, for the transcription check
TOTALS = {1: (24, 4119, 3349, 1037, 148, 160, 571, 719), 2: (18, 4238, 3515, 1212, 182, 143, 564, 706), 3: (23, 4314, 3492, 1164, 151, 119, 652, 582), 4: (27, 4193, 3430, 1061, 151, 176, 625, 725), 5: (23, 4092, 3309, 1037, 123, 155, 651, 793),
          6: (19, 4311, 3605, 1293, 173, 178, 532, 671), 7: (29, 4296, 3591, 1304, 184, 98, 543, 686), 8: (29, 4092, 3516, 1089, 193, 83, 484, 757), 9: (29, 3986, 3398, 1008, 121, 159, 462, 720), 10: (21, 4300, 3539, 1191, 158, 200, 587, 716), 11: (29, 3894, 3362, 934, 136, 232, 432, 926),
          12: (27, 4183, 3392, 1104, 91, 179, 558, 706)}
# block -> team (blocks 1-11 identified 2026-10-13 via players also in the team-labelled pitching table, one-to-one; block 12 pasted by the user with its team name)
TEAMS = {1: "Long Beach Coast", 2: "Billings Mustangs", 3: "Missoula PaddleHeads", 4: "Modesto Roadsters", 5: "Glacier Range Riders", 6: "Ogden Raptors", 7: "Idaho Falls Chukars",
         8: "Oakland Ballers", 9: "Yuba-Sutter Freebirds", 10: "Great Falls Voyagers", 11: "RedPocket Mobiles", 12: "Boise Hawks"}
NUM = ["G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "TB", "HBP", "SH", "SF"]


def parse():
    rows, block = [], 0; names = None
    for line in open(RAW, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("#BLOCK"):
            block = int(line.split()[1]); continue
        if line.startswith("Rk,"):
            names = line.split(","); continue
        f = line.split(",")
        if len(f) < len(names) or not f[0].isdigit():
            continue
        d = dict(zip(names, f)); d["block"] = block; rows.append(d)
    df = pd.DataFrame(rows)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["bat"] = np.where(df["Name"].str.contains(r"\*"), "L", np.where(df["Name"].str.contains("#"), "S", "R")); df["name"] = df["Name"].str.replace(r"[\*#?]", "", regex=True).str.strip(); df["br_id"] = df["Name-additional"].str.replace("player.fcgi?id=", "", regex=False)
    df["age"] = pd.to_numeric(df["Age"], errors="coerce"); return df


def check(df):
    print("TRANSCRIPTION CHECK vs the team totals rows (parsed - pasted): players, PA, AB, H, HR, SB, BB, SO")
    bad = 0
    for b, tot in TOTALS.items():
        g = df[df["block"] == b]; got = (len(g), int(g["PA"].sum()), int(g["AB"].sum()), int(g["H"].sum()), int(g["HR"].sum()), int(g["SB"].sum()), int(g["BB"].sum()), int(g["SO"].sum())); diff = tuple(a - c for a, c in zip(got, tot)); bad += any(diff)
        print(f"  block {b:2d}: {'OK' if not any(diff) else 'DIFF ' + str(diff)}  {got}")
    print(f"  blocks with a difference: {bad} of {len(TOTALS)}")


def main():
    df = parse(); check(df); print(f"\n{len(df)} player-team rows, {df['br_id'].nunique()} distinct players; on two teams: {int((df.groupby('br_id').size() > 1).sum())}")
    df["team"] = df["block"].map(TEAMS); agg = df.groupby("br_id").agg(name=("name", "first"), team=("team", lambda t: "/".join(sorted(set(t)))), bat=("bat", "first"), age=("age", "max"), teams=("block", "nunique"), **{c: (c, "sum") for c in NUM}).reset_index()
    agg["ibb"] = 0.0; den = agg["AB"] + agg["BB"] + agg["SF"] + agg["HBP"]; h1 = agg["H"] - agg["2B"] - agg["3B"] - agg["HR"]
    agg["woba"] = (0.69 * agg["BB"] + 0.72 * agg["HBP"] + 0.88 * h1 + 1.25 * agg["2B"] + 1.58 * agg["3B"] + 2.03 * agg["HR"]) / den.replace(0, np.nan); agg["k"] = agg["SO"] / agg["PA"].replace(0, np.nan); agg["bbr"] = agg["BB"] / agg["PA"].replace(0, np.nan); agg["hrr"] = agg["HR"] / agg["PA"].replace(0, np.nan)
    agg["name_clean"] = agg["name"].map(lambda s: " ".join(re.sub(r"[^a-z ]", " ", s.lower()).split()))
    lg = {s: float(np.average(agg.loc[agg[s].notna(), s], weights=agg.loc[agg[s].notna(), "PA"])) for s in ("woba", "k", "bbr", "hrr")}; print("Pioneer 2026 league means (PA-weighted, all 12 teams): " + ", ".join(f"{k} {v:.4f}" for k, v in lg.items()) + f"; hitters with >= 100 PA: {int((agg['PA'] >= 100).sum())}")
    agg.to_parquet(OUT, index=False); print(f"saved {OUT} ({len(agg)} hitters)")


if __name__ == "__main__":
    main()
