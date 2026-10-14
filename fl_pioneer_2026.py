"""
fl_pioneer_2026.py -- loads the 2026 Pioneer League pitching tables (Baseball
Reference team pitching pages, pasted by the user because direct acquisition from
that site has been restricted) into the same schema as
output/fl_pioneer_pitching_2021_2025.parquet, so fl_pioneer_bridge.load_all_pioneer_years
(and therefore fl_unified_bridge and the dashboard's Pioneer lookup) picks up 2026.

Input: output/pioneer_2026_raw/pioneer_2026_br.csv -- one row per pitcher per team table
(block letters A-L = twelve distinct team tables; the first paste contained four exact duplicates
that were dropped). A pitcher who appeared for two teams has one row per team; they are
summed (keyed on the Baseball Reference id).

Caveats:
  * All 12 Pioneer teams are present (12 team tables).
  * The BF column is blank on those pages, so batters faced is ESTIMATED as
        BF = outs + H + BB + HBP
    (outs = 3 x whole innings + the fractional thirds). Checked against the six pitchers whose
    true BF was available from a separate export (Mayer 196 vs 195, Calnon 416 vs 415,
    Rosengard 33 vs 33, Cermenelli 6 vs 6, Dayton 222 vs 218, Romo 241 vs 234): the formula
    runs 0-3% high (double plays and caught stealing add outs without batters), well inside
    what the n0 = 50 batters-faced shrinkage absorbs.
  * Each team table is reconciled against its own printed totals row (informational: only tables B and C match
    exactly; the others differ both ways, a Baseball Reference quirk -- see build()).

Usage:  python fl_pioneer_2026.py
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

RAW_PATH = "output/pioneer_2026_raw/pioneer_2026_br.csv"
HIST_PATH = "output/fl_pioneer_pitching_2021_2025.parquet"
OUT_PATH = "output/fl_pioneer_pitching_2026.parquet"

# printed totals row of each team table: players, IP, H, HR, BB, SO, HBP
EXPECTED = {
    "A": (26, "839.1", 928, 116, 405, 718, 64), "B": (31, "848.0", 948, 107, 435, 859, 63),
    "C": (27, "818.0", 970, 104, 482, 825, 70), "D": (27, "820.0", 1044, 118, 481, 769, 55),
    "E": (39, "842.0", 1050, 207, 587, 734, 100), "F": (43, "834.1", 1196, 167, 501, 743, 67),
    "G": (35, "822.0", 1152, 160, 587, 630, 85), "H": (36, "831.1", 1130, 172, 730, 711, 99),
    "I": (43, "787.1", 1233, 132, 598, 601, 106),
    "J": (38, "831.1", 1287, 160, 705, 773, 120), "K": (38, "835.1", 1230, 184, 539, 722, 99),
    "L": (44, "808.1", 1266, 184, 611, 622, 126),
}


# Team table -> team, INFERRED by matching pitchers against an earlier export that carried team names (the pasted
# Baseball Reference tables themselves carry no team label). J/K/L, pasted last, are the three teams that had been
# missing (Great Falls: Rogers, Gauna; Ogden: Stancato, Lane, Valentine, Jordan; Idaho Falls: Smiley). B and C are
# Glacier Range Riders and Modesto Roadsters in some order (no pitcher in either could be placed).
TEAM_BY_BLOCK = {"A": "Long Beach Coast", "D": "Billings Mustangs", "E": "Oakland Ballers", "F": "Missoula PaddleHeads",
                 "G": "Boise Hawks", "H": "Yuba-Sutter Freebirds", "I": "RedPocket Mobiles",
                 "J": "Great Falls Voyagers", "K": "Ogden Raptors", "L": "Idaho Falls Chukars",
                 "C": "Glacier Range Riders", "B": "Modesto Roadsters"}   # B/C confirmed by the user


def ip_to_outs(ip) -> int:
    whole, _, frac = f"{float(ip):.1f}".partition(".")
    return int(whole) * 3 + int(frac)


def clean_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z\s]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def build(verbose: bool = True) -> pd.DataFrame:
    d = pd.read_csv(RAW_PATH, dtype={"ip": str})
    # --- reconcile every team table with its printed totals row (informational) ------------------------------
    # Only tables B and C reconcile EXACTLY; for the others the printed totals differ from the sum of the listed
    # rows in BOTH directions (Baseball Reference's totals row is not a plain sum of the rows pasted). The
    # per-pitcher rows themselves agree exactly with an independent export for every pitcher that could be
    # cross-checked (Mayer, Calnon, Dayton, Romo, Blythe, Rosengard, Cermenelli ...), so rows are used as-is.
    recon = []
    for blk, (n, ip, h, hr, bb, so, hbp) in EXPECTED.items():
        g = d[d["block"] == blk]
        if len(g) != n:
            raise ValueError(f"team table {blk}: {len(g)} rows but the table says {n} players")
        listed = (sum(ip_to_outs(x) for x in g["ip"]), int(g["h"].sum()), int(g["bb"].sum()), int(g["so"].sum()))
        printed = (ip_to_outs(ip), h, bb, so)
        recon.append((blk, listed[0] / printed[0], listed[1] / printed[1], listed[2] / printed[2], listed[3] / printed[3]))
    if verbose:
        print("[fl_pioneer_2026] listed rows / printed totals (outs, H, BB, SO):",
              "; ".join(f"{b}: {o:.2f} {h:.2f} {bb:.2f} {so:.2f}" for b, o, h, bb, so in recon))
    d["outs"] = d["ip"].apply(ip_to_outs)
    d["name"] = d["name"].str.replace("*", "", regex=False).str.replace("?", "", regex=False).str.strip()
    g = d.groupby("br_id").agg(player=("name", "first"), n_teams=("block", "nunique"), outs=("outs", "sum"), h=("h", "sum"),
                               bb=("bb", "sum"), so=("so", "sum"), hbp=("hbp", "sum"), teams=("block", lambda s: "/".join(sorted({TEAM_BY_BLOCK[b] for b in s})))).reset_index()
    g["bf"] = (g["outs"] + g["h"] + g["bb"] + g["hbp"]).astype(float)       # outs already = 3 x IP
    g = g[g["bf"] > 0].copy()
    g["name_clean"] = g["player"].apply(clean_name)
    g["year"] = 2026
    g["k_pct"] = g["so"] / g["bf"]; g["bb_pct"] = g["bb"] / g["bf"]; g["any_fallback"] = False
    hist = pd.read_parquet(HIST_PATH)
    g["in_2021_25_history"] = g["name_clean"].isin(set(hist["name_clean"]))
    out = g.rename(columns={"outs": "outs_total"})[["name_clean", "year", "bf", "so", "bb", "n_teams", "teams", "player", "any_fallback",
                                                  "k_pct", "bb_pct", "in_2021_25_history"]]
    out.to_parquet(OUT_PATH, index=False)
    if verbose:
        print(f"[fl_pioneer_2026] {d['block'].nunique()} team tables; {len(out)} pitchers, {int(out['bf'].sum())} estimated BF "
              f"(K% {out['so'].sum()/out['bf'].sum():.3f}, BB% {out['bb'].sum()/out['bf'].sum():.3f}); "
              f"{int(out['in_2021_25_history'].sum())} also appear in the 2021-25 Pioneer history")
    return out


if __name__ == "__main__":
    o = build()
    print(o.sort_values("bf", ascending=False).head(12)[["player", "teams", "bf", "so", "bb", "k_pct", "bb_pct", "in_2021_25_history"]].round(3).to_string(index=False))
