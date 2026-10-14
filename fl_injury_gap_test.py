"""
fl_injury_gap_test.py -- tests whether a large within-season GAP between a
pitcher's FL appearances (already derivable from the existing pitch-level
data, for any year including 2026) is a usable PROXY for a real injury, by
checking it against actual ground truth: naprobaseball.com's Frontier League
transactions page, which logs real dated IL placements/activations but is
stale (no 2026 entries as of this pull -- see DECISION_LOG.md) and so can
only validate the proxy for 2025, not be used directly as a live 2026
feature itself.

Raw transactions text was fetched via the browser tool (scrolled to force
full lazy-load) and saved to the scratchpad -- this script parses it, not a
live fetch.

Usage:
    python fl_injury_gap_test.py
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_college_bridge as bridge

RAW_PATH = (r"C:\Users\215nt\AppData\Local\Temp\claude\C--Users-215nt-PycharmProjects-"
            r"FLStuff-FLHitting\10970283-d3af-4ce4-a21e-e91dde5580c0\scratchpad\fl_transactions_2025.txt")

DATE_RE = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December) "
                      r"\d{1,2}, \d{4}$")
IL_RE = re.compile(
    r"(Placed|Activated|Transferred)\s+(?:[A-Z]{1,3}\s+)?([A-Z][a-zA-Z'.\-]+(?:\s+[A-Z][a-zA-Z'.\-]+)+)"
    r"\s+(?:on|from|to)\s+the\s+(7-Day|14-Day|60-Day|[Ss]eason-ending)\s+Injured\s+List",
    re.IGNORECASE,
)


def parse_transactions(path: str = RAW_PATH) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = [l.strip() for l in text.split("\n")]

    rows = []
    current_date = None
    for line in lines:
        if not line:
            continue
        if DATE_RE.match(line):
            current_date = pd.Timestamp(line)
            continue
        if current_date is None:
            continue
        for m in IL_RE.finditer(line):
            action, name, il_type = m.groups()
            rows.append({
                "date": current_date, "action": action.lower(),
                "name": name.strip(), "il_type": il_type.lower(),
            })
    df = pd.DataFrame(rows)
    df["name_clean"] = df["name"].apply(bridge._clean_name)
    return df


def build_appearance_gaps(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    pool = fp.filter_period(raw, start, end)
    appearances = pool.drop_duplicates(["pitcher_id", "date"])[["pitcher_id", "pitcher", "date"]].copy()
    appearances = appearances.sort_values(["pitcher_id", "date"])
    appearances["gap_days"] = appearances.groupby("pitcher_id")["date"].diff().dt.days
    appearances["gap_start"] = appearances.groupby("pitcher_id")["date"].shift(1)
    appearances["name_clean"] = appearances["pitcher"].apply(
        lambda s: bridge._clean_name(s.split(",")[1] + " " + s.split(",")[0]) if "," in str(s) else bridge._clean_name(s))
    return appearances


if __name__ == "__main__":
    print("Parsing Frontier League transactions (2025 ground truth)...")
    txn = parse_transactions()
    print(f"  {len(txn)} IL-related events parsed, {txn['name_clean'].nunique()} distinct players")
    print(txn["action"].value_counts())
    print(txn["il_type"].value_counts())

    print("\nLoading FL pitches, building 2025 within-season appearance gaps...")
    raw = fp.load_fl_pitches_cached()
    gaps = build_appearance_gaps(raw, bt.START_2025, bt.END_2025)

    placed = txn[txn["action"].isin(["placed", "transferred"])][["name_clean", "date", "il_type"]].rename(
        columns={"date": "il_date"})

    # for each large gap, check if there's a REAL recorded IL placement for
    # that same player within a reasonable window of the gap's start
    for thresh in [10, 14, 21]:
        big_gaps = gaps[gaps["gap_days"] >= thresh].copy()
        matched = 0
        for _, g in big_gaps.iterrows():
            cand = placed[placed["name_clean"] == g["name_clean"]]
            if cand.empty:
                continue
            # a real IL placement within [gap_start - 3d, gap_start + gap_days] counts as confirming this gap
            window_end = g["gap_start"] + pd.Timedelta(days=int(g["gap_days"]))
            hit = cand[(cand["il_date"] >= g["gap_start"] - pd.Timedelta(days=3)) & (cand["il_date"] <= window_end)]
            if not hit.empty:
                matched += 1
        print(f"\ngap >= {thresh}d: {len(big_gaps)} gaps, {matched} ({matched/max(len(big_gaps),1):.1%}) "
              f"confirmed by a real recorded IL placement in the transactions log")
