"""mlb_opencommand_locvar_test.py -- how much does OpenCommand (real,
computer-vision-derived distance-from-catcher's-target, github.com/tomdoyo/
open-command) relate to this project's own horiz_var/vert_var (location
DISPERSION around a pitcher's own mean, NOT distance to an actual target --
see fl_pitcher_features.compute_loc_var's docstring)? These are conceptually
adjacent (both "command") but structurally different: a pitcher who
consistently misses in the same direction has LOW horiz_var/vert_var
(very repeatable) but could have a large OpenCommand miss (aiming
incorrectly, precisely). If they correlate strongly, that's real validation
that the project's cheap, label-free proxy captures real command skill; if
weakly, that's a real limitation worth knowing.

Uses this project's existing MLB Statcast parquet files (2024+2025 overlap
with OpenCommand's own coverage) to build horiz_var/vert_var exactly as
fl_pitcher_features.compute_loc_var does it (same MIN_TYPE_N_FOR_LOC_VAR
floor, same within-pitch-type-then-pooled-across-types construction), and
OpenCommand's own published command_scores.csv (pulled directly from
Hugging Face, pitch_type=="ALL" rows), matched by player name (OpenCommand:
"First Last"; Statcast: "Last, First", converted the same way this project
converts names everywhere else).

Usage:
    python mlb_opencommand_locvar_test.py
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MIN_TYPE_N = 15  # same floor as fl_pitcher_features.MIN_TYPE_N_FOR_LOC_VAR
YEARS = [2024, 2025]


def clean_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def statcast_name_clean(s: str) -> str:
    s = str(s)
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return clean_name(s)


def compute_loc_var_mlb(df: pd.DataFrame, min_pitches: int = 1) -> pd.DataFrame:
    d = df.dropna(subset=["plate_x_as", "plate_z", "pitch_type", "pitcher"]).copy()
    grp = d.groupby(["pitcher", "pitch_type"]).agg(
        n=("plate_x_as", "size"), hvar=("plate_x_as", "var"), vvar=("plate_z", "var"),
    ).reset_index()
    grp = grp[grp["n"] >= MIN_TYPE_N]

    def _pool(g):
        w = g["n"]
        return pd.Series({"horiz_var": np.average(g["hvar"], weights=w),
                           "vert_var": np.average(g["vvar"], weights=w), "n_pitches": w.sum()})

    out = grp.groupby("pitcher").apply(_pool, include_groups=False).reset_index()
    return out[out["n_pitches"] >= min_pitches]


if __name__ == "__main__":
    print(f"Loading Statcast {YEARS}...")
    frames = []
    for y in YEARS:
        d = pd.read_parquet(f"statcast_{y}.parquet", columns=["pitcher", "pitch_type", "plate_x", "plate_z", "p_throws"])
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    arm_sign = np.where(raw["p_throws"] == "R", 1, -1)
    raw["plate_x_as"] = raw["plate_x"] * arm_sign
    print(f"  {len(raw):,} pitches")

    print("Computing horiz_var/vert_var (pooled 2024+2025, same construction as fl_pitcher_features.compute_loc_var)...")
    lv = compute_loc_var_mlb(raw)
    print(f"  n={len(lv)} pitchers")

    names = pd.read_parquet("output/pitcher_id_name_map.parquet")
    names["name_clean"] = names["name"].apply(statcast_name_clean)
    lv = lv.merge(names, left_on="pitcher", right_on="pitcher_id", how="left")

    print("\nLoading OpenCommand command_scores (2024+2025, pitch_type=ALL)...")
    oc_frames = []
    for y in YEARS:
        d = pd.read_csv(f"oc_{y}_command_scores.csv")
        d = d[d["pitch_type"] == "ALL"].copy()
        d["year"] = y
        oc_frames.append(d)
    oc = pd.concat(oc_frames, ignore_index=True)
    oc["name_clean"] = oc["pitcher"].apply(clean_name)
    # n-weighted pool across the two years per pitcher
    oc_pooled = oc.groupby("name_clean").apply(lambda g: pd.Series({
        "naive_in": np.average(g["naive_in"], weights=g["n"]),
        "inferred_in": np.average(g["inferred_in"], weights=g["n"]),
        "n_oc": g["n"].sum(),
    }), include_groups=False).reset_index()
    print(f"  n={len(oc_pooled)} pitchers (pooled across available years)")

    m = lv.merge(oc_pooled, on="name_clean", how="inner")
    print(f"\nMatched: n={len(m)} pitchers with both real horiz_var/vert_var and real OpenCommand scores")

    for ocf in ["naive_in", "inferred_in"]:
        for lvf in ["horiz_var", "vert_var"]:
            print(f"  corr({ocf}, {lvf}) = {m[ocf].corr(m[lvf]):.4f}")
        combined = np.sqrt(m["horiz_var"] + m["vert_var"])
        print(f"  corr({ocf}, sqrt(horiz_var+vert_var)) = {m[ocf].corr(combined):.4f}")
        print()

    m.to_parquet("output/mlb_opencommand_locvar_matched.parquet", index=False)
    print("Saved -> output/mlb_opencommand_locvar_matched.parquet")
