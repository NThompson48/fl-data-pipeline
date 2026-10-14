"""
Pulls MLB Statcast pitch-level data for 2018-2021 via pybaseball (the existing
2022-2025 files in this repo were pulled the same underlying way -- Baseball
Savant does not offer this pre-2022 data through the MLB Stats API, only through
Savant's own search/export, which pybaseball wraps).

Chunked by calendar month (regular season only, roughly March-October) with each
chunk checkpointed to disk immediately so a rerun skips whatever already
completed -- a full 4-year pull realistically takes hours and WILL be
interrupted, so this must be resumable. (Confirmed: the first attempt was
killed externally -- no Python traceback, just cut off mid-chunk after ~16
chunks/~18 minutes -- so resumability is not optional here.)

Only the columns actually needed by the feature-engineering plan are kept
(pybaseball still downloads the full Savant CSV per chunk server-side, so this
doesn't shrink network time, but it substantially cuts local memory/disk/concat
cost, which matters for a run this long).

Usage:
    python pull_statcast_2018_2021.py
"""
import os
import time

import pandas as pd
import pybaseball as pb

pb.cache.enable()

# Columns needed downstream: PA/flag construction (events, description, type),
# pitch-type family + shape features for wh_sva/swstr_sva, location for
# horiz_dist/vert_dist, join/ordering keys, and age_pit as an age cross-check.
# Excludes win-probability, batted-ball-outcome, deprecated, and fielder-position
# columns that mlb_pitcher_features.py has no use for.
KEEP_COLS = [
    "pitcher", "player_name", "batter", "game_pk", "game_date", "game_year",
    "game_type", "at_bat_number", "pitch_number", "inning", "outs_when_up",
    "balls", "strikes", "pitch_type", "p_throws", "stand",
    "events", "description", "type", "zone",
    "plate_x", "plate_z", "sz_top", "sz_bot",
    "pfx_x", "pfx_z", "api_break_x_arm", "api_break_x_batter_in", "api_break_z_with_gravity",
    "release_speed", "effective_speed", "release_spin_rate", "release_extension",
    "release_pos_x", "release_pos_y", "release_pos_z", "arm_angle",
    "age_pit",
]

# Downcast dtypes on top of the column trim -- further cuts memory/disk without
# losing anything the feature plan needs (float32 has ~7 sig figs, plenty for
# plate coordinates/movement/spin; small-cardinality strings as category).
FLOAT32_COLS = [
    "plate_x", "plate_z", "sz_top", "sz_bot", "pfx_x", "pfx_z",
    "api_break_x_arm", "api_break_x_batter_in", "api_break_z_with_gravity",
    "release_speed", "effective_speed", "release_spin_rate", "release_extension",
    "release_pos_x", "release_pos_y", "release_pos_z", "arm_angle",
]
CATEGORY_COLS = ["pitch_type", "p_throws", "stand", "events", "description", "type", "game_type"]


def _trim(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()
    for c in FLOAT32_COLS:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    for c in CATEGORY_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    for c in ("pitcher", "batter", "game_pk", "at_bat_number", "pitch_number",
              "inning", "outs_when_up", "balls", "strikes", "zone", "age_pit"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


CHUNK_DIR = "output/statcast_raw_chunks"
YEARS = [2018, 2019, 2020, 2021]
# Regular season month windows. 2020 was a shortened (60-game) season -- pull
# July-September only for that year rather than guessing at exact boundaries;
# Marcel's recency weighting will down-weight it appropriately once it's outside
# the trailing-3-year window anyway, so no special handling beyond this is needed.
MONTH_WINDOWS = {
    2018: [("03-29", "04-30"), ("05-01", "05-31"), ("06-01", "06-30"), ("07-01", "07-31"),
           ("08-01", "08-31"), ("09-01", "09-30"), ("10-01", "10-31")],
    2019: [("03-28", "04-30"), ("05-01", "05-31"), ("06-01", "06-30"), ("07-01", "07-31"),
           ("08-01", "08-31"), ("09-01", "09-30"), ("10-01", "10-31")],
    2020: [("07-23", "08-31"), ("09-01", "09-30"), ("10-01", "10-28")],
    2021: [("04-01", "04-30"), ("05-01", "05-31"), ("06-01", "06-30"), ("07-01", "07-31"),
           ("08-01", "08-31"), ("09-01", "09-30"), ("10-01", "10-31")],
}


def pull_all():
    os.makedirs(CHUNK_DIR, exist_ok=True)
    for year in YEARS:
        for i, (m1, m2) in enumerate(MONTH_WINDOWS[year]):
            chunk_path = f"{CHUNK_DIR}/{year}_{i:02d}.parquet"
            if os.path.exists(chunk_path):
                print(f"skip (exists): {chunk_path}")
                continue
            start_dt, end_dt = f"{year}-{m1}", f"{year}-{m2}"
            print(f"pulling {start_dt} .. {end_dt} -> {chunk_path}")
            for attempt in range(3):
                try:
                    df = pb.statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
                    break
                except Exception as e:
                    print(f"  attempt {attempt+1} failed: {e}")
                    time.sleep(5)
            else:
                print(f"  GAVE UP on {chunk_path} after 3 attempts -- rerun script later")
                continue
            if df is None or df.empty:
                print(f"  empty result for {start_dt}..{end_dt} -- writing empty marker")
                df = pd.DataFrame()
            else:
                df = _trim(df)
            df.to_parquet(chunk_path)
            print(f"  saved {len(df):,} rows, {df.shape[1]} cols")


def assemble_year(year):
    import glob
    chunks = sorted(glob.glob(f"{CHUNK_DIR}/{year}_*.parquet"))
    if not chunks:
        print(f"no chunks found for {year}")
        return
    frames = [pd.read_parquet(c) for c in chunks]
    frames = [f for f in frames if not f.empty]
    if not frames:
        print(f"all chunks empty for {year} -- skipping assembly")
        return
    full = pd.concat(frames, ignore_index=True)
    full["game_year"] = year
    out_path = f"statcast_{year}.parquet"
    full.to_parquet(out_path)
    print(f"assembled {out_path}: {len(full):,} rows, {full.shape[1]} columns")

    # column-parity check against KEEP_COLS (not the full 119-col statcast_2022.parquet
    # schema -- this pull deliberately keeps only the subset the feature plan needs,
    # so "missing vs statcast_2022" would just relist every column we intentionally
    # dropped; what actually matters is that we got everything we asked for).
    missing = set(KEEP_COLS) - set(full.columns)
    if missing:
        print(f"  WARNING: missing expected columns: {sorted(missing)}")


if __name__ == "__main__":
    pull_all()
    for year in YEARS:
        assemble_year(year)
    print("done")
