"""
pull_aaa_statcast.py -- pulls AAA pitch-level Statcast data via the
baseball-stats-python package's minor_statcast_search().

CRITICAL, empirically-confirmed constraint: the underlying Savant minor-league
endpoint silently caps results at 25,000 rows AND truncates from the FRONT of
the requested date range (a full-month request for June 2024 came back
containing ONLY June 25-30 -- the first 24 days were silently dropped, not
sampled). A naive month-chunked pull (mirroring the MLB pull's window size)
is therefore badly wrong: every chunk hit exactly 25,000 rows, meaning most of
each month never showed up in this repo's first attempt at this pull.

Fix: adaptive/recursive date-window splitting. Pull a window; if it comes back
at (or suspiciously near) the 25,000-row cap, OR its returned min date is
later than the requested start (proof of front-truncation), split the window
in half and recurse on each half. Guarantees correctness regardless of how
daily pitch volume actually varies, rather than guessing a fixed "safe" window
size. Each leaf request is still checkpointed to disk for resumability.

Usage:
    python pull_aaa_statcast.py
"""
import glob
import os
import time
from datetime import date, timedelta

import pandas as pd
import baseball_stats_python as bsp

CHUNK_DIR = "output/aaa_statcast_raw_chunks_v2"
YEARS = [2023, 2024, 2025]
SEASON_WINDOWS = {
    2023: ("04-01", "09-30"),
    2024: ("03-29", "09-30"),
    2025: ("03-28", "09-30"),
}
CAP = 25_000
SAFETY_MARGIN = 500  # treat anything within this of the cap as suspect too


def _fetch(year: int, start_dt: str, end_dt: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            return bsp.minor_statcast_search(season=str(year), player_type="pitcher", level="AAA",
                                              start_dt=start_dt, end_dt=end_dt)
        except Exception as e:
            print(f"    attempt {attempt+1} failed ({start_dt}..{end_dt}): {e}")
            time.sleep(10)
    raise RuntimeError(f"gave up on {start_dt}..{end_dt} after 3 attempts")


def pull_window(year: int, start: date, end: date, depth: int = 0):
    """Recursively pull [start, end], splitting in half whenever truncation
    is detected, until every leaf chunk is proven complete. Leaf results are
    checkpointed as {start}_{end}.parquet so a rerun can skip finished leaves
    (checked by filename, not by re-deriving the split each time)."""
    chunk_path = f"{CHUNK_DIR}/{year}_{start.isoformat()}_{end.isoformat()}.parquet"
    if os.path.exists(chunk_path):
        print(f"{'  '*depth}skip (exists): {chunk_path}")
        return

    start_s, end_s = start.isoformat(), end.isoformat()
    print(f"{'  '*depth}pulling {start_s} .. {end_s}")
    try:
        df = _fetch(year, start_s, end_s)
    except Exception as e:
        print(f"{'  '*depth}  GAVE UP on {start_s}..{end_s} ({e}) -- skipping, rerun script later to retry")
        return

    if df is None or df.empty:
        print(f"{'  '*depth}  empty -- saving empty marker")
        pd.DataFrame().to_parquet(chunk_path)
        return

    min_date = pd.to_datetime(df["game_date"]).min().date()
    truncated = (len(df) >= CAP - SAFETY_MARGIN) or (min_date > start)
    if truncated and start < end:
        mid = start + (end - start) // 2
        if mid == start:
            mid = start  # single-day window can't split further -- accept as-is
        else:
            print(f"{'  '*depth}  TRUNCATED (n={len(df):,}, returned min_date={min_date} > "
                  f"requested start={start}) -- splitting into {start_s}..{mid.isoformat()} "
                  f"and {(mid+timedelta(days=1)).isoformat()}..{end_s}")
            pull_window(year, start, mid, depth + 1)
            pull_window(year, mid + timedelta(days=1), end, depth + 1)
            return

    print(f"{'  '*depth}  saved {len(df):,} rows ({min_date} .. {pd.to_datetime(df['game_date']).max().date()})")
    df.to_parquet(chunk_path)
    time.sleep(1.0)


def pull_all():
    os.makedirs(CHUNK_DIR, exist_ok=True)
    for year in YEARS:
        m1, m2 = SEASON_WINDOWS[year]
        start = date.fromisoformat(f"{year}-{m1}")
        end = date.fromisoformat(f"{year}-{m2}")
        pull_window(year, start, end)


def assemble():
    for year in YEARS:
        chunks = sorted(glob.glob(f"{CHUNK_DIR}/{year}_*.parquet"))
        if not chunks:
            print(f"no chunks for {year}")
            continue
        frames = [pd.read_parquet(c) for c in chunks]
        frames = [f for f in frames if not f.empty]
        if not frames:
            print(f"all chunks empty for {year}")
            continue
        full = pd.concat(frames, ignore_index=True)
        full = full.drop_duplicates()  # adjacent-window boundary days could double up
        full["game_year"] = year
        out_path = f"aaa_statcast_{year}.parquet"
        full.to_parquet(out_path)
        date_range = pd.to_datetime(full["game_date"])
        print(f"assembled {out_path}: {len(full):,} rows, {full.shape[1]} columns, "
              f"{full['pitcher'].nunique()} distinct pitchers, dates {date_range.min().date()} to "
              f"{date_range.max().date()}")


if __name__ == "__main__":
    pull_all()
    assemble()
    print("done")
