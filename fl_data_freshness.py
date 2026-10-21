"""
fl_data_freshness.py -- pre-flight check for run_all.py: are the LOCAL data files the pipeline reads in step with the database?  (2026-10-20: 2025_2026_pitches.parquet had been left at
2026-08-16 while the season ran to 09-06, about 200 regular-season games missing, and nothing noticed.)
Checks (each compares the local file with the same query the loader would run):
  1. 2025_2026_pitches.parquet       games / pitches per (year, game_type) and latest date vs `pitches`                  (refresh: python pull_2025_2026.py)
  2. output/fl_raw_pitches_flagged   regular-season pitcher-id rows per year vs `pitches`; rows in canonical order       (refresh: delete the file; the loader rebuilds it)
  3. output/hitter_bbe_2025_2026     regular-season balls in play per year vs `pitches`                                  (refresh: fl_hitter_ev.load_bbe(refresh=True))
  4. output/hitter_pitches_2025_2026 regular-season pitches per year vs `pitches`                                        (refresh: fl_hitter_whiff.load(refresh=True))
Exit status 1 if any check fails.  Usage: python fl_data_freshness.py   (run_all.py runs it first; --skip-check bypasses it).
"""
import sys
import warnings

import pandas as pd
from sqlalchemy import text

from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
BIP = "('single','double','triple','home_run','out','fielders_choice','error')"


def db(q):
    return read_sql_df(text(q))


def main():
    fails = []

    def check(name, ok, detail):
        print(f"[{'ok' if ok else 'STALE'}] {name}: {detail}")
        if not ok:
            fails.append(name)

    # 1. tracking parquet
    import pyarrow.parquet as pq
    has_gt = "game_type" in pq.ParquetFile("2025_2026_pitches.parquet").schema.names
    t = pd.read_parquet("2025_2026_pitches.parquet", columns=["year", "game_id", "date"] + (["game_type"] if has_gt else []))
    if not has_gt:
        check("tracking parquet has game_type", False, "column missing -- re-run pull_2025_2026.py")
    else:
        d = db("select year, game_type, count(distinct game_id) games, count(*) pitches, max(date) hi from pitches where year in (2025, 2026) group by 1, 2")
        L = t.groupby(["year", "game_type"]).agg(games=("game_id", "nunique"), pitches=("game_id", "size"), hi=("date", "max")).reset_index()
        m = d.merge(L, on=["year", "game_type"], how="left", suffixes=("_db", "_local")).fillna({"games_local": 0, "pitches_local": 0})
        bad = m[(m["games_db"] != m["games_local"]) | (m["pitches_db"] != m["pitches_local"])]
        check("tracking parquet vs database", bad.empty, f"{len(m)} year x game-type groups" + ("" if bad.empty else "; differing: " + bad[["year", "game_type", "games_db", "games_local", "pitches_db", "pitches_local"]].to_string(index=False)))

    # 2. pitcher cache
    import fl_pitcher_features as fp
    c = pd.read_parquet(fp.CACHE_PATH)
    d = db("select year, count(*) n from pitches where game_type = 'reg_season' and pitcher_id is not null and year in (2025, 2026) group by 1").set_index("year")["n"]
    l = c[c["year"].isin([2025, 2026])].groupby("year").size()
    check("pitcher cache vs database", all(int(d.get(y, 0)) == int(l.get(y, 0)) for y in (2025, 2026)), f"db {d.astype(int).to_dict()} local {l.astype(int).to_dict()}")
    check("pitcher cache in canonical order with one name per id", fp.canonical_order(c)[["game_id", "pa_id", "pitch_no", "pitcher"]].astype(str).equals(c[["game_id", "pa_id", "pitch_no", "pitcher"]].astype(str)), "sorted on fl_pitcher_features.ORDER_KEY")

    # 3 / 4. hitter caches
    b = pd.read_parquet("output/hitter_bbe_2025_2026.parquet", columns=["year"]).groupby("year").size()
    d = db(f"select year, count(*) n from pitches where game_type='reg_season' and year in (2025,2026) and play_result in {BIP} and exit_speed is not null and angle is not null group by 1").set_index("year")["n"]
    check("hitter BBE cache vs database", all(int(d.get(y, 0)) == int(b.get(y, 0)) for y in (2025, 2026)), f"db {d.astype(int).to_dict()} local {b.astype(int).to_dict()}")
    p = pd.read_parquet("output/hitter_pitches_2025_2026.parquet", columns=["year"]).groupby("year").size()
    d = db("select year, count(*) n from pitches where game_type='reg_season' and year in (2025,2026) and pitch_call is not null group by 1").set_index("year")["n"]
    check("hitter pitch cache vs database", all(int(d.get(y, 0)) == int(p.get(y, 0)) for y in (2025, 2026)), f"db {d.astype(int).to_dict()} local {p.astype(int).to_dict()}")

    print("\nFRESH" if not fails else f"\nSTALE INPUTS: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
