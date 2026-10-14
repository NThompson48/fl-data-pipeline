"""
Supplemental targeted pull: adds columns not worth carrying in the full
wide pull (which is already near the pooler's practical width/timeout limit).
Pulls only the terminal, in-play row population (~12k rows) and merges into
the existing 2025_2026_pitches.parquet by pitch_uid (verified unique: 479,801
distinct values across 479,801 rows).

Usage:
    python pull_catch_prob_supplement.py
"""

import pandas as pd
from sqlalchemy import text

from fl_supabase import read_sql_df

PARQUET_PATH = "2025_2026_pitches.parquet"

SUPPLEMENT_COLUMNS = [
    "catch_prob", "catch_prob_exact",
    "outlier", "outlier_bad_pitch_loc", "potential_outlier_game_pitch_loc",
    "outlier_cs_model", "outlier_missing_tracking",
]


def pull_supplement(years=(2025, 2026)) -> pd.DataFrame:
    col_sql = ",\n           ".join(SUPPLEMENT_COLUMNS)
    q = text(f"""
        SELECT pitch_uid,
           {col_sql}
        FROM pitches
        WHERE year = ANY(:years)
          AND is_terminal_pitch = '1'
          AND pitch_call = 'in_play'
    """)
    return read_sql_df(q, params={"years": list(years)})


if __name__ == "__main__":
    base = pd.read_parquet(PARQUET_PATH)
    new_cols = [c for c in SUPPLEMENT_COLUMNS if c not in base.columns]
    if not new_cols:
        print("all supplement columns already present, nothing to do")
    else:
        supp = pull_supplement()
        print(f"pulled {len(supp):,} rows x {len(new_cols)} new columns")
        # drop any of the new columns from base first in case of a partial re-run
        base = base.drop(columns=[c for c in new_cols if c in base.columns], errors="ignore")
        merged = base.merge(supp[["pitch_uid"] + new_cols], on="pitch_uid", how="left")
        assert len(merged) == len(base), "merge changed row count -- pitch_uid not unique?"
        merged.to_parquet(PARQUET_PATH, index=False)
        print(f"merged and saved: {merged.shape}")
