"""
Supplemental targeted pull: batter/count-adjusted strike-zone edges (sz_top, sz_bot, sz_left,
sz_right), used to construct a Heart% definition (middle-of-the-zone rate) consistent with the
outside_zone_adjusted convention. Merges into 2025_2026_pitches.parquet by pitch_uid.

Usage:
    python pull_sz_bounds_supplement.py
"""

import pandas as pd
from sqlalchemy import text

from fl_supabase import read_sql_df

PARQUET_PATH = "2025_2026_pitches.parquet"

SUPPLEMENT_COLUMNS = ["sz_top", "sz_bot", "sz_left", "sz_right"]


def pull_supplement(years=(2025, 2026)) -> pd.DataFrame:
    col_sql = ",\n           ".join(SUPPLEMENT_COLUMNS)
    q = text(f"""
        SELECT pitch_uid,
           {col_sql}
        FROM pitches
        WHERE year = ANY(:years)
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
        base = base.drop(columns=[c for c in new_cols if c in base.columns], errors="ignore")
        merged = base.merge(supp[["pitch_uid"] + new_cols], on="pitch_uid", how="left")
        assert len(merged) == len(base), "merge changed row count"
        merged.to_parquet(PARQUET_PATH, index=False)
        print(f"merged and saved: {merged.shape}")
