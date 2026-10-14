"""
Supplemental targeted pull: pitch movement/velocity metrics (ivb, horz_break,
spin_rate, extension, effective_velo) weren't in the original wide pull.
Unlike catch_prob (batted-ball only), these apply to every pitch, so this
pulls the full row population but keeps the column list narrow to stay fast.
Merges into 2025_2026_pitches.parquet by pitch_uid (verified unique: 479,801
distinct values across 479,801 rows).

Usage:
    python pull_movement_supplement.py
"""

import pandas as pd
from sqlalchemy import text

from fl_supabase import read_sql_df

PARQUET_PATH = "2025_2026_pitches.parquet"

SUPPLEMENT_COLUMNS = ["ivb", "horz_break", "spin_rate", "extension", "effective_velo"]


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
        assert len(merged) == len(base), "merge changed row count -- pitch_uid not unique?"
        merged.to_parquet(PARQUET_PATH, index=False)
        print(f"merged and saved: {merged.shape}")
