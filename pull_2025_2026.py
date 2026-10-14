"""
Pull all 2025 + 2026 (so far) pitch-level data from Supabase, narrowed to the
columns actually used in hitting analysis (vs. all 293 columns on `pitches`).

Usage:
    python pull_2025_2026.py

Writes: 2025_2026_pitches.parquet (same directory)
"""

import pandas as pd
from sqlalchemy import text

from fl_supabase import read_sql_df

COLUMNS = [
    # --- core / context ---
    "game_id", "pitch_uid", "date", "year", "inning",
    "home_team", "away_team",
    "batter", "batter_id", "batter_team", "batter_side",
    "pitcher", "pitcher_id", "pitcher_team", "pitcher_throws",
    "balls", "strikes", "outs",
    "pitch_call", "play_result", "batted_ball_type",
    "pa_id", "pa_of_inning", "pa_pitch_count", "is_terminal_pitch", "k_or_bb",

    # --- batted-ball physics ---
    "exit_speed", "angle", "direction", "distance", "hang_time",
    "bearing", "max_height", "hit_spin_rate",
    "contact_pos_x", "contact_pos_y", "contact_pos_z",

    # --- bat speed & swing decisions ---
    "batspeed", "swing", "whiff", "chase", "csoe", "outside_zone",

    # --- outcome / value metrics ---
    "xdamage", "xwoba", "xslg", "xba", "barrel", "woba",
    "rv_batter", "war_rv", "re_before", "re_after",

    # --- pitch characteristics faced ---
    "rel_speed", "pitch_type", "zone_speed", "plate_x", "plate_z",
    "vert_appr_angle", "horz_appr_angle",

    # --- fielder positioning + stadium ---
    "stadium", "oaa", "rv_fielder", "fielded_by", "fielded_by_pos", "temp", "wind",
    "num_parks_hr", "num_parks_hr_denom",
    "pos_1b_x", "pos_1b_z", "pos_2b_x", "pos_2b_z",
    "pos_3b_x", "pos_3b_z", "pos_ss_x", "pos_ss_z",
    "pos_lf_x", "pos_lf_z", "pos_cf_x", "pos_cf_z",
    "pos_rf_x", "pos_rf_z",
    "name_1b", "name_2b", "name_3b", "name_ss",
    "name_lf", "name_cf", "name_rf",
]

OUT_PATH = "2025_2026_pitches.parquet"


def pull(years=(2025, 2026)) -> pd.DataFrame:
    col_sql = ",\n           ".join(COLUMNS)
    q = text(f"""
        SELECT {col_sql}
        FROM pitches
        WHERE year = ANY(:years)
    """)
    return read_sql_df(q, params={"years": list(years)})


if __name__ == "__main__":
    df = pull()
    print(f"Pulled {len(df):,} rows x {len(df.columns)} columns for years {sorted(df['year'].unique())}")
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")
