"""
run_naa_reliability.py
======================
Pulls BBE data from Supabase and runs the full Natural Attack Angle
reliability analysis.  No release_view dependency.

Usage
-----
    python run_naa_reliability.py
    # or %run run_naa_reliability.py in a notebook
"""

import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# ── Path to the reliability analysis module ───────────────────────────────────
sys.path.append(r'C:\Users\215nt\PycharmProjects\FLStuff')
from natural_attack_angle_reliability import run_full_analysis

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ENV_PATH = r'C:\Users\215nt\PycharmProjects\FLStuff\FLPitching\FLCommand\.env'

YEARS = [2025]

MIN_BBE_RELIABLE = 20
N_SPLITS         = 80   # raise to 200 for a tighter reliability curve

OUTPUT_CSV    = r'C:\Users\215nt\PycharmProjects\FLStuff\naa_output.csv'
OUTPUT_FIGURE = r'C:\Users\215nt\PycharmProjects\FLStuff\naa_reliability.png'

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_engine(env_path: str):
    load_dotenv(Path(env_path))
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.environ["SUPABASE_USER"],
        password=os.environ["SUPABASE_PASSWORD"],
        host=os.environ["SUPABASE_HOST"],
        port=int(os.environ.get("SUPABASE_PORT", 5432)),
        database=os.environ.get("SUPABASE_DATABASE", "postgres"),
    )
    return create_engine(url, pool_pre_ping=True)

# ─────────────────────────────────────────────────────────────────────────────
# PULL
# ─────────────────────────────────────────────────────────────────────────────

def pull_bbe(engine, years=YEARS) -> pd.DataFrame:
    year_list = ", ".join(str(y) for y in years)
    sql = f"""
        SELECT
            batter_id,
            batter,
            year,
            exit_speed,
            angle
        FROM pitches
        WHERE exit_speed IS NOT NULL
          AND angle       IS NOT NULL
          AND year IN ({year_list})
    """
    df = pd.read_sql(sql, engine)
    print(f"Pulled {len(df):,} BBE rows — "
          f"{df['batter_id'].nunique():,} batters, "
          f"years {sorted(df['year'].unique())}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Connecting ──────────────────────────────────────────────────")
    engine = get_engine(ENV_PATH)

    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM pitches")).scalar()
        print(f"   Connected. {n:,} total rows in pitches.")

    print("\n── Pulling BBE ─────────────────────────────────────────────────")
    raw_df = pull_bbe(engine)

    print("\n── Running reliability analysis ────────────────────────────────")
    results = run_full_analysis(
        raw_df,
        min_bbe_reliable=MIN_BBE_RELIABLE,
        n_splits_curve=N_SPLITS,
        output_path=OUTPUT_FIGURE,
    )

    results["naa_df"].to_csv(OUTPUT_CSV, index=False)
    print(f"\nNAA output saved → {OUTPUT_CSV}")

    print("\n── Reliability curve ───────────────────────────────────────────")
    print(results["rel_curve"].dropna(subset=["r_full"]).to_string(index=False))

    print("\n── Method audit ────────────────────────────────────────────────")
    print(results["audit"].to_string(index=False))