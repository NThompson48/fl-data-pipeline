"""
naa_vaa_matchup.py
==================
For each pitcher on TEAM_B (2026), identifies the 3 hitters on TEAM_A
whose Natural Attack Angle creates the biggest plane mismatch against
that pitcher's average fastball VAA.

Data strategy
-------------
  Hitters  : 2026 roster to identify who's on the team,
              2025 BBE data to compute NAA (larger, more stable sample)
  Pitchers : 2026 fastball VAA only (FF / SI / FT — true fastballs,
              not cutters/splitters whose VAA profile is shape-driven)

Matchup metric
--------------
  Plane Delta = NAA + VAA_fb
  VAA is negative (downward); NAA is positive (upward swing plane).
  Δ ≈ 0  →  in-plane, hitter squares it up  (hitter advantage)
  Δ ≠ 0  →  swing plane and pitch plane mismatched  (pitcher advantage)
  |Δ| is the mismatch score — higher = bigger pitcher edge.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

sys.path.append(r'C:\Users\215nt\PycharmProjects\FLStuff')
from natural_attack_angle_reliability import compute_natural_attack_angle

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ENV_PATH = r'C:\Users\215nt\PycharmProjects\FLStuff\FLPitching\FLCommand\.env'

TEAM_A = "Windy City ThunderBolts"   # hitters
TEAM_B = "Mississippi Mud Monsters"          # pitchers

# Fastball pitch types to include in VAA average
FASTBALL_TYPES = ("four_seam", "sinker", "fastball")

MIN_BBE_HITTER   = 70    # min 2025 BBE for a hitter's NAA to qualify
MIN_FB_PITCHES   = 15    # min 2026 fastballs for a pitcher's VAA to qualify

TOP_N_MATCHUPS   = 3     # hitters shown per pitcher

OUTPUT_FIG = r'C:\Users\215nt\PycharmProjects\FLStuff\matchup_output.png'
OUTPUT_CSV = r'C:\Users\215nt\PycharmProjects\FLStuff\matchup_output.csv'

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
# DATA PULLS
# ─────────────────────────────────────────────────────────────────────────────

def pull_team_roster(engine, team: str, year: int, side: str) -> list:
    """
    Return a list of batter_ids (side='hitter') or pitcher_ids (side='pitcher')
    who appear on `team` in `year`.
    """
    if side == "hitter":
        id_col   = "batter_id"
        team_col = "batter_team"
    else:
        id_col   = "pitcher_id"
        team_col = "pitcher_team"

    sql = text(f"""
        SELECT DISTINCT {id_col}
        FROM pitches
        WHERE {team_col} ILIKE :team
          AND year = :year
    """)
    with engine.connect() as conn:
        result = pd.read_sql(sql, conn, params={"team": f"%{team}%", "year": year})
    ids = result[id_col].tolist()
    print(f"  {team} {year} roster ({side}): {len(ids)} players")
    return ids


def pull_hitter_bbe_by_ids(engine, batter_ids: list, year: int = 2025) -> pd.DataFrame:
    """
    Pull 2025 BBE data for the specific batter_ids on the 2026 roster.
    """
    batter_ids = [i for i in batter_ids if i is not None]
    if not batter_ids:
        return pd.DataFrame()

    id_list = ", ".join(f"'{i}'" for i in batter_ids)
    sql = text(f"""
        SELECT
            batter_id,
            batter,
            year,
            exit_speed,
            angle
        FROM pitches
        WHERE exit_speed IS NOT NULL
          AND angle      IS NOT NULL
          AND batter_id  IN ({id_list})
          AND year       = :year
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"year": year})
    print(f"  Hitter BBE (2025, {len(batter_ids)} roster IDs): "
          f"{len(df):,} rows, {df['batter_id'].nunique()} with data")
    return df


def pull_fastball_vaa(engine, pitcher_ids: list, year: int = 2026) -> pd.DataFrame:
    """
    Pull fastball pitches for the given pitcher_ids and compute
    average VAA per pitcher.
    """
    pitcher_ids = [i for i in pitcher_ids if i is not None]
    if not pitcher_ids:
        return pd.DataFrame()

    id_list   = ", ".join(f"'{i}'" for i in pitcher_ids)
    fb_list   = ", ".join(f"'{pt}'" for pt in FASTBALL_TYPES)

    sql = text(f"""
        SELECT
            pitcher_id,
            pitcher,
            pitch_type,
            vert_appr_angle
        FROM pitches
        WHERE vert_appr_angle IS NOT NULL
          AND pitcher_id      IN ({id_list})
          AND pitch_type      IN ({fb_list})
          AND year            = :year
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"year": year})
    print(f"  Fastball VAA ({year}, {len(pitcher_ids)} roster IDs): "
          f"{len(df):,} pitches")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# VAA AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def build_fastball_vaa_profiles(fb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted-average fastball VAA per pitcher across all fastball types.
    Also reports per-type breakdown as a readable string for display.
    """
    overall = (
        fb_df
        .groupby(["pitcher_id", "pitcher"])
        .agg(
            vaa_fb       = ("vert_appr_angle", "mean"),
            vaa_fb_std   = ("vert_appr_angle", "std"),
            n_fastballs  = ("vert_appr_angle", "count"),
        )
        .reset_index()
    )
    overall = overall[overall["n_fastballs"] >= MIN_FB_PITCHES].copy()

    # Per-type summary string  e.g. "FF: -4.2° (n=312), SI: -3.8° (n=88)"
    by_type = (
        fb_df
        .groupby(["pitcher_id", "pitch_type"])
        .agg(vaa_mean=("vert_appr_angle", "mean"),
             n=("vert_appr_angle", "count"))
        .reset_index()
    )
    def fmt_types(pid):
        sub = by_type[by_type["pitcher_id"] == pid]
        return "  |  ".join(
            f"{r['pitch_type']}: {r['vaa_mean']:.1f}° (n={int(r['n'])})"
            for _, r in sub.iterrows()
        )
    overall["fb_type_breakdown"] = overall["pitcher_id"].apply(fmt_types)

    print(f"  {len(overall)} pitchers qualify (≥ {MIN_FB_PITCHES} fastballs)")
    return overall

# ─────────────────────────────────────────────────────────────────────────────
# MATCHUP SCORING
# ─────────────────────────────────────────────────────────────────────────────

def build_matchup_matrix(naa_df: pd.DataFrame, vaa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-join hitters × pitchers, compute plane_delta for every pairing.

    plane_delta = NAA + VAA_fb
        ≈  0  →  in-plane (hitter advantage)
        ≠  0  →  swing/pitch plane mismatch (pitcher advantage)
    """
    hitters = naa_df[
        naa_df["Natural_Attack_Angle"].notna() &
        (naa_df["BBE"] >= MIN_BBE_HITTER)
    ][["batter_id", "batter", "Natural_Attack_Angle", "BBE", "method"]].copy()
    hitters["_key"] = 1

    pitchers = vaa_df[
        ["pitcher_id", "pitcher", "vaa_fb", "n_fastballs", "fb_type_breakdown"]
    ].copy()
    pitchers["_key"] = 1

    matrix = hitters.merge(pitchers, on="_key").drop(columns="_key")
    matrix["plane_delta"]    = matrix["Natural_Attack_Angle"] + matrix["vaa_fb"]
    matrix["mismatch_score"] = matrix["plane_delta"].abs()

    def advantage(delta):
        if abs(delta) <= 2.0:
            return "In-Plane (Hitter)"
        elif delta > 0:
            return "Over-the-Top (Pitcher)"
        else:
            return "Depth Mismatch (Pitcher)"

    matrix["advantage"] = matrix["plane_delta"].apply(advantage)
    return matrix


def top_matchups_per_pitcher(matrix: pd.DataFrame, n: int = TOP_N_MATCHUPS) -> pd.DataFrame:
    """
    For each pitcher, return the N hitters with the highest mismatch_score
    (i.e., the batters whose swing plane is most disrupted by this pitcher's
    fastball VAA — the matchups the pitcher "wins" most decisively).
    """
    rows = []
    for (pid, pitcher_name), grp in matrix.groupby(["pitcher_id", "pitcher"]):
        top = grp.nlargest(n, "mismatch_score").copy()
        top["rank"] = range(1, len(top) + 1)
        rows.append(top)
    return pd.concat(rows, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

ADV_COLORS = {
    "In-Plane (Hitter)":       "#2c7bb6",
    "Over-the-Top (Pitcher)":  "#d7191c",
    "Depth Mismatch (Pitcher)":"#fdae61",
}

def plot_matchup(top_df: pd.DataFrame, naa_df: pd.DataFrame,
                 vaa_df: pd.DataFrame, team_a: str, team_b: str,
                 output_path: str = None):

    pitchers = vaa_df.sort_values("vaa_fb")
    n_pitchers = len(pitchers)

    # One row per pitcher
    fig_h = max(8, 2.2 * n_pitchers)
    fig, axes = plt.subplots(
        n_pitchers, 1,
        figsize=(14, fig_h),
        facecolor="#0f1117"
    )
    if n_pitchers == 1:
        axes = [axes]

    fig.suptitle(
        f"NAA vs. Fastball VAA — Top-{TOP_N_MATCHUPS} Matchups Per Pitcher\n"
        f"{team_b} Pitchers  ×  {team_a} Hitters  "
        f"(NAA from 2025 | VAA 2026 FB)",
        color="white", fontsize=12, fontweight="bold", y=1.01
    )

    for ax, (_, p_row) in zip(axes, pitchers.iterrows()):
        pid  = p_row["pitcher_id"]
        pname = p_row["pitcher"]
        vaa  = p_row["vaa_fb"]
        nfb  = int(p_row["n_fastballs"])
        breakdown = p_row["fb_type_breakdown"]

        sub = top_df[top_df["pitcher_id"] == pid].sort_values(
            "mismatch_score", ascending=True
        )

        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        bar_colors = [ADV_COLORS.get(a, "#888") for a in sub["advantage"]]
        y_pos = range(len(sub))
        y_labels = sub["batter"].tolist()

        bars = ax.barh(y_pos, sub["mismatch_score"],
                       color=bar_colors, edgecolor="none", height=0.55)

        for bar, row in zip(bars, sub.itertuples()):
            delta_str = f"Δ={row.plane_delta:+.1f}°"
            detail    = (f"  NAA={row.Natural_Attack_Angle:.1f}°  |  "
                         f"VAA={vaa:.1f}°  |  {row.advantage}  |  "
                         f"BBE={row.BBE}")
            ax.text(
                bar.get_width() + 0.15,
                bar.get_y() + bar.get_height() / 2,
                f"{delta_str}{detail}",
                va="center", color="#ccc", fontsize=7.5
            )

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(y_labels, color="white", fontsize=8)
        ax.set_xlim(0, sub["mismatch_score"].max() * 2.2)
        ax.set_xlabel("Mismatch Score  |Plane Delta|  (°)",
                      color="#aaa", fontsize=7)

        title = (f"{pname}  —  FB VAA: {vaa:.1f}°  "
                 f"(n={nfb} fastballs)   [{breakdown}]")
        ax.set_title(title, color="white", fontsize=8.5,
                     fontweight="bold", pad=5, loc="left")

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#0f1117")
        print(f"Figure saved → {output_path}")

    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("── Connecting ──────────────────────────────────────────────────")
    engine = get_engine(ENV_PATH)

    # ── Step 1: 2026 rosters ──────────────────────────────────────────────────
    print(f"\n── 2026 rosters ─────────────────────────────────────────────")
    hitter_ids  = pull_team_roster(engine, TEAM_A, year=2026, side="hitter")
    pitcher_ids = pull_team_roster(engine, TEAM_B, year=2026, side="pitcher")

    # ── Step 2: Pull hitter 2025 BBE (roster-filtered) ───────────────────────
    print(f"\n── Hitter BBE data (2025) ───────────────────────────────────")
    hitter_df = pull_hitter_bbe_by_ids(engine, hitter_ids, year=2025)

    if hitter_df.empty:
        print("  No 2025 data found for any 2026 roster hitter. "
              "Check batter_id consistency across years.")
        sys.exit(1)

    # ── Step 3: Compute NAA ───────────────────────────────────────────────────
    print(f"\n── Computing Natural Attack Angle ───────────────────────────")
    naa_df  = compute_natural_attack_angle(hitter_df)
    valid_h = naa_df[naa_df["Natural_Attack_Angle"].notna() &
                     (naa_df["BBE"] >= MIN_BBE_HITTER)]
    print(f"  {len(valid_h)} hitters qualify (BBE ≥ {MIN_BBE_HITTER})\n")
    print(valid_h[["batter", "Natural_Attack_Angle", "BBE", "method"]]
          .sort_values("Natural_Attack_Angle", ascending=False)
          .to_string(index=False))

    # ── Step 4: Pull fastball VAA (2026) ──────────────────────────────────────
    print(f"\n── Fastball VAA data (2026) — pitch types: "
          f"{', '.join(FASTBALL_TYPES)} ──────")
    fb_df = pull_fastball_vaa(engine, pitcher_ids, year=2026)

    if fb_df.empty:
        print("  No fastball VAA data found. Check pitch_type values in DB.")
        sys.exit(1)

    # ── Step 5: Build VAA profiles ────────────────────────────────────────────
    print(f"\n── Building fastball VAA profiles ───────────────────────────")
    vaa_df = build_fastball_vaa_profiles(fb_df)
    print(vaa_df[["pitcher", "vaa_fb", "vaa_fb_std", "n_fastballs",
                  "fb_type_breakdown"]]
          .sort_values("vaa_fb")
          .to_string(index=False))

    # ── Step 6: Matchup matrix ────────────────────────────────────────────────
    print(f"\n── Building matchup matrix ────────────────────────────────────")
    matrix = build_matchup_matrix(naa_df, vaa_df)

    # ── Step 7: Top-N hitters per pitcher ─────────────────────────────────────
    print(f"\n── Top-{TOP_N_MATCHUPS} hitters per pitcher "
          f"(biggest plane mismatch) ──────────────")
    top_df = top_matchups_per_pitcher(matrix, n=TOP_N_MATCHUPS)

    for (pid, pname), grp in top_df.groupby(["pitcher_id", "pitcher"]):
        vaa = vaa_df.loc[vaa_df["pitcher_id"] == pid, "vaa_fb"].values[0]
        print(f"\n  {pname}  (FB VAA: {vaa:.1f}°)")
        print(grp[["rank", "batter", "plane_delta",
                   "Natural_Attack_Angle", "advantage"]]
              .rename(columns={"Natural_Attack_Angle": "NAA"})
              .to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    top_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nMatchup output saved → {OUTPUT_CSV}")

    plot_matchup(top_df, naa_df, vaa_df, TEAM_A, TEAM_B,
                 output_path=OUTPUT_FIG)
