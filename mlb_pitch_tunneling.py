"""
mlb_pitch_tunneling.py -- quantify pitch tunneling: do a pitcher's pitch types
look alike for as long as possible (deceiving the batter's swing decision),
then diverge sharply before the plate?

Needs the full 9-parameter trajectory (release position x0/y0/z0, initial
velocity vx0/vy0/vz0, acceleration ax/ay/az) to reconstruct where the ball
actually is at the batter's DECISION point -- not just release and plate,
which is all wh_sva's location/shape features use. Only the untrimmed
2022-2025 Statcast files still carry vx0/vy0/vz0/ax/ay/az (dropped from the
2018-2021 pull's KEEP_COLS to save space), so this is scoped to those 4 years.

Method (matches the public "tunnel point" convention, e.g. Baseball
Prospectus's Pavlidis/Long/Judge tunneling work): the batter's swing-commit
point is ~167ms before the ball crosses the front of the plate (human
reaction-time research) -- reconstruct the ball's (x, z) position at that
instant for every pitch, via the standard constant-acceleration equations:
    y(t) = y0 + vy0*t + 0.5*ay*t^2     (solve for t at y=Y_PLATE, then again at
    x(t) = x0 + vx0*t + 0.5*ax*t^2      t_tunnel = t_plate - 0.167s)
    z(t) = z0 + vz0*t + 0.5*az*t^2

Then for each pitcher-season's primary two pitch types (by usage), compute:
  release_sep -- distance between the two pitch types' avg release points
  tunnel_sep  -- distance between their avg positions at the decision point
  plate_sep   -- distance between their avg plate_x/plate_z
  tunnel_ratio = plate_sep / tunnel_sep   (higher = pitches still look alike
                                            at the decision point but diverge
                                            sharply by the plate -- effective
                                            tunneling)

Usage:
    python mlb_pitch_tunneling.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlb_marcel as mm
from mlb_kbb_model import MIN_PA

pd.set_option("display.width", 160, "display.max_columns", 40)

Y_PLATE = 1.417   # ft, front edge of home plate -- Statcast's own plate_x/plate_z convention
DECISION_MS = 0.167  # seconds before plate-crossing a batter must commit (reaction-time literature)
YEARS = [2022, 2023, 2024, 2025]
MIN_PITCH_USAGE = 100  # min pitches of a type in a pitcher-season to count as a real offering


def solve_t_plate(y0, vy0, ay, y_target=Y_PLATE):
    """0.5*ay*t^2 + vy0*t + (y0-y_target) = 0 -- take the smaller positive root
    (the ball's first crossing of y_target, i.e. actual flight time ~0.4-0.5s)."""
    a, b, c = 0.5 * ay, vy0, (y0 - y_target)
    disc = b ** 2 - 4 * a * c
    disc = np.where(disc < 0, np.nan, disc)
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    # physical flight time is small and positive (~0.35-0.55s); pick whichever root qualifies
    t = np.where((t1 > 0.1) & (t1 < 1.0), t1, t2)
    return t


def pos_at_t(x0, v0, a, t):
    return x0 + v0 * t + 0.5 * a * t ** 2


if __name__ == "__main__":
    cols = ["pitcher", "game_year", "pitch_type", "player_name",
            "release_pos_x", "release_pos_y", "release_pos_z",
            "vx0", "vy0", "vz0", "ax", "ay", "az",
            "plate_x", "plate_z", "p_throws", "description", "events"]
    frames = []
    for y in YEARS:
        d = pd.read_parquet(f"statcast_{y}.parquet", columns=cols)
        frames.append(d)
        print(f"  loaded {y}: {len(d):,} rows")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["vy0", "ay", "release_pos_y", "release_pos_x", "release_pos_z",
                            "vx0", "vz0", "ax", "az", "plate_x", "plate_z"])
    print(f"total usable rows: {len(df):,}")

    print("\nReconstructing trajectories...")
    df["t_plate"] = solve_t_plate(df["release_pos_y"].values, df["vy0"].values, df["ay"].values)
    df["x_plate_recon"] = pos_at_t(df["release_pos_x"].values, df["vx0"].values, df["ax"].values, df["t_plate"].values)
    df["z_plate_recon"] = pos_at_t(df["release_pos_z"].values, df["vz0"].values, df["az"].values, df["t_plate"].values)

    # --- validation: does the reconstructed plate position match Statcast's own? ---
    err_x = (df["x_plate_recon"] - df["plate_x"]).abs()
    err_z = (df["z_plate_recon"] - df["plate_z"]).abs()
    print(f"reconstruction check vs Statcast's own plate_x/plate_z: "
          f"median|err_x|={err_x.median():.4f}ft, median|err_z|={err_z.median():.4f}ft, "
          f"95th pct |err_x|={err_x.quantile(.95):.4f}ft")
    df = df[(err_x < 0.5) & (err_z < 0.5)]  # drop any pitch where reconstruction clearly failed
    print(f"rows retained after validation filter: {len(df):,}")

    df["t_tunnel"] = df["t_plate"] - DECISION_MS
    df["x_tunnel"] = pos_at_t(df["release_pos_x"].values, df["vx0"].values, df["ax"].values, df["t_tunnel"].values)
    df["z_tunnel"] = pos_at_t(df["release_pos_z"].values, df["vz0"].values, df["az"].values, df["t_tunnel"].values)

    print("\nComputing per-pitcher-season pitch-type usage...")
    usage = df.groupby(["pitcher", "game_year", "pitch_type"]).size().reset_index(name="n")
    usage = usage[usage["n"] >= MIN_PITCH_USAGE]
    top2 = usage.sort_values("n", ascending=False).groupby(["pitcher", "game_year"]).head(2)
    top2["rank"] = top2.groupby(["pitcher", "game_year"]).cumcount()
    pairs = top2.pivot(index=["pitcher", "game_year"], columns="rank", values="pitch_type").reset_index()
    pairs = pairs.dropna(subset=[0, 1]).rename(columns={0: "type_A", 1: "type_B"})
    print(f"pitcher-seasons with a qualifying primary pitch-type pair: {len(pairs)}")

    agg_cols = ["release_pos_x", "release_pos_z", "x_tunnel", "z_tunnel", "plate_x", "plate_z"]
    means = df.groupby(["pitcher", "game_year", "pitch_type"])[agg_cols].mean().reset_index()

    m = pairs.merge(means, left_on=["pitcher", "game_year", "type_A"], right_on=["pitcher", "game_year", "pitch_type"])
    m = m.merge(means, left_on=["pitcher", "game_year", "type_B"], right_on=["pitcher", "game_year", "pitch_type"],
                suffixes=("_A", "_B"))

    def sep(a_x, a_z, b_x, b_z):
        return np.sqrt((m[a_x] - m[b_x]) ** 2 + (m[a_z] - m[b_z]) ** 2)

    m["release_sep"] = sep("release_pos_x_A", "release_pos_z_A", "release_pos_x_B", "release_pos_z_B")
    m["tunnel_sep"] = sep("x_tunnel_A", "z_tunnel_A", "x_tunnel_B", "z_tunnel_B")
    m["plate_sep"] = sep("plate_x_A", "plate_z_A", "plate_x_B", "plate_z_B")
    m["tunnel_ratio"] = m["plate_sep"] / m["tunnel_sep"].replace(0, np.nan)

    print(f"\ntunnel_ratio distribution:\n{m['tunnel_ratio'].describe()}")
    print(f"\nrelease_sep (ft): median={m['release_sep'].median():.3f}, "
          f"tunnel_sep (ft): median={m['tunnel_sep'].median():.3f}, "
          f"plate_sep (ft): median={m['plate_sep'].median():.3f}")

    # --- does tunnel_ratio actually predict whiff rate / K rate? ---
    is_swing = df["description"].isin(["foul", "foul_tip", "hit_into_play", "swinging_strike",
                                         "swinging_strike_blocked", "foul_bunt", "missed_bunt", "bunt_foul_tip"])
    is_whiff = df["description"].isin(["swinging_strike", "swinging_strike_blocked", "missed_bunt"])
    df["is_swing"], df["is_whiff"] = is_swing, is_whiff
    season_whiff = df[df["is_swing"]].groupby(["pitcher", "game_year"]).agg(
        n_swing=("is_whiff", "size"), whiff_rate=("is_whiff", "mean")).reset_index()

    from mlb_kbb_model import PANEL_PATH
    panel = pd.read_parquet(PANEL_PATH)
    check = m.merge(season_whiff, on=["pitcher", "game_year"], how="left")
    check = check.merge(panel[["pitcher", "game_year", "K_rate", "PA", "wh_sva_fb", "wh_sva_brk"]],
                         on=["pitcher", "game_year"], how="left")
    check = check[(check["PA"] >= MIN_PA) & check["n_swing"].fillna(0).ge(200)]
    print(f"\nn pitcher-seasons for validation = {len(check)}")
    print(f"corr(tunnel_ratio, whiff_rate) = {check['tunnel_ratio'].corr(check['whiff_rate']):.4f}")
    print(f"corr(tunnel_ratio, K_rate) = {check['tunnel_ratio'].corr(check['K_rate']):.4f}")
    print(f"corr(release_sep, K_rate) = {check['release_sep'].corr(check['K_rate']):.4f}  "
          f"(smaller release_sep = more consistent arm action -- does that alone matter?)")
    print(f"corr(plate_sep, K_rate) = {check['plate_sep'].corr(check['K_rate']):.4f}  "
          f"(raw arsenal separation at the plate, no tunneling logic at all)")

    resid = check["K_rate"] - check[["wh_sva_fb", "wh_sva_brk"]].fillna(0).sum(axis=1)
    print(f"\npartial check -- corr(tunnel_ratio, K_rate residual after wh_sva_fb+wh_sva_brk) = "
          f"{check['tunnel_ratio'].corr(resid):.4f}  (does tunneling add anything wh_sva doesn't already capture?)")

    check.to_csv("output/pitch_tunneling_2022_2025.csv", index=False)
    print("\nSaved -> output/pitch_tunneling_2022_2025.csv")
