"""
fl_hitter_panel.py -- one hitter-season panel for the Frontier League, 2022-2026.

  2022-24  StatsCrew batting lines (output/fl_statscrew_batting_2022_2024.parquet), no pitch data.
  2025-26  Supabase: lineup_stats (PA/AB/H/TB/BB/HBP/SO/SF/SH, FL wOBA, xwOBA, wRC+), 1B/2B/3B/HR counts from the
           `pitches` table (reg_season terminal pitches), war_players (WAR parts: park factor, position adj,
           fielding, baserunning, replacement runs, WAR), advanced_splits (Trackman skill metrics, All/All rows).

Every season gets the same count-based wOBA, computed with the project's FL linear weights (woba_weights table),
so 2022-24 and 2025-26 are measured identically; the pipeline's own 2025-26 wOBA is kept for reconciliation.
Players are keyed by cleaned name ("first last"); traded players are summed across stints within a season.

Usage:  python fl_hitter_panel.py        (rebuilds output/fl_hitter_panel.parquet and prints diagnostics)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

import fl_innings_projection as ip
import fl_statscrew_pull as sc
from fl_supabase import read_sql_df

OUT_PATH = "output/fl_hitter_panel.parquet"
SC_PATH = "output/fl_statscrew_batting_2022_2024.parquet"


def woba_weights() -> dict:
    w = read_sql_df(text("select outcome, woba_weight from woba_weights")).set_index("outcome")["woba_weight"]
    return {k: float(w[k]) for k in ("BB", "HBP", "1B", "2B", "3B", "HR")}


def count_woba(d: pd.DataFrame, w: dict) -> pd.Series:
    """wOBA from counts. uBB = BB - IBB (IBB only recorded in 2022 StatsCrew lines)."""
    ubb = d["bb"] - d["ibb"].fillna(0)
    den = d["ab"] + ubb + d["sf"].fillna(0) + d["hbp"].fillna(0)
    num = w["BB"] * ubb + w["HBP"] * d["hbp"].fillna(0) + w["1B"] * d["x1b"] + w["2B"] * d["x2b"] + w["3B"] * d["x3b"] + w["HR"] * d["hr"]
    return num / den.replace(0, np.nan)


def load_statscrew() -> pd.DataFrame:
    s = pd.read_parquet(SC_PATH)
    s["name_clean"] = s["player"].apply(sc.clean_name)
    cols = ["pa", "ab", "h", "x2b", "x3b", "hr", "bb", "so", "hbp", "sf", "sh", "ibb", "sb", "cs", "r", "rbi", "tb", "gp"]
    g = s.groupby(["name_clean", "year"], as_index=False).agg(**{c: (c, "sum") for c in cols}, n_teams=("team", "nunique"), team=("team", "first"))
    g["x1b"] = g["h"] - g["x2b"] - g["x3b"] - g["hr"]
    g["source"] = "statscrew"
    return g


def load_trackman_seasons() -> pd.DataFrame:
    ls = read_sql_df(text("""select player, team, year_label as year, pa, ab, h, tb, bb, hbp, so, sf, sh, woba as woba_pipeline, xwoba, xba, xslg,
                                    wrc_plus, primary_pos, side from lineup_stats
                             where game_type='reg_season' and vs='All' and role='batter' and year_label in ('2025','2026')"""))
    ls["year"] = ls["year"].astype(int)
    hits = read_sql_df(text("""select batter as player, year,
            sum(case when play_result='double' then 1 else 0 end) x2b, sum(case when play_result='triple' then 1 else 0 end) x3b,
            sum(case when play_result='home_run' then 1 else 0 end) hr
          from pitches where game_type='reg_season' and year in (2025, 2026) and play_result in ('double','triple','home_run')
          group by 1, 2"""))
    hits["year"] = hits["year"].astype(int)
    ls = ls.merge(hits, on=["player", "year"], how="left")
    for c in ("x2b", "x3b", "hr"):
        ls[c] = ls[c].fillna(0)
    # batted-ball profile (air balls = launch angle >= 10 deg, the same definition as the pitcher-side Air%) and the share of
    # the hitter's plate appearances taken at TrustmarkPark -- both feed the Mississippi translation / park neutralisation
    bb = read_sql_df(text("""select batter as player, year,
            sum(case when k_or_bb in ('strikeout','walk') or pitch_call = 'hit_by_pitch'
                       or play_result in ('out','single','double','triple','home_run','fielders_choice','error','sacrifice') then 1 else 0 end) as pa_pitch,
            sum(case when (k_or_bb in ('strikeout','walk') or pitch_call = 'hit_by_pitch'
                       or play_result in ('out','single','double','triple','home_run','fielders_choice','error','sacrifice'))
                      and split_part(game_id, '-', 2) = 'TrustmarkPark' then 1 else 0 end) as pa_trustmark,
            sum(case when exit_speed is not null and angle is not null and play_result in ('out','single','double','triple','home_run','fielders_choice','error') then 1 else 0 end) as bbe,
            sum(case when exit_speed is not null and angle >= 10 and play_result in ('out','single','double','triple','home_run','fielders_choice','error') then 1 else 0 end) as air_n,
            sum(case when angle >= 10 and play_result = 'home_run' then 1 else 0 end) as air_hr,
            sum(case when angle >= 10 and play_result in ('single','double','triple') then 1 else 0 end) as air_hit_nonhr
          from pitches where game_type='reg_season' and year in (2025, 2026) group by 1, 2"""))
    bb["year"] = bb["year"].astype(int)
    ls = ls.merge(bb, on=["player", "year"], how="left")
    ls["x1b"] = ls["h"] - ls["x2b"] - ls["x3b"] - ls["hr"]
    ls["name_clean"] = ls["player"].apply(ip.lastfirst_to_clean)
    # one row per (name_clean, year): name aliases (fl_statscrew_pull.NAME_ALIASES, 2026-10-13) can map two source spellings to one hitter -> sum the counts,
    # PA-weight the rate columns, take descriptors from his larger stint
    cnt = ["pa", "ab", "h", "tb", "bb", "hbp", "so", "sf", "sh", "x2b", "x3b", "hr", "x1b", "pa_pitch", "pa_trustmark", "bbe", "air_n", "air_hr", "air_hit_nonhr"]
    rate = ["woba_pipeline", "xwoba", "xba", "xslg", "wrc_plus"]
    ls = ls.sort_values("pa", ascending=False)
    agg = ls.groupby(["name_clean", "year"]).agg(**{c: (c, "sum") for c in cnt if c in ls}, **{c: (c, "first") for c in ("player", "team", "primary_pos", "side")})
    for c in rate:
        agg[c] = ls.assign(_w=ls[c] * ls["pa"], _p=ls["pa"].where(ls[c].notna(), 0)).groupby(["name_clean", "year"]).apply(lambda g: g["_w"].sum() / g["_p"].sum() if g["_p"].sum() > 0 else np.nan)
    ls = agg.reset_index()
    war = read_sql_df(text("""select player, year, war, rar, repl_runs, pos_adj_runs, fielding_runs, bsr_runs, batting_runs, park_factor as bat_park_factor,
                                     primary_pos as war_pos from war_players where game_type='reg_season'"""))
    war["name_clean"] = war["player"].apply(ip.lastfirst_to_clean)
    war["year"] = war["year"].astype(int)
    war = war.groupby(["name_clean", "year"], as_index=False).agg(**{c: (c, "sum") for c in ("war", "rar", "repl_runs", "pos_adj_runs", "fielding_runs", "bsr_runs", "batting_runs")},
                                                                 bat_park_factor=("bat_park_factor", "first"), war_pos=("war_pos", "first"), player=("player", "first"))
    ls = ls.merge(war.drop(columns="player"), on=["name_clean", "year"], how="left")
    adv = read_sql_df(text("""select player, year, avg_ev, ev90, max_ev, barrel_pct, hardhit_pct, sweetspot_pct, pull_air_pct, gb_pct, whiff_pct, chase_pct,
                                     zswing_pct, zcontact_pct, swing_pct, xdamage, "PA" as pa_adv, game_type
                              from advanced_splits where role='batter' and venue='All' and opp_tier='All' and game_type in ('reg_season','All')
                                    and year in ('2025','2026')"""))
    adv["name_clean"] = adv["player"].apply(ip.lastfirst_to_clean)
    adv["year"] = adv["year"].astype(int)
    # 2026 rows carry game_type 'All'; for 2025 prefer the regular-season row over the all-games row
    adv = adv.assign(_pref=(adv["game_type"] == "reg_season").astype(int)).sort_values(["_pref", "pa_adv"], ascending=False)
    adv = adv.drop_duplicates(["name_clean", "year"]).drop(columns=["_pref", "game_type"])
    ls = ls.merge(adv.drop(columns=["player"]), on=["name_clean", "year"], how="left")
    ls["source"] = "trackman"
    ls["n_teams"] = 1
    return ls


def build_panel() -> pd.DataFrame:
    w = woba_weights()
    a, b = load_statscrew(), load_trackman_seasons()
    p = pd.concat([a, b], ignore_index=True)
    p["woba"] = count_woba(p, w)
    p["k"] = p["so"] / p["pa"].replace(0, np.nan)
    p["bbr"] = p["bb"] / p["pa"].replace(0, np.nan)
    p["hbpr"] = p["hbp"].fillna(0) / p["pa"].replace(0, np.nan)
    p["hrr"] = p["hr"] / p["pa"].replace(0, np.nan)
    p["iso"] = (p["tb"] - p["h"]) / p["ab"].replace(0, np.nan)
    p["babip"] = (p["h"] - p["hr"]) / (p["ab"] - p["so"] - p["hr"] + p["sf"].fillna(0)).replace(0, np.nan)
    return p


if __name__ == "__main__":
    p = build_panel()
    p.to_parquet(OUT_PATH, index=False)
    print(f"saved {OUT_PATH}: {len(p)} hitter-seasons")
    lg = p.groupby("year").apply(lambda g: pd.Series({
        "hitters": len(g), "PA": g["pa"].sum(), "K%": g["so"].sum() / g["pa"].sum(), "BB%": g["bb"].sum() / g["pa"].sum(),
        "HR/PA": g["hr"].sum() / g["pa"].sum(), "ISO": (g["tb"].sum() - g["h"].sum()) / g["ab"].sum(),
        "wOBA (count-based)": np.average(g["woba"].fillna(0), weights=g["pa"]), "n>=100 PA": int((g["pa"] >= 100).sum())}))
    print(lg.round(4).to_string())
    t = p[p["source"] == "trackman"].dropna(subset=["woba", "woba_pipeline"])
    for y, g in t.groupby("year"):
        d = g[g["pa"] >= 50]
        print(f"{y}: count-based vs pipeline wOBA (PA>=50, n={len(d)}): corr {d['woba'].corr(d['woba_pipeline']):.4f}, mean diff {(d['woba'] - d['woba_pipeline']).mean():+.4f}, "
              f"max |diff| {(d['woba'] - d['woba_pipeline']).abs().max():.4f}")
        hh = g.assign(hits_chk=g["x1b"] + g["x2b"] + g["x3b"] + g["hr"])
        print(f"    hit decomposition: negative singles {(g['x1b'] < 0).sum()}; hr total {int(g['hr'].sum())}; PA-weighted HR/PA {g['hr'].sum()/g['pa'].sum():.4f}")
    print("share of 2025-26 hitters with Trackman skill metrics:", round(t["avg_ev"].notna().mean(), 3), "| with WAR parts:", round(t["war"].notna().mean(), 3))
