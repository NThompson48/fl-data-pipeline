"""
fl_pitcher_runs_dated.py -- dated outs and runs charged per pitcher, 2026, from the iScore play-by-play (pbp_events; 2025 has no play-by-play).

outs_on_play is credited to the pitcher on the mound (`pitcher_name`); `runs_charged_to` is a pipe-joined list with one pitcher name per run scored on the play (inherited runners are
charged to the pitcher who put them on).  Output: output/pitcher_runs_dated_2026.parquet (name_clean, date, outs, runs) plus a reconciliation against the published season totals.
Usage:  python fl_pitcher_runs_dated.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

import fl_innings_projection as ip
from fl_supabase import read_sql_df

OUT = "output/pitcher_runs_dated_2026.parquet"


def first_last_to_clean(n: str) -> str:
    return ip.sc.clean_name(n)


def build():
    e = read_sql_df(text("select date, game_guid, pitcher_name, outs_on_play, runs_on_play, runs_charged_to from pbp_events where outs_on_play > 0 or runs_on_play > 0"))
    e["date"] = pd.to_datetime(e["date"]); e["name_clean"] = e["pitcher_name"].astype(str).apply(first_last_to_clean)
    outs = e.groupby(["name_clean", "date"])["outs_on_play"].sum().rename("outs").reset_index()
    toks = e[e["runs_charged_to"].notna()][["date", "runs_charged_to"]].copy()
    toks["name"] = toks["runs_charged_to"].str.split("|"); toks = toks.explode("name"); toks["name_clean"] = toks["name"].apply(first_last_to_clean)
    runs = toks.groupby(["name_clean", "date"]).size().rename("runs").reset_index()
    d = outs.merge(runs, on=["name_clean", "date"], how="outer").fillna({"outs": 0, "runs": 0})
    return d


def main():
    d = build(); d.to_parquet(OUT, index=False)
    w = read_sql_df(text("select player, ip, ra from war_pitchers where year=2026 and game_type='reg_season'")); w["name_clean"] = w["player"].apply(ip.lastfirst_to_clean)
    w = w.groupby("name_clean").agg(ip=("ip", "sum"), ra=("ra", "sum"))
    tot = d.groupby("name_clean").agg(outs=("outs", "sum"), runs=("runs", "sum")); tot["ip_pbp"] = tot["outs"] / 3
    j = tot.join(w, how="inner"); j = j[j["ip"] >= 20]
    print(f"{len(j)} pitchers with >= 20 IP; IP: pbp vs published corr {j['ip_pbp'].corr(j['ip']):.4f}, mean ratio {(j['ip_pbp']/j['ip']).median():.3f}; runs: corr {j['runs'].corr(j['ra']):.4f}, mean ratio {(j['runs']/j['ra']).median():.3f}")
    print(f"league RA9 pbp {9*tot['runs'].sum()/tot['ip_pbp'].sum():.3f} vs published {9*w['ra'].sum()/w['ip'].sum():.3f}; dates {d['date'].min().date()} to {d['date'].max().date()}")


if __name__ == "__main__":
    main()
