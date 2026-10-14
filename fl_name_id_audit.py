"""
fl_name_id_audit.py -- name-keyed identity audit (hitters AND pitchers): the models key hitters on the cleaned name (the pitch tables carry batter_id but the panels do not).
(1) names mapping to > 1 id: either ONE person re-issued an id (merge is right) or DIFFERENT people sharing a name (merge is wrong).  Distinguished by CONCURRENCY: two ids active in the same game / same season on
    different teams must be different people.
(2) ids carrying > 1 cleaned name: one person under several spellings (a name-keyed history is split).
Output: output/name_id_audit_{hitters,pitchers}.parquet
"""
import warnings
import pandas as pd
from sqlalchemy import text
from fl_supabase import read_sql_df
import fl_innings_projection as ip

warnings.filterwarnings("ignore")


def audit(lab, idc, nc, tc):
    d = read_sql_df(text(f"select {idc} as pid, {nc} as nm, {tc} as team, year, game_id, count(*) as n from pitches where game_type='reg_season' and {idc} is not null group by {idc}, {nc}, {tc}, year, game_id"))
    d["name_clean"] = d["nm"].apply(ip.lastfirst_to_clean); d = d[d["year"].notna()]
    per_id = d.groupby(["name_clean", "pid"]).agg(n=("n", "sum"), games=("game_id", "nunique"), first_year=("year", "min"), last_year=("year", "max"), teams=("team", lambda s: "/".join(sorted(set(map(str, s)))))).reset_index()
    multi = per_id.groupby("name_clean")["pid"].nunique(); multi = multi[multi > 1].index
    # concurrency: any game in which two ids of the same name both appear => different people
    g = d[d["name_clean"].isin(multi)].groupby(["name_clean", "game_id"])["pid"].nunique(); conc_names = set(g[g > 1].reset_index()["name_clean"])
    # same season, different teams, different ids (could still be a traded player with a re-issued id)
    ys = d[d["name_clean"].isin(multi)].groupby(["name_clean", "pid", "year"])["team"].agg(lambda s: set(map(str, s))).reset_index()
    out = per_id[per_id["name_clean"].isin(multi)].copy(); out["same_game_collision"] = out["name_clean"].isin(conc_names)
    out["total_n"] = out.groupby("name_clean")["n"].transform("sum")
    out.to_parquet(f"output/name_id_audit_{lab}.parquet", index=False)
    split = d.groupby("pid")["name_clean"].nunique(); split = split[split > 1].index
    sp = d[d["pid"].isin(split)].groupby(["pid", "name_clean"]).agg(n=("n", "sum"), years=("year", lambda s: ",".join(sorted({str(int(x)) for x in s})))).reset_index()
    print(f"\n{lab.upper()}: {d.pid.nunique()} ids, {d.name_clean.nunique()} names | names with >1 id: {len(multi)} (of which two ids in the SAME game: {len(conc_names)}) | ids with >1 name: {len(split)}")
    print("  same-game collisions (certainly different people):"); print(out[out["same_game_collision"]].sort_values(["name_clean", "n"], ascending=[True, False])[["name_clean", "pid", "n", "games", "first_year", "last_year", "teams"]].to_string(index=False))
    print("  names with >1 id, NOT concurrent (id re-issue or same-name players in different periods): ", out[~out["same_game_collision"]]["name_clean"].nunique())
    print("  split spellings:"); print(sp.to_string(index=False))
    return out, sp


if __name__ == "__main__":
    audit("hitters", "batter_id", "batter", "batter_team"); audit("pitchers", "pitcher_id", "pitcher", "pitcher_team")
