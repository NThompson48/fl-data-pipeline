"""
fl_prospect_position_export.py -- position / role tables for the dashboard's Other-League Lookup and Manual Calculator pages  (user 2026-10-28: "add a position toggle ... where you can choose the position")
HITTERS: for each position group (C, 1B, 2B, 3B, SS, LF, CF, RF, DH) the production WAR parts that depend on position: the position adjustment (runs per PA: 2026 position runs / PA of the group, fl_hitter_projection: pos_group_rate) and the baserunning runs per PA the production
baserunning model gives that position (fl_bsr_proxy), plus the constants the WAR simulation uses (replacement credit, runs per win, position residual SD per PA, baserunning true-talent variance and one-season noise).  Fielding is not in the production WAR (WAR_INCLUDE_FIELDING = False).
  A prospect's WAR per 400 PA at a position = 400 x ((wOBA - lg wOBA) / wOBA scale + position runs + baserunning runs + replacement credit) / runs per win.
PITCHERS: role SP / RP: the innings per appearance (the RA9 baseline's IP-per-appearance input; IP-weighted mean of the published ip_per_g_prior among 2026 pitchers with >= 20 IP, by role) and the median first-season innings of pitchers new to the league by role (2026 debuts with no 2025 record, war_pitchers).
Output: output/fl_prospect_position_table.json  (+ copy to the dashboard data folder with --publish)
"""
import json
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

import fl_bsr_proxy as bsr
import fl_hitter_projection as hp
from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
POS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]


def main(publish=False):
    pt = bsr.positions(2026); pt = pt[pt["pos_adj"].notna()]; grp = pt["pos"] if hp.POS_DH_GROUP else pt["field_pos"]; pos_rate = (pt.groupby(grp)["pos_adj"].sum() / pt.groupby(grp)["pa"].sum())
    bm = bsr.production_model(); bs = bsr.predict(bm, pd.Series(POS), pd.Series([np.nan] * len(POS)), pd.Series([np.nan] * len(POS))); proj = pd.read_parquet("output/fl_2027_hitter_projections.parquet")
    hit = {p: {"pos_rate": float(pos_rate.get(p, 0.0)), "bsr_rate": float(bs[i]), "pa_2026": float(pt.groupby(grp)["pa"].sum().get(p, 0.0))} for i, p in enumerate(POS)}
    consts = {"repl_per_pa": float(proj["repl_per_pa"].iloc[0]), "rpw": float(proj["rpw"].iloc[0]), "pos_resid_per_pa": hp.POS_RESID_PER_PA, "bsr_tau2": float(bm["tau2"]), "bsr_noise": float(bm["noise"])}
    print("HITTER position runs per PA (and per 600 PA) | baserunning per PA:"); print(pd.DataFrame(hit).T.assign(pos_600=lambda d: 600 * d["pos_rate"], bsr_600=lambda d: 600 * d["bsr_rate"]).round(4).to_string()); print(consts)
    pr = pd.read_parquet("output/fl_ra9_trustmark_projection.parquet"); extra = pr[["pitcher_id", "role", "ip_per_g_prior", "ip_2026", "ip_2025"]].copy(); extra = extra[extra["ip_2026"].fillna(0) >= 20]
    ipg = {r: float(np.average(g["ip_per_g_prior"], weights=g["ip_2026"])) for r, g in extra.groupby("role")}
    w = read_sql_df("select player, year, gs, g, ip from war_pitchers where game_type='reg_season'"); w["role"] = np.where(w["gs"] / w["g"].clip(lower=1) >= 0.5, "SP", "RP"); w26 = w[w["year"] == 2026]; has25 = set(w[w["year"] == 2025]["player"]); deb = w26[~w26["player"].isin(has25) & (w26["ip"] > 0)]
    debut = {r: {"median_ip": float(g["ip"].median()), "n": int(len(g))} for r, g in deb.groupby("role")}
    print("\nPITCHER role: IP per appearance (IP-weighted, 2026 >= 20 IP):", {k: round(v, 2) for k, v in ipg.items()}, "| debut first-season IP:", debut, "| all debuts median", round(float(deb["ip"].median()), 1))
    out = {"hitters": hit, "hitter_constants": consts, "pitchers": {r: {"ip_per_g": ipg.get(full), "debut_ip": debut.get(r, {}).get("median_ip"), "debut_n": debut.get(r, {}).get("n")} for r, full in (("SP", "Starter"), ("RP", "Reliever"))}, "roles_in_table": sorted(ipg)}
    json.dump(out, open("output/fl_prospect_position_table.json", "w"), indent=1)
    if publish:
        shutil.copy("output/fl_prospect_position_table.json", "../projections-dashboard/data/fl_prospect_position_table.json"); print("published")


if __name__ == "__main__":
    main(publish="--publish" in sys.argv)
