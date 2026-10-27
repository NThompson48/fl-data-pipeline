"""
fl_hitter_calc_export.py -- reference means for the dashboard's hitter MANUAL CALCULATOR  (user 2026-10-27: "the hitter tab also needs a manual calculator")
The other-league hitter bridges (fl_hitter_bridge.py MiLB / fl_hitter_bridge_pioneer.py / fl_hitter_bridge_college.py) measure each source season against a reference mean: MiLB = the same level-season PA-weighted mean, Pioneer = the same season's Pioneer mean,
College = the all-college season mean (basis raw) or the conference-season mean (basis conf / conf+, plus the conference-strength term).  This writes those reference means (wOBA proxy, K%, BB%, HR/PA) to output/fl_hitter_calc_tables.json so a typed-in line can be translated with the
identical arithmetic (the coefficients are already in fl_hitter_bridge_coefs.json).   --publish copies it to the dashboard data folder.
"""
import json
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

import fl_hitter_bridge_college as hc
import fl_hitter_bridge_test as hb
from fl_hitter_bridge_pioneer_fit import pioneer_table

warnings.filterwarnings("ignore")
S = ("woba", "k", "bbr", "hrr")


def wmean(g, s):
    ok = g[s].notna(); return float(np.average(g.loc[ok, s], weights=g.loc[ok, "pa"]))


def main(publish=False):
    m = hb.milb_table(); milb = [{"level": lv, "season": int(se), **{s: wmean(g, s) for s in S}, "pa": float(g["pa"].sum())} for (lv, se), g in m.groupby(["level", "season"])]
    t = pioneer_table(); pio = [{"season": int(se), **{s: wmean(g, s) for s in S}, "pa": float(g["pa"].sum())} for se, g in t.groupby("season")]
    c = hc.college_table(); nat = [{"season": int(se), **{s: wmean(g, s) for s in S}} for se, g in c.groupby("season")]
    conf = [{"season": int(se), "conference": cf, **{s: wmean(g, s) for s in S}, "pa": float(g["pa"].sum())} for (se, cf), g in c.groupby(["season", "conference"])]
    out = {"milb": milb, "milb_groups": hb.GROUPS, "pioneer": pio, "college_national": nat, "college_conference": conf, "conferences": sorted(set(c["conference"]) - {"UNKNOWN"}) + ["UNKNOWN"]}
    json.dump(out, open("output/fl_hitter_calc_tables.json", "w"), indent=1)
    print(f"{len(milb)} MiLB level-seasons, {len(pio)} Pioneer seasons, {len(nat)} college seasons, {len(conf)} conference-seasons, {len(out['conferences'])} conferences")
    if publish:
        shutil.copy("output/fl_hitter_calc_tables.json", "../projections-dashboard/data/fl_hitter_calc_tables.json"); print("published")


if __name__ == "__main__":
    main(publish="--publish" in sys.argv)
