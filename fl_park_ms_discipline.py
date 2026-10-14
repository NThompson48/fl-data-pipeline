"""
fl_park_ms_discipline.py -- WHY do Mississippi hitters strike out so much less at home?  (pitch level, 2025-26; follows fl_park_ms_kbb.py: K% 14.1% home vs 19.6% road, DiD -4.7 pts)
Pitch-level fixed-effects regressions: outcome = year + hitter + pitcher + park effects + MS_HOME (a Mississippi hitter's pitches at Trustmark) + HOME_OTHER (a non-Mississippi hitter's pitches in his own park, generic home advantage).
The MS_HOME coefficient = what Mississippi hitters do differently at home beyond other teams' generic home advantage, holding hitter, pitcher and park fixed.  Game-cluster robust SEs.
Outcomes: swing (all pitches), chase (swing | outside the zone), zone swing (swing | in the zone), whiff per swing, zone contact (1 - whiff | zone swing); and what PITCHERS do against them: share of pitches in the zone, pitch speed, share of fastballs.
Pitch-level K and BB per PA with the same design as a cross-check.
"""
import warnings

import numpy as np
import pandas as pd

from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")
MS = "Mississippi Mud Monsters"


def fac(x):
    return pd.factorize(pd.Series(np.where(pd.isna(x), "unk", x.astype(str))))[0]


def main():
    d = pd.read_parquet("2025_2026_pitches.parquet", columns=["game_id", "year", "stadium", "home_team", "batter_team", "batter_id", "pitcher_id", "swing", "whiff", "chase", "outside_zone", "pitch_call", "rel_speed", "pitch_type", "balls", "strikes", "k_or_bb", "is_terminal_pitch"])
    d = d[d["outside_zone"].notna() & d["swing"].notna()].reset_index(drop=True)
    d["ms_home"] = ((d["batter_team"] == MS) & (d["home_team"] == MS)).astype(float); d["home_other"] = ((d["batter_team"] == d["home_team"]) & (d["batter_team"] != MS)).astype(float)
    d["zone"] = 1 - d["outside_zone"]; d["fastball"] = d["pitch_type"].isin(["four_seam", "sinker", "cutter"]).astype(float)
    d["zone_swing"] = np.where(d["zone"] == 1, d["swing"], np.nan); d["zone_contact"] = np.where((d["zone"] == 1) & (d["swing"] == 1), 1 - d["whiff"], np.nan); d["chase_"] = np.where(d["outside_zone"] == 1, d["swing"], np.nan); d["whiff_sw"] = np.where(d["swing"] == 1, d["whiff"], np.nan)
    d["two_strike"] = (d["strikes"] == 2).astype(float); d["is_k"] = np.where(d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"]), (d["k_or_bb"] == "strikeout").astype(float), np.nan); d["is_bb"] = np.where(d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"]), (d["k_or_bb"] == "walk").astype(float), np.nan)
    outs = [("swing (all pitches)", "swing"), ("chase (swing | outside zone)", "chase_"), ("zone swing (swing | in zone)", "zone_swing"), ("whiff per swing", "whiff_sw"), ("zone contact (contact | zone swing)", "zone_contact"),
            ("PITCHERS: share of pitches in the zone", "zone"), ("PITCHERS: release speed (mph)", "rel_speed"), ("PITCHERS: fastball share", "fastball"), ("K per PA (terminal pitch)", "is_k"), ("BB per PA (terminal pitch)", "is_bb")]
    rows = []
    for lab, col in outs:
        s = d[d[col].notna()].reset_index(drop=True)
        g = [fac(s["year"]), fac(s["batter_id"]), fac(s["pitcher_id"]), fac(s["stadium"])]
        X = np.column_stack([s["ms_home"], s["home_other"]]); Xr = partial_out(X, g, iters=25); yr = partial_out(s[col].to_numpy(float)[:, None], g, iters=25)[:, 0]; b, se = cluster_ols(Xr, yr, pd.factorize(s["game_id"])[0])
        m = s["ms_home"] == 1; rows.append({"outcome": lab, "n": len(s), "MS hitters at home (mean)": s.loc[m, col].mean(), "MS hitters on the road": s.loc[(s["batter_team"] == MS) & ~m, col].mean(), "MS_HOME coef (FE)": b[0], "SE": se[0], "t": b[0] / se[0], "other teams' home coef": b[1]})
        print(rows[-1]["outcome"], round(b[0], 4), round(b[0] / se[0], 2), flush=True)
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
