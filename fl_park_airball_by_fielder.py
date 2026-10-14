"""
fl_park_airball_by_fielder.py -- WHERE does the Trustmark x Air% hit-rate gain (fl_park_airball_mechanism.py) live: outfield-fielded balls or infield / other balls?  (follow-up to the OAA finding, user 2026-10-09)
All non-HR balls in play (terminal pitches with exit speed), split by the fielder group (OF = fielded_by_pos in LF/CF/RF, which is exactly where OAA exists; other = infield / unknown).
Cross-fitted traits-only expected-hit model (exit speed, launch angle, direction, distance, hang time, batter side) fitted separately within each group; residual = hit - expected.
For VISITING hitters: slope of the residual on the shrunk Air% profile (per +1 SD) at Trustmark vs the other parks, game-cluster bootstrap; also the plain Trustmark park residual by group (visiting + home fielders / hitters both).
Also the raw hit rate on balls with launch angle >= 10 (the earlier 'air-ball hit rate' definition) by group, so the earlier +.0534/SD can be decomposed: share of those balls in each group x the group's slope.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

import fl_hitter_ros as hr

warnings.filterwarnings("ignore")
TM = "TrustmarkPark"


def load():
    d = pd.read_parquet("2025_2026_pitches.parquet")
    b = d[d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"]) & d["exit_speed"].notna() & d["play_result"].isin(["out", "sacrifice", "single", "double", "triple", "error", "fielders_choice"])].copy().reset_index(drop=True)
    b["hit"] = b["play_result"].isin(["single", "double", "triple"]).astype(float); b["of"] = b["fielded_by_pos"].isin(["LF", "CF", "RF"]).astype(int); b["tm"] = (b["stadium"] == TM).astype(int); b["vis"] = (b["batter_team"] != b["home_team"]).astype(int)
    pa = hr.Tables().pa[["game_id", "pa_id", "name_clean"]].drop_duplicates(["game_id", "pa_id"]); b = b.merge(pa, on=["game_id", "pa_id"], how="left")
    prof = pd.read_parquet("output/hitter_park_profile.parquet"); b["z_air"] = b["name_clean"].map(prof["z_air"]); return b


def cross_fit(g):
    X = pd.DataFrame({"ev": g["exit_speed"], "angle": g["angle"], "dir": g["direction"], "dist": g["distance"], "hang": g["hang_time"], "side": (g["batter_side"].astype(str) == "Left").astype(int)}); y = g["hit"].to_numpy(); p = np.zeros(len(g))
    for tr, te in GroupKFold(5).split(X, y, groups=g["game_id"]):
        p[te] = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, min_samples_leaf=40).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return p


def slope_boot(x, y, games, B=400, seed=5):
    sl = np.polyfit(x, y, 1)[0]; by = pd.Series(np.arange(len(x))).groupby(games).apply(np.array); ug = by.index.to_numpy(); rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); bs.append(np.polyfit(x[ii], y[ii], 1)[0])
    return sl, float(np.std(bs))


def main():
    b = load(); b = b[b["play_result"] != "home_run"].reset_index(drop=True); b["resid"] = np.nan
    for grp in (1, 0):
        m = b["of"] == grp; b.loc[m, "resid"] = b.loc[m, "hit"].to_numpy() - cross_fit(b[m].reset_index(drop=True))
    print(f"{len(b):,} balls in play (non-HR): outfield-fielded {int(b['of'].sum()):,}, other {int((1 - b['of']).sum()):,}")
    rows = []
    for lab, grp in (("outfield-fielded", 1), ("infield / other", 0)):
        g = b[b["of"] == grp]; tm = g[g["tm"] == 1]; ot = g[g["tm"] == 0]
        rows.append({"group": lab, "n": len(g), "n at Trustmark": len(tm), "hit rate": g["hit"].mean(), "Trustmark hit - expected": tm["resid"].mean(), "other parks": ot["resid"].mean(), "Trustmark minus other": tm["resid"].mean() - ot["resid"].mean()})
    print("\nPARK RESIDUAL by fielder group (all hitters)\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    v = b[(b["vis"] == 1) & b["z_air"].notna()]; rows = []
    for lab, grp in (("outfield-fielded", 1), ("infield / other", 0), ("all non-HR balls in play", None)):
        g = v if grp is None else v[v["of"] == grp]
        if grp is None:
            g = g.copy(); y = g["hit"].to_numpy() - g.groupby("of")["hit"].transform("mean").to_numpy()      # hit rate net of group mean (no trait model across groups)
            g["resid2"] = g["resid"]
        for park, mm in (("Trustmark", g["tm"] == 1), ("other parks", g["tm"] == 0)):
            s, se = slope_boot(g.loc[mm, "z_air"].to_numpy(), g.loc[mm, "resid"].to_numpy(), g.loc[mm, "game_id"].to_numpy()); rows.append({"group": lab, "park": park, "n": int(mm.sum()), "slope per +1 SD Air% (hit - expected)": s, "SE": se})
    r = pd.DataFrame(rows); print("\nVISITING hitters: residual hit rate on Air% profile\n" + r.round(4).to_string(index=False))
    for lab in ("outfield-fielded", "infield / other", "all non-HR balls in play"):
        a = r[(r["group"] == lab) & (r["park"] == "Trustmark")].iloc[0]; o = r[(r["group"] == lab) & (r["park"] == "other parks")].iloc[0]; d = a.iloc[3] - o.iloc[3]; se = np.hypot(a["SE"], o["SE"]); print(f"  {lab}: Trustmark minus other {d:+.4f} (SE {se:.4f}, t {d / se:+.1f})")
    # raw air-ball (angle >= 10) hit rate: share by group at Trustmark
    air = v[(v["angle"] >= 10)]; print("\nballs with launch angle >= 10 (visiting hitters): share outfield-fielded: Trustmark %.2f, other parks %.2f; hit rate OF %.3f / other %.3f" % (air[air.tm == 1]["of"].mean(), air[air.tm == 0]["of"].mean(), air[air.of == 1]["hit"].mean(), air[air.of == 0]["hit"].mean()))


if __name__ == "__main__":
    main()
