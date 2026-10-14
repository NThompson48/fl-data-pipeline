"""
fl_park_airball_mechanism.py -- WHY do air-ball hitters gain at Trustmark?  (visiting hitters only, pooled 2025-26; Trustmark x Air% interaction, hitter + pitcher + year + [hitter x home] + park effects, cluster-robust SEs by game)
(1) PER PA: coefficient per +1 SD of the shrunk Air% profile on HR, K, BB(+HBP), singles, 2B+3B, balls in play, and on wOBA, with each event's contribution to the wOBA effect (coefficient x its FL linear weight):
    which events carry the +.03 wOBA?
(2) PER BALL IN PLAY (non-HR): hit rate (BABIP), hit rate on AIR balls (launch angle >= 10) and on GROUND balls (< 10), share of balls in play that are air balls, extra-base-hit share of hits.
Air% profile = the hitter's shrunk air-ball share on his tracked balls (z-scored, fl_park_profile_final.py); power (HR/PA profile) is included as a second interaction so Air% is read holding power fixed.
"""
import warnings

import numpy as np
import pandas as pd

from fl_park_final_visitors import hitter_frame
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")
W = {"single": 0.945, "xbh": 1.28, "hr": 1.863, "bb": 0.82}      # FL linear weights (mean wOBA by event: single .945, double 1.251 / triple 1.534 (xbh ~1.28), HR 1.863, walk .811 / HBP .841)


def fit(est, y, label_rows, groups):
    tmv = est["tm"].to_numpy(float) * (est["b_home"] == 0).to_numpy(float)
    X = np.column_stack([tmv * est["z_air"], tmv * est["z_power"]]); Xr = partial_out(X, groups, iters=30); yr = partial_out(np.asarray(y, float)[:, None], groups, iters=30)[:, 0]
    return cluster_ols(Xr, yr, pd.factorize(est["game_id"])[0])


def main():
    est, ml = hitter_frame(); kind = est["kind"].astype(str); est["is_hr"] = (kind == "home_run").astype(float); est["is_k"] = (kind == "strikeout").astype(float); est["is_bb"] = kind.isin(["walk", "hbp"]).astype(float)
    est["is_1b"] = (kind == "single").astype(float); est["is_xbh"] = kind.isin(["double", "triple"]).astype(float); est["is_bip"] = kind.isin(["single", "double", "triple", "out", "fielders_choice", "error"]).astype(float)
    hh = np.where(est["b_home"] == 1, est["name_clean"], "AWAY")
    groups = lambda d: [pd.factorize(d["year"])[0], pd.factorize(d["name_clean"])[0], pd.factorize(d["pitcher"])[0], pd.factorize(np.where(d["b_home"] == 1, d["name_clean"], "AWAY"))[0], pd.factorize(d["park"])[0]]
    g = groups(est); rows = []
    for lab, col, wt in (("HR/PA", "is_hr", W["hr"]), ("K%", "is_k", 0.0), ("BB+HBP %", "is_bb", W["bb"]), ("single/PA", "is_1b", W["single"]), ("2B+3B/PA", "is_xbh", W["xbh"]), ("balls in play / PA", "is_bip", 0.0), ("wOBA", "woba", 1.0)):
        b, se = fit(est, est[col].to_numpy(float), None, g); rows.append({"outcome (per PA)": lab, "Air% per SD": f"{b[0]:+.4f} ({se[0]:.4f})", "t": b[0] / se[0], "power per SD": f"{b[1]:+.4f} ({se[1]:.4f})", "contribution of Air% to wOBA (coef x weight)": b[0] * wt if col != "woba" else b[0]})
    print(f"(1) PER PLATE APPEARANCE, visiting hitters (n = {len(est):,} PA, {int(((est['tm'] == 1) & (est['b_home'] == 0)).sum()):,} visitor PA at Trustmark); coefficient (cluster-robust SE) per +1 SD\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    # (2) per ball in play
    bp = est[(est["is_bip"] == 1)].copy(); bp["hit"] = bp["kind"].astype(str).isin(["single", "double", "triple"]).astype(float); bp["air"] = (bp["angle"] >= 10).astype(float); bp = bp[bp["angle"].notna()].reset_index(drop=True)
    gb = groups(bp); rows = []
    for lab, sub, col in (("hit rate on balls in play (BABIP)", bp, "hit"), ("hit rate on AIR balls (>= 10 deg)", bp[bp["air"] == 1], "hit"), ("hit rate on GROUND balls (< 10 deg)", bp[bp["air"] == 0], "hit"), ("share of balls in play that are air balls", bp, "air")):
        sub = sub.reset_index(drop=True); b, se = fit(sub, sub[col].to_numpy(float), None, groups(sub)); rows.append({"outcome (per ball in play)": lab, "n": len(sub), "league mean": sub[col].mean(), "Air% per SD": f"{b[0]:+.4f} ({se[0]:.4f})", "t": b[0] / se[0], "power per SD": f"{b[1]:+.4f} ({se[1]:.4f})"})
    hits = bp[bp["hit"] == 1].reset_index(drop=True); hits["xbh"] = hits["kind"].astype(str).isin(["double", "triple"]).astype(float); b, se = fit(hits, hits["xbh"].to_numpy(float), None, groups(hits)); rows.append({"outcome (per ball in play)": "extra-base share of hits", "n": len(hits), "league mean": hits["xbh"].mean(), "Air% per SD": f"{b[0]:+.4f} ({se[0]:.4f})", "t": b[0] / se[0], "power per SD": f"{b[1]:+.4f} ({se[1]:.4f})"})
    print("\n(2) PER BALL IN PLAY (non-HR), same design\n" + pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
