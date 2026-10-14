"""
fl_park_airball_contact.py -- does the Trustmark x Air% effect work through the QUALITY / SHAPE of contact rather than through fielding?  (follows fl_park_airball_by_fielder.py: with batted-ball traits held fixed the hit rate shows no Trustmark x Air% effect)
Same visitors-only design as fl_park_final_visitors.py (year + hitter + pitcher + [hitter x home] + park fixed effects, game-cluster SEs), interaction = Trustmark (visiting hitter) x z_air, z_power held in the model.
Outcomes (per ball in play / per PA, contact rows only where stated): exit speed, launch angle, share of balls at >= 10 deg (air), distance (HR included), xwOBA on contact, wOBA on contact, wOBA (all PA, the production target).
"""
import warnings

import numpy as np
import pandas as pd

from fl_park_final_visitors import hitter_frame
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")


def main():
    est, ml = hitter_frame(); print(est.columns.tolist()[:40])
    est["air"] = np.where(est["angle"].notna(), (est["angle"] >= 10).astype(float), np.nan); vis = (est["b_home"] == 0).to_numpy(float); tmv = est["tm"].to_numpy(float) * vis; rows = []
    for lab, col, sub in (("wOBA (all PA)", "woba", None), ("wOBA on contact", "woba", "bip"), ("xwOBA on contact", "xwoba", "bip"), ("exit speed (mph)", "exit_speed", "bip"), ("launch angle (deg)", "angle", "bip"), ("air-ball share (>= 10 deg)", "air", "bip")):
        d = est if sub is None else est[est["exit_speed"].notna() & est[col].notna()]; d = d.reset_index(drop=True); v = (d["b_home"] == 0).to_numpy(float); t = d["tm"].to_numpy(float) * v
        g = [pd.factorize(d["year"])[0], pd.factorize(d["name_clean"])[0], pd.factorize(d["pitcher"])[0], pd.factorize(np.where(d["b_home"] == 1, d["name_clean"], "AWAY"))[0], pd.factorize(d["park"])[0]]
        X = np.column_stack([t * d["z_air"], t * d["z_power"]]); Xr = partial_out(X, g, iters=30); yr = partial_out(d[col].to_numpy(float)[:, None], g, iters=30)[:, 0]; b, se = cluster_ols(Xr, yr, pd.factorize(d["game_id"])[0])
        rows.append({"outcome": lab, "n": len(d), "Air% x Trustmark per SD": f"{b[0]:+.4f} ({se[0]:.4f})", "t": b[0] / se[0], "power x Trustmark per SD": f"{b[1]:+.4f} ({se[1]:.4f})", "t power": b[1] / se[1]})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
