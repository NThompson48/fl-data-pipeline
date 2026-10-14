"""
fl_park_xwoba_decomp.py -- Part 2 of fl_park_xwoba_surface.py, done correctly: split the Trustmark effect on wOBA on contact into CONTACT QUALITY (xwOBA) and the PARK EFFECT on that quality (wOBA - xwOBA), within hitter.
Design: ALL balls in play (so Mississippi pitchers are identified from their road games; a visitors-only subset makes the Trustmark dummy collinear with their pitcher effects), hitter + pitcher + year + [hitter x home] fixed effects,
regressor = Trustmark dummy for VISITING hitters (home hitters' home balls are absorbed by their own home effect), game-cluster SEs.  Overall, by batter side, by launch-angle band and by exit-speed band.
"""
import warnings

import numpy as np
import pandas as pd

from fl_park_profile_fe import cluster_ols, partial_out
from fl_park_xwoba_surface import EVB, LAB, load

warnings.filterwarnings("ignore")


def main():
    A = load().reset_index(drop=True); bid = A["batter_id"].astype(object).where(A["batter_id"].notna(), "unk").astype(str); pid = A["pitcher_id"].astype(object).where(A["pitcher_id"].notna(), "unk").astype(str)
    g = [pd.factorize(A["year"].astype(str))[0], pd.factorize(bid)[0], pd.factorize(pid)[0], pd.factorize(np.where(A["vis"] == 0, bid, "AWAY"))[0]]; cl = pd.factorize(A["game_id"])[0]; tmv = (A["tm"] * A["vis"]).to_numpy(float)

    def eff(mask, label):
        sub = np.where(mask)[0]; As = A.iloc[sub]; gs = [pd.factorize(x[sub])[0] for x in g]; out = {"group": label, "visiting TM balls": int(tmv[sub].sum())}; tr = partial_out(tmv[sub][:, None], gs, iters=30)[:, 0]
        for k, c in (("wOBAcon", "woba"), ("xwOBA (quality)", "xwoba"), ("residual (park)", "res")):
            yr = partial_out(As[c].to_numpy(float)[:, None], gs, iters=30)[:, 0]; bb, se = cluster_ols(tr[:, None], yr, cl[sub]); out[k] = f"{bb[0]:+.4f} (t {bb[0] / se[0]:+.1f})"
        return out
    rows = [eff(np.ones(len(A), bool), "all hitters"), eff((A["left"] == 1).to_numpy(), "LHH"), eff((A["left"] == 0).to_numpy(), "RHH")]
    for lo, hi, ll in LAB:
        rows.append(eff(((A["angle"] >= lo) & (A["angle"] < hi)).to_numpy(), f"launch angle {ll}"))
    for elo, ehi, el in EVB:
        rows.append(eff(((A["exit_speed"] >= elo) & (A["exit_speed"] < ehi)).to_numpy(), f"exit speed {el}"))
    print("WITHIN-HITTER TRUSTMARK EFFECT per ball in play for VISITING hitters = contact quality (xwOBA) + park effect on that quality (residual)\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
