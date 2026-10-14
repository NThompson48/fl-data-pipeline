"""
fl_park_ev_by_angle.py -- is the Trustmark exit-speed gain for air-ball hitters a TRACKING artefact (exit speed read differently on fly balls at Trustmark) or hitter-specific?  (follows fl_park_airball_contact.py)
(1) Trustmark effect on exit speed by launch-angle bin (ground < 10, line drive 10-25, fly 25-40, pop-up > 40), VISITING hitters, hitter + pitcher + year + [hitter x home] + park fixed effects, game-cluster SEs: a tracking artefact on fly balls would lift every hitter's
    fly balls; a hitter-specific effect would not.
(2) Same, with the interaction by z_air inside each bin.
(3) Same two cuts for HOME hitters' balls... not applicable to the visitors-only design; instead the Trustmark effect on EV by bin for HOME PITCHERS' side is shown as the exit speed allowed by home pitchers vs visiting pitchers at Trustmark (pitcher-side check, needs pitcher FE): skipped; see (1)-(2).
"""
import warnings

import numpy as np
import pandas as pd

from fl_park_final_visitors import hitter_frame
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")
BINS = [(-90, 10, "ground (< 10)"), (10, 25, "line drive (10-25)"), (25, 40, "fly ball (25-40)"), (40, 91, "pop-up (> 40)")]


def main():
    est, ml = hitter_frame(); est = est[est["exit_speed"].notna() & est["angle"].notna()].reset_index(drop=True)
    g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pitcher"])[0], pd.factorize(np.where(est["b_home"] == 1, est["name_clean"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    v = (est["b_home"] == 0).to_numpy(float); t = est["tm"].to_numpy(float) * v; cl = pd.factorize(est["game_id"])[0]; y = est["exit_speed"].to_numpy(float); yr = partial_out(y[:, None], g, iters=30)[:, 0]; rows = []
    cols = [t * ((est["angle"] >= lo) & (est["angle"] < hi)).to_numpy(float) for lo, hi, _ in BINS]; Xr = partial_out(np.column_stack(cols), g, iters=30); b, se = cluster_ols(Xr, yr, cl)
    colz = [c * est["z_air"].to_numpy(float) for c in cols]; Xz = partial_out(np.column_stack(colz), g, iters=30); bz, sez = cluster_ols(Xz, yr, cl)
    for i, (lo, hi, lab) in enumerate(BINS):
        m = ((est["angle"] >= lo) & (est["angle"] < hi)).to_numpy(); rows.append({"launch angle": lab, "n": int(m.sum()), "n visitor at Trustmark": int((m & (t == 1)).sum()), "Trustmark main effect on EV (mph)": f"{b[i]:+.2f} ({se[i]:.2f})", "t": b[i] / se[i],
                                                                              "x Air% per SD": f"{bz[i]:+.2f} ({sez[i]:.2f})", "t Air%": bz[i] / sez[i]})
    print("(1) Trustmark effect on exit speed by launch-angle bin, visiting hitters (note: main effects here are relative to the park fixed effect, so only DIFFERENCES between bins are identified)\n" + pd.DataFrame(rows).round(2).to_string(index=False))
    print("\n(2) joint: Trustmark x Air% interaction on EV within each bin is the last columns above (per +1 SD Air%).")


if __name__ == "__main__":
    main()
