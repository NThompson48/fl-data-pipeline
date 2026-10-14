"""
fl_park_ev_tracking_check.py -- is exit speed over-read on airborne balls at Trustmark (a tracking artefact) or is the ball really hit harder?  (user 2026-10-09 OAA observation + fl_park_ev_by_angle.py)
Physical consistency: for a real exit speed and launch angle the ball's distance / hang time are set by physics + air + wind.  If Trustmark's radar over-reads EV on airborne balls, then GIVEN the recorded EV and angle the ball travels / hangs LESS than at other parks.
Cross-fitted gradient boosting (5-fold by game) of distance and hang time on exit speed, launch angle (airborne 10-45 deg balls; includes HR) WITHOUT park information; residual by park.  Also exit speed itself by park for airborne balls hit by HITTERS WHO PLAYED IN MANY PARKS
(hitter fixed effect, all hitters: home + visitors, to see whether the Trustmark deviation is a park-level property).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


def main():
    d = pd.read_parquet("2025_2026_pitches.parquet", columns=["game_id", "year", "stadium", "batter", "batter_side", "is_terminal_pitch", "exit_speed", "angle", "direction", "distance", "hang_time", "play_result", "temp", "wind"])
    d = d[d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"]) & d["exit_speed"].notna() & d["angle"].between(10, 45) & d["distance"].notna() & d["hang_time"].notna()].reset_index(drop=True)
    print(f"{len(d):,} airborne (10-45 deg) tracked balls")
    X = pd.DataFrame({"ev": d["exit_speed"], "angle": d["angle"], "dir": d["direction"]})
    for tgt in ("distance", "hang_time"):
        y = d[tgt].to_numpy(float); p = np.zeros(len(d))
        for tr, te in GroupKFold(5).split(X, y, groups=d["game_id"]):
            p[te] = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, min_samples_leaf=40).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        d[f"{tgt}_resid"] = y - p; print(f"  {tgt}: cross-fitted R^2 {1 - np.var(y - p) / np.var(y):.3f}")
    t = d.groupby("stadium").agg(n=("exit_speed", "size"), ev=("exit_speed", "mean"), dist_resid_ft=("distance_resid", "mean"), hang_resid_s=("hang_time_resid", "mean"), temp=("temp", "mean")).round(3).sort_values("dist_resid_ft")
    print("\nBY PARK: airborne balls, distance / hang time relative to what EV + angle + spray predict (no park info in the model)\n" + t.to_string())
    # same, hitter-fixed-effect exit speed by park (demean EV by hitter, then average by park)
    d["ev_dm"] = d["exit_speed"] - d.groupby("batter")["exit_speed"].transform("mean"); print("\nEV relative to the hitter's own mean (airborne balls), by park:"); print(d.groupby("stadium")["ev_dm"].mean().round(3).sort_values().to_string())
    tm = d["stadium"] == "TrustmarkPark"; print(f"\nTrustmark vs other parks: distance resid {d.loc[tm, 'distance_resid'].mean():+.2f} ft (other {d.loc[~tm, 'distance_resid'].mean():+.2f}), hang resid {d.loc[tm, 'hang_time_resid'].mean():+.3f}s (other {d.loc[~tm, 'hang_time_resid'].mean():+.3f}s), EV vs own mean {d.loc[tm, 'ev_dm'].mean():+.2f} mph")


if __name__ == "__main__":
    main()
