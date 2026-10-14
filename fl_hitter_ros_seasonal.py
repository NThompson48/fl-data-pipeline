"""
fl_hitter_ros_seasonal.py -- the league level drifts within a season (wOBA peaks in June and falls ~.010-.013 by 60-75% of the season; K% and BB% run higher early).  Does a seasonal
level correction -- (league ROS level - league YTD level) at the same cutoff fraction in the OTHER season -- improve the ROS forecasts?  Forward only: 2025's pattern applied to 2026
(the reverse is shown as a check).  The correction is added to the model's prediction; the weights and bootstrap are as in fl_hitter_ros_backtest.py.

Usage:  python fl_hitter_ros_seasonal.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_ros as ros
import fl_hitter_ros_backtest as rb

warnings.filterwarnings("ignore")
rng = np.random.default_rng(5)


def deltas(T):
    out = {}
    for y in (2025, 2026):
        t0, t1 = T.season_bounds(y); d = T.pa[T.pa["year"] == y]
        for c in rb.CUTS:
            a = t0 + (t1 - t0) * c; ytd, r = d[d["date"] < a], d[d["date"] >= a]
            out[(y, c)] = {"woba": r["woba"].mean() - ytd["woba"].mean(), "k": r["k"].mean() - ytd["k"].mean(), "bbr": r["bb"].mean() - ytd["bb"].mean()}
    return out


def main():
    T = ros.Tables(); frames = rb.build(T); D = rb.collect(frames); dl = deltas(T)
    print("seasonal level change (ROS league - YTD league) by cutoff:")
    print(pd.DataFrame({f"{y} {int(c*100)}%": dl[(y, c)] for y in (2025, 2026) for c in rb.CUTS}).T.round(4).to_string())
    for tag, train in (("FORWARD 2025 -> 2026", 2025), ("reverse check 2026 -> 2025", 2026)):
        test = 2026 if train == 2025 else 2025
        print(f"\n{tag}")
        for s in rb.STATS:
            d = D[s]; d = d[d["year"] == test].copy()
            w, _, _ = ew.weights_from_base(d["y"].to_numpy(), d["4 ROS update (no Trackman)"].to_numpy(), d["n_ros"].to_numpy())
            adj = d["cut"].map(lambda c: dl[(train, c)][s]).to_numpy()
            y = d["y"].to_numpy(); hit = d["hitter"].to_numpy()
            ug = pd.Series(hit).unique(); by = pd.Series(np.arange(len(d))).groupby(hit).apply(np.array)
            rows = []
            for m in ("1 preseason only", "4 ROS update (no Trackman)", "5 + pitch-level terms"):
                p0 = d[m].to_numpy(); p1 = p0 + adj
                dd = []
                for _ in range(500):
                    ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(rb.wrmse(y[ii], p1[ii], w[ii]) - rb.wrmse(y[ii], p0[ii], w[ii]))
                rows.append({"model": m, "RMSE": rb.wrmse(y, p0, w), "RMSE + seasonal level": rb.wrmse(y, p1, w), "change": np.mean(dd), "90% interval": f"[{np.percentile(dd,5):+.5f}, {np.percentile(dd,95):+.5f}]",
                             "mean err before": float(np.average(y - p0, weights=w)), "after": float(np.average(y - p1, weights=w))})
            print(f"  {s}:"); print(pd.DataFrame(rows).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
