"""
fl_pitcher_ros_backtest.py -- walk-forward backtest of the pitcher ROS K% / BB% update (2025 and 2026, cutoffs at 15 / 30 / 45 / 60 / 75% of the season).

Constants (prior-season weight d, n0) are fit only on transitions whose TARGET year is before the season being scored (2026 uses 2022->23 ... 2024->25; 2025 uses 2022->23, 2023->24),
standard inverse-variance weights, then frozen.  Models: 0 league now; 1 preseason only (history); 2 YTD face value; 3 YTD naive shrinkage (n0, ignores history); 4 ROS update (in-season BF = newest season).
Usage:  python fl_pitcher_ros_backtest.py
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_pitcher_ros as pr

warnings.filterwarnings("ignore")
rng = np.random.default_rng(14)
CUTS = (0.15, 0.30, 0.45, 0.60, 0.75)


def wrmse(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def main():
    consts = {2025: pr.fit_constants(2024), 2026: pr.fit_constants(2025)}      # constants never see the season being scored
    for y, c in consts.items():
        print(f"constants used to score {y}: " + "; ".join(f"{s}: d={v['delta']}, n0={v['n0']:.0f} (LOTO RMSE {v['rmse']:.4f} vs league {v['rmse_league']:.4f}, last-year {v['rmse_last']:.4f}; n={v['n']}, y/y {v['y/y']:.2f})" for s, v in c.items()))
    T = pr.Tables()
    recs = {s: [] for s in ("k", "bbr")}
    for year in (2025, 2026):
        t0, t1 = T.season_bounds(year)
        for c in CUTS:
            asof = t0 + (t1 - t0) * c
            f, lg = pr.features(T, year, asof)
            ros = pr.predict(f, lg, consts[year]); pre = pr.predict(f, lg, consts[year], use_ytd=False)
            for s, ycol, cnt in (("k", "k_ros", "k"), ("bbr", "bb_ros", "bb")):
                n0 = consts[year][s]["n0"]; den = f["n"]; x = np.where(den > 0, f[cnt] / den.replace(0, np.nan), lg[s])
                d = pd.DataFrame({"y": f[ycol].to_numpy(), "n_ros": f["n_ros"].to_numpy(), "year": year, "cut": c, "pitcher": (f.index + "|" + str(year)).to_numpy(), "ytd_bf": den.to_numpy(),
                                  "0 league average now": lg[s], "1 preseason only": pre[s].to_numpy(), "2 YTD face value": x, "3 YTD naive shrinkage": lg[s] + den.to_numpy() * (x - lg[s]) / (den.to_numpy() + n0),
                                  "4 ROS update": ros[s].to_numpy()})
                recs[s].append(d)
    for s, lab in (("k", "K%"), ("bbr", "BB%")):
        d = pd.concat(recs[s], ignore_index=True); models = [m for m in d.columns if m[0].isdigit()]
        w, a, b = ew.weights_from_base(d["y"].to_numpy(), d["4 ROS update"].to_numpy(), d["n_ros"].to_numpy()); d["w"] = w
        print(f"\n{'=' * 90}\n{lab}: n = {len(d)} pitcher-cutoffs (>= 30 BF after the cutoff); law a={a:.5f}, b={b:.4f}")
        rows = []
        for m in models:
            r = {"model": m, "RMSE all": wrmse(d["y"].to_numpy(), d[m].to_numpy(), w), "mean err": float(np.average(d["y"] - d[m], weights=w))}
            for c in CUTS:
                mk = (d["cut"] == c).to_numpy(); r[f"{int(c*100)}%"] = wrmse(d["y"].to_numpy()[mk], d[m].to_numpy()[mk], w[mk])
            for yy in (2025, 2026):
                mk = (d["year"] == yy).to_numpy(); r[str(yy)] = wrmse(d["y"].to_numpy()[mk], d[m].to_numpy()[mk], w[mk])
            rows.append(r)
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        ug = pd.Series(d["pitcher"]).unique(); by = pd.Series(np.arange(len(d))).groupby(d["pitcher"].to_numpy()).apply(np.array)
        print("   paired bootstrap of RMSE differences (90% interval, pitcher clusters):")
        for A, B in (("4 ROS update", "1 preseason only"), ("4 ROS update", "3 YTD naive shrinkage")):
            dd = []
            for _ in range(500):
                ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(wrmse(d["y"].to_numpy()[ii], d[A].to_numpy()[ii], w[ii]) - wrmse(d["y"].to_numpy()[ii], d[B].to_numpy()[ii], w[ii]))
            print(f"     {A} minus {B}: {np.mean(dd):+.5f}  [{np.percentile(dd,5):+.5f}, {np.percentile(dd,95):+.5f}]")
        d["bk"] = pd.cut(d["ytd_bf"], [-1, 29, 99, 10 ** 5], labels=["< 30 BF YTD", "30-99", "100+"])
        t = d.groupby("bk", observed=True).apply(lambda g: pd.Series({"n": len(g), **{m[:22]: wrmse(g["y"].to_numpy(), g[m].to_numpy(), g["w"].to_numpy()) for m in ("1 preseason only", "3 YTD naive shrinkage", "4 ROS update")}}))
        print("   by the pitcher's BF so far this season (RMSE):"); print(t.round(4).to_string())


if __name__ == "__main__":
    main()
