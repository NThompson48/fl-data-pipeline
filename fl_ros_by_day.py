"""
fl_ros_by_day.py -- ROS models tested at different DAYS INTO THE SEASON (as-of day 7, 14, ..., 105 after the first game; target = the rest of the season).

1. Error and gains by as-of day (preseason / ROS update / update + pitch-level terms), per season.
2. Should the parameters change with the day?  In-season weight nu, preseason-prior scale n0_scale, pitch-level-term scale are tuned separately in four day bins
   (<= 28, 35-56, 63-84, >= 91 days) on one season and scored on the other, against the single production setting.  Standard inverse-variance weights, frozen from the production predictions.
Hitters: wOBA, K%, BB%.  Pitchers: K%, BB% (update only; their stuff/command terms are tested by cutoff in fl_pitcher_ros_stuff.py).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_ros as hr
import fl_pitcher_ros as pr

warnings.filterwarnings("ignore")
rng = np.random.default_rng(77)
DAYS = list(range(7, 106, 7))
BINS = {"<=28": (0, 28), "35-56": (35, 56), "63-84": (63, 84), ">=91": (91, 999)}
GRID = {"nu": (0.5, 0.75, 1.0, 1.5, 2.0), "n0_scale": (0.5, 0.75, 1.0, 1.5, 2.0), "tscale": (0.0, 0.5, 1.0, 1.5, 2.0)}


def wr(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def hitter_table(T):
    out = []
    for yr in (2025, 2026):
        t0, t1 = T.season_bounds(yr)
        for d in DAYS:
            f, lg = hr.features(T, yr, t0 + pd.Timedelta(days=d), min_ros_pa=30)
            if len(f) < 40:
                continue
            base = dict(pre=hr.predict(f, lg, trackman=False, use_ytd=False), upd=hr.predict(f, lg, trackman=False))
            var = {"prod": hr.predict(f, lg)}
            for nu in GRID["nu"]:
                var[f"nu={nu}"] = hr.predict(f, lg, nu=nu)
            for s in GRID["n0_scale"]:
                var[f"n0_scale={s}"] = hr.predict(f, lg, n0_scale=s)
            for s in GRID["tscale"]:
                var[f"tscale={s}"] = hr.predict(f, lg, params={"trackman_scale": s})
            for stat, (ycol, ucol, fcol) in {"woba": ("woba_ros", "blend", "woba_final"), "k": ("k_ros", "k", "k_final"), "bbr": ("bb_ros", "bbr", "bbr_final")}.items():
                df = pd.DataFrame({"y": f[ycol].to_numpy(), "n": f["n_ros"].to_numpy(), "id": (f.index + "|" + str(yr)).to_numpy(), "year": yr, "day": d, "stat": stat,
                                   "pre": base["pre"][ucol].to_numpy(), "upd": base["upd"][ucol].to_numpy()})
                for k, v in var.items():
                    df[k] = v[fcol].to_numpy()
                out.append(df)
    return pd.concat(out, ignore_index=True)


def pitcher_table(T, consts):
    out = []
    for yr in (2025, 2026):
        t0, t1 = T.season_bounds(yr)
        for d in DAYS:
            f, lg = pr.features(T, yr, t0 + pd.Timedelta(days=d))
            if len(f) < 40:
                continue
            pre = pr.predict(f, lg, consts[yr], use_ytd=False); prod = pr.predict(f, lg, consts[yr])
            var = {f"nu={nu}": pr.predict(f, lg, consts[yr], nu=nu) for nu in GRID["nu"]}
            var.update({f"n0_scale={s}": pr.predict(f, lg, consts[yr], n0_scale=s) for s in GRID["n0_scale"]})
            for stat, (ycol, key) in {"k": ("k_ros", "k"), "bb": ("bb_ros", "bbr")}.items():
                df = pd.DataFrame({"y": f[ycol].to_numpy(), "n": f["n_ros"].to_numpy(), "id": (f.index + "|" + str(yr)).to_numpy(), "year": yr, "day": d, "stat": stat,
                                   "pre": pre[key].to_numpy(), "upd": prod[key].to_numpy(), "prod": prod[key].to_numpy()})
                for k, v in var.items():
                    df[k] = v[key].to_numpy()
                out.append(df)
    return pd.concat(out, ignore_index=True)


def by_day(tab, label, with_terms):
    print(f"\n{label}: error by as-of day (weighted RMSE; weights frozen from the production predictions)")
    for stat, d in tab.groupby("stat"):
        w = ew.weights_from_base(d["y"].to_numpy(), d["prod"].to_numpy(), d["n"].to_numpy())[0]; d = d.assign(w=w)
        rows = []
        for (yr, day), g in d.groupby(["year", "day"]):
            r = {"year": yr, "day": day, "n": len(g), "mean ROS n": g["n"].mean(), "preseason": wr(g["y"], g["pre"], g["w"]), "update": wr(g["y"], g["upd"], g["w"]),
                 "update gain": wr(g["y"], g["upd"], g["w"]) - wr(g["y"], g["pre"], g["w"])}
            if with_terms:
                r["full"] = wr(g["y"], g["prod"], g["w"]); r["terms gain"] = r["full"] - r["update"]
            rows.append(r)
        t = pd.DataFrame(rows)
        piv = t.pivot(index="day", columns="year", values=["n", "preseason", "update gain"] + (["terms gain"] if with_terms else []))
        print(f"  {stat}:"); print(piv.round(5).to_string())


def tune_bins(tab, label, params):
    print(f"\n{label}: do the best parameters change with the day?  (best value per day-bin chosen on one season, scored on the other; change vs the single production setting)")
    for stat, d in tab.groupby("stat"):
        w = ew.weights_from_base(d["y"].to_numpy(), d["prod"].to_numpy(), d["n"].to_numpy())[0]; d = d.assign(w=w)
        for p in params:
            cols = [f"{p}={v}" for v in GRID[p]]; rows = []
            tot = {"sel": [], "prod": [], "ids": [], "y": [], "w": []}
            for tr, te in ((2025, 2026), (2026, 2025)):
                for bn, (lo, hi) in BINS.items():
                    mtr = (d["year"] == tr) & d["day"].between(lo, hi); mte = (d["year"] == te) & d["day"].between(lo, hi)
                    if mtr.sum() < 50 or mte.sum() < 50:
                        continue
                    best = min(cols, key=lambda c: wr(d.loc[mtr, "y"], d.loc[mtr, c], d.loc[mtr, "w"]))
                    e_best = wr(d.loc[mte, "y"], d.loc[mte, best], d.loc[mte, "w"]); e_prod = wr(d.loc[mte, "y"], d.loc[mte, "prod"], d.loc[mte, "w"])
                    own = min(cols, key=lambda c: wr(d.loc[mte, "y"], d.loc[mte, c], d.loc[mte, "w"]))
                    rows.append({"train->test": f"{tr}->{te}", "bin": bn, "best on train": best.split("=")[1], "best on test itself": own.split("=")[1], "test change vs production": e_best - e_prod, "n test": int(mte.sum())})
                    tot["sel"].append(d.loc[mte, best].to_numpy()); tot["prod"].append(d.loc[mte, "prod"].to_numpy()); tot["ids"].append(d.loc[mte, "id"].to_numpy())
                    tot["y"].append(d.loc[mte, "y"].to_numpy()); tot["w"].append(d.loc[mte, "w"].to_numpy())
            if not rows:
                continue
            y = np.concatenate(tot["y"]); ww = np.concatenate(tot["w"]); ps = np.concatenate(tot["sel"]); pp = np.concatenate(tot["prod"]); ids = np.concatenate(tot["ids"])
            by = pd.Series(np.arange(len(y))).groupby(ids).apply(np.array); ug = by.index.to_numpy(); dd = []
            for _ in range(400):
                ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(wr(y[ii], ps[ii], ww[ii]) - wr(y[ii], pp[ii], ww[ii]))
            dd = np.array(dd)
            print(f"\n  {stat} / {p}:"); print(pd.DataFrame(rows).round(5).to_string(index=False))
            print(f"    all bins, selected-by-bin vs production: {np.mean(dd):+.5f} [{np.percentile(dd, 5):+.5f}, {np.percentile(dd, 95):+.5f}]")


def main():
    T = hr.Tables(); ht = hitter_table(T)
    by_day(ht, "HITTERS", True); tune_bins(ht, "HITTERS", ("nu", "n0_scale", "tscale"))
    Tp = pr.Tables(); consts = {2025: pr.fit_constants(2024), 2026: pr.fit_constants(2025)}; pt = pitcher_table(Tp, consts)
    by_day(pt, "PITCHERS", False); tune_bins(pt, "PITCHERS", ("nu", "n0_scale"))


if __name__ == "__main__":
    main()
