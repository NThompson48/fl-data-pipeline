"""
fl_ros_hitter_published_prior.py -- should the HITTER ROS update start from the published preseason projection (fl_hitter_pre2026.py: the production path run on 2025-and-earlier data only: leak-free for every 2026 date)
instead of the ROS model's own history regression + pitch terms?

For each stat s in {wOBA, K%, BB%}:   est_s = lg_now + (n_p dev0_s + nu_s n_ytd (x_ytd,s - lg_now)) / (n_p + nu_s n_ytd),   n_p = (sum_j delta_s^j n_j + n0_s)   [phi = 1: the weight the Marcel update gives its own history]
    dev0_s = published_s - the 2025 league level of s (the level the published projection is built around; hitters' preseason numbers are NOT recentred on their own mean, unlike the pitcher table);  nu = 1 (wOBA, K%), .4 (BB%, as production)
    final_s = est_s + tau x (the ROS model's own pitch-level increment: final - regressed)       tau in {0, .5, 1}: the published number already contains 2025 pitch information
Reference = production ROS (history regression + pitch terms).  2026 only, as-of days 7-105 by 7, standard inverse-variance weights frozen from the reference, hitter-cluster bootstrap.
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_ros as hr

warnings.filterwarnings("ignore")
rng = np.random.default_rng(404)
DAYS = list(range(7, 106, 7)); TAUS = (0.0, 0.5, 1.0)
NU = {"woba": 1.0, "k": 1.0, "bbr": hr.PROD["nu_bbr"]}


def wr(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def main():
    T = hr.Tables(); t0, _ = T.season_bounds(2026)
    global MIS_NAMES
    MIS_NAMES = set(T.pa[(T.pa["year"] == 2026) & (T.pa["batter_team"] == "Mississippi Mud Monsters")]["name_clean"])
    pub = pd.read_parquet("output/hitter_pre2026_published.parquet").set_index("name_clean")
    rows = []
    for d in DAYS:
        f, lg = hr.features(T, 2026, t0 + pd.Timedelta(days=d), min_ros_pa=30)
        if len(f) < 40:
            continue
        full = hr.predict(f, lg); hist = hr.predict(f, lg, use_ytd=False)
        j = f.join(pub[["proj_woba", "proj_k", "proj_bbr", "gap"]], how="left"); has = j[["proj_woba", "proj_k", "proj_bbr"]].notna().all(axis=1).to_numpy()
        for stat, ycol, final, pcol, num_den in (("woba", "woba_ros", "woba_final", "proj_woba", ("woba_sum", "n")), ("k", "k_ros", "k_final", "proj_k", ("k", "n")), ("bbr", "bb_ros", "bbr_final", "proj_bbr", ("bb", "n"))):
            delta, n0 = hr.SPEC[stat]
            w_hist = sum(delta ** jj * f[f"h_{stat}_n{jj}"] for jj in (1, 2, 3)).to_numpy(); n_p = w_hist + n0
            n = f["n"].to_numpy(); x_ytd = np.where(n > 0, f[num_den[0]].to_numpy() / np.where(n > 0, n, 1), 0.0); nu = NU[stat]
            pubv = j[pcol].to_numpy(); dev0 = np.where(has, pubv - T.lg[stat][2025], 0.0)          # relative to the 2025 league level the projection was built around (NOT recentred on its own mean: that would remove the returner premium)
            est = lg[stat] + (n_p * dev0 + nu * n * (x_ytd - lg[stat])) / (n_p + nu * n)
            regressed = hist[stat].to_numpy() if False else (hr.predict(f, lg, trackman=False)["blend" if stat == "woba" else stat]).to_numpy()
            incr = full[final].to_numpy() - regressed                                         # the ROS model's own pitch-level increment
            df = pd.DataFrame({"y": f[ycol].to_numpy(), "n": f["n_ros"].to_numpy(), "id": f.index.to_numpy(), "day": d, "stat": stat, "has_pub": has, "mis": f.index.isin(MIS_NAMES), "prod": full[final].to_numpy(), "gap": j["gap"].fillna(False).to_numpy() if "gap" in j else False})
            for tau in TAUS:
                df[f"pub|{tau}"] = np.where(has, est + tau * incr, full[final].to_numpy())
            rows.append(df)
    D = pd.concat(rows, ignore_index=True)
    for stat in ("woba", "k", "bbr"):
        d = D[D["stat"] == stat].reset_index(drop=True); w = ew.weights_from_base(d["y"].to_numpy(), d["prod"].to_numpy(), d["n"].to_numpy())[0]; y = d["y"].to_numpy()
        by = pd.Series(np.arange(len(d))).groupby(d["id"].to_numpy()).apply(np.array); ug = by.index.to_numpy(); hp = d["has_pub"].to_numpy(); out = []
        print(f"\n{stat.upper()}: {len(d)} hitter-dates, {hp.mean():.0%} with a published preseason projection")
        for c in ["prod"] + [f"pub|{t}" for t in TAUS]:
            r = {"model": c, "RMSE": wr(y, d[c].to_numpy(), w), "RMSE (with published)": wr(y[hp], d[c].to_numpy()[hp], w[hp])}
            if c != "prod":
                dd = []
                for _ in range(400):
                    ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(wr(y[ii], d[c].to_numpy()[ii], w[ii]) - wr(y[ii], d["prod"].to_numpy()[ii], w[ii]))
                dd = np.array(dd); r.update({"vs production": np.mean(dd), "90% interval": f"[{np.percentile(dd, 5):+.5f}, {np.percentile(dd, 95):+.5f}]", "P(better)": float((dd < 0).mean())})
            out.append(r)
        print(pd.DataFrame(out).round(5).to_string(index=False))
        for mflag, lab in ((True, "Mississippi hitters (park-neutral published number vs park-inclusive target)"), (False, "all others")):
            m = hp & (d["mis"].to_numpy() == mflag); print(f"   {lab}: n={m.sum()}  " + "  ".join(f"{c}: {wr(y[m], d[c].to_numpy()[m], w[m]):.5f}" for c in ["prod"] + [f"pub|{t}" for t in TAUS]))
        for g, lab in ((False, "NON-gap (played the latest season)"), (True, "GAP (last season two years back; production multiplies n0 by 2.0 / 1.0 / 1.7)")):
            m = hp & (d["gap"].to_numpy() == g); print(f"   {lab}: n={m.sum()}  " + "  ".join(f"{c}: {wr(y[m], d[c].to_numpy()[m], w[m]):.5f}" for c in ["prod"] + [f"pub|{t}" for t in TAUS]))


if __name__ == "__main__":
    main()
