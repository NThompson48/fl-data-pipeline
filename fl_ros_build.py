"""
fl_ros_build.py -- (1) scores the pitcher ROS RA9 and innings forecasts against the 2026 play-by-play at the five backtest cutoffs, and (2) builds the as-of snapshots the dashboard reads
(output/ros_hitters_2026.parquet, output/ros_pitchers_2026.parquet, copied to ../projections-dashboard/data).

Snapshot honesty: every 2026 snapshot uses only information before its as-of date (history from earlier seasons, the season to date, workload / stuff constants fit on 2025 only, the 2025-only
stuff coefficients, seasonal league-level correction from 2025).  The few static inputs that are not date-limited are display-only: the position-adjustment group rates and the trait-model Air%
used for the Mississippi shift.  RA9 / innings scoring uses the 2026 play-by-play; the RA9 baseline's own coefficients were fit on 2025 -> 2026 transitions (including the 2026 season), so the RA9
check is mildly optimistic.

Usage:  python fl_ros_build.py
"""
from __future__ import annotations

import shutil
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_ros as hr
import fl_pitcher_ros as pr
import fl_pitcher_ros_stuff as pst
import fl_ros_production as rp
import fl_ros_workload as wl

warnings.filterwarnings("ignore")
rng = np.random.default_rng(3)
DASH = "../projections-dashboard/data"
SNAP_DATES = ["2026-05-25", "2026-06-08", "2026-06-22", "2026-07-06", "2026-07-20", "2026-08-03", "2026-08-17", "2026-08-31"]


def wrmse(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def main():
    T_h = hr.Tables(); T_p = pr.Tables(); ctx = rp.PitcherContext(T_p)
    D = wl.collect(T_h, T_p)
    wf_h = rp.workload_fit(D, "hit", (2025,)); wf_p = rp.workload_fit(D, "pit", (2025,))
    consts = pr.fit_constants(2025); n0s = pst.reliability(ctx.raw); B = rp.STUFF_B["clean"]
    t0, t1 = T_p.season_bounds(2026)

    # ------------------------------------------------------------------ (1) score RA9 / innings at the five cutoffs (2026, play-by-play)
    rows = []
    for c in wl.CUTS:
        asof = t0 + (t1 - t0) * c
        p = rp.pitcher_snapshot(T_p, ctx, 2026, asof, consts, n0s, B, wf_p); p["cut"] = c; rows.append(p)
    P = pd.concat(rows)
    d = P[(P["act_ip"] >= 10)].copy()
    lg_ytd = {c: (ctx.runs[ctx.runs["date"] < t0 + (t1 - t0) * c]["runs"].sum() * 9) / (ctx.runs[ctx.runs["date"] < t0 + (t1 - t0) * c]["outs"].sum() / 3) for c in wl.CUTS}
    d["lg_ra9"] = d["cut"].map(lg_ytd)
    d["ytd_shr"] = d["lg_ra9"] + (d["ytd_ip"] / (d["ytd_ip"] + 40.0)) * (d["ytd_ra9"].fillna(d["lg_ra9"]) - d["lg_ra9"])
    y = d["act_ra9"].to_numpy(); w, a, b = ew.weights_from_base(y, d["ros_ra9"].to_numpy(), d["act_ip"].to_numpy())
    print(f"2026 ROS RA9 (play-by-play): {len(d)} pitcher-cutoffs with >= 10 ROS IP; law a={a:.3f} b={b:.1f}")
    models = {"league RA9 so far": d["lg_ra9"], "YTD RA9 face value": d["ytd_ra9"].fillna(d["lg_ra9"]), "YTD RA9 shrunk (40 IP)": d["ytd_shr"], "preseason inputs -> RA9 baseline": d["pre_ra9"], "ROS K%/BB% -> RA9 baseline": d["ros_ra9"]}
    rr = []
    for k, v in models.items():
        r = {"model": k, "RMSE": wrmse(y, v.to_numpy(), w), "mean err": float(np.average(y - v.to_numpy(), weights=w))}
        for c in wl.CUTS:
            m = (d["cut"] == c).to_numpy(); r[f"{int(c*100)}%"] = wrmse(y[m], v.to_numpy()[m], w[m])
        rr.append(r)
    print(pd.DataFrame(rr).round(3).to_string(index=False))
    ug = pd.Series(d.index).unique(); by = pd.Series(np.arange(len(d))).groupby(d.index.to_numpy()).apply(np.array)
    for A, Bm in (("ROS K%/BB% -> RA9 baseline", "preseason inputs -> RA9 baseline"), ("ROS K%/BB% -> RA9 baseline", "YTD RA9 shrunk (40 IP)")):
        dd = []
        for _ in range(500):
            ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(wrmse(y[ii], models[A].to_numpy()[ii], w[ii]) - wrmse(y[ii], models[Bm].to_numpy()[ii], w[ii]))
        print(f"   {A} minus {Bm}: {np.mean(dd):+.4f} [{np.percentile(dd,5):+.4f}, {np.percentile(dd,95):+.4f}]")
    # innings
    di = P[P["ros_ip"].notna()].copy(); ip_act = di["act_ip"].fillna(0)
    naive = di["ytd_ip"].fillna(0) / np.maximum(di["frac"], 0.05) * (1 - di["frac"]) * 0.0
    print(f"\n2026 ROS innings: {len(di)} pitcher-cutoffs (everyone who had appeared); mean actual {ip_act.mean():.1f}; RMSE {np.sqrt(np.mean((ip_act - di['ros_ip']) ** 2)):.1f}; MAE {np.mean(np.abs(ip_act - di['ros_ip'])):.1f}; bias {np.mean(ip_act - di['ros_ip']):+.2f}; "
          f"corr {np.corrcoef(ip_act, di['ros_ip'])[0, 1]:.3f}; league BF per IP {4.512} vs actual ROS BF/IP {di['act_bf'].sum() / (di['act_ip'].sum()):.3f}")

    # ------------------------------------------------------------------ (2) snapshots for the dashboard
    hd = rp.display_names(T_h.pa, "batter"); pdn = rp.display_names(T_p.pa, "pitcher")
    H, Pp = [], []
    for ds in SNAP_DATES:
        asof = pd.Timestamp(ds)
        h = rp.hitter_snapshot(T_h, 2026, asof, wf_h, (2025,)); h["asof"] = ds; h["name"] = hd.reindex(h.index).to_numpy(); H.append(h.reset_index())
        p = rp.pitcher_snapshot(T_p, ctx, 2026, asof, consts, n0s, B, wf_p); p["asof"] = ds; p["name"] = pdn.reindex(p.index).to_numpy(); Pp.append(p.reset_index())
        print("snapshot built:", ds, len(h), "hitters,", len(p), "pitchers", flush=True)
    H = pd.concat(H, ignore_index=True); Pp = pd.concat(Pp, ignore_index=True)
    H.to_parquet("output/ros_hitters_2026.parquet", index=False); Pp.to_parquet("output/ros_pitchers_2026.parquet", index=False)
    for f in ("output/ros_hitters_2026.parquet", "output/ros_pitchers_2026.parquet"):
        shutil.copy(f, DASH)
    print("saved and copied to the dashboard data folder")


if __name__ == "__main__":
    main()
