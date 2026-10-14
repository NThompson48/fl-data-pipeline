"""
fl_ros_robust.py -- robustness of the rest-of-season (ROS) models: cutoffs, horizons, in-season recency, and bootstrapped constants.  Hitters (wOBA / K% / BB%) and pitchers (K% / BB%).

1  CUTOFFS    16 cutoffs from 10% to 85% of each season (not 5): how consistent are the gains of (ROS update vs preseason), (+ pitch-level terms vs ROS update) across cutoffs and seasons?
              Cell-level table (season x cutoff = 32 cells), share of cells where the gain is negative, and a BLOCK bootstrap over cutoffs within season (the cells are not independent) next to the
              player-cluster bootstrap.
2  HORIZONS   forecast the next 14 / 28 / 56 days and the rest of the season (8 cutoffs, 10-80%): does the update / term gain survive a short horizon, and does it grow or shrink?
3  RECENCY    in-season rate evidence: full season to date vs exponentially recency-weighted (half-life 90 / 45 / 21 days) vs a trailing window (90 / 60 / 30 days).  Selection on 2025, scored on 2026
              (and the reverse as a check).
4  WEIGHT     the in-season weight nu (1 = production): grid 0.5 ... 3, error with the pitch-level terms, and a player bootstrap of the best nu.
Standard inverse-variance weights (law fit on the ROS-update residuals, frozen per table); frozen production constants unless stated.

Usage:  python fl_ros_robust.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_ros as hr
import fl_hitter_ros_backtest as hb
import fl_pitcher_ros as pr
import fl_pitcher_ros_stuff as pst
import fl_ros_production as rp

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2027)
C16 = [round(x, 2) for x in np.arange(0.10, 0.851, 0.05)]
C8 = [round(x, 2) for x in np.arange(0.10, 0.801, 0.10)]
HSTATS = {"woba": ("woba_ros", "woba_final"), "k": ("k_ros", "k_final"), "bbr": ("bb_ros", "bbr_final")}
PSTATS = {"k": ("k_ros", "k"), "bb": ("bb_ros", "bbr")}


def wr(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


# ---------------------------------------------------------------------------------------------------------------------------- row builders
def hitter_rows(T, year, c, min_n=30, nu=1.0, **fk):
    t0, t1 = T.season_bounds(year); asof = t0 + (t1 - t0) * c
    f, lg = hr.features(T, year, asof, min_ros_pa=min_n, **fk)
    m = hb.run_models(f, lg, nu=nu)
    out = {}
    for s in HSTATS:
        d = pd.DataFrame({"y": f[HSTATS[s][0]].to_numpy(), "n_ros": f["n_ros"].to_numpy(), "year": year, "cut": c, "id": (f.index + "|" + str(year)).to_numpy(), "ytd": f["n"].to_numpy()})
        for k, v in m[s].items():
            d[k] = v
        out[s] = d
    return out


class PitcherStuff:
    """Caches the (expensive) stuff / command features per (year, cutoff)."""
    def __init__(self, T_p):
        self.T = T_p; self.raw = rp.pst.fp.load_fl_pitches_cached(); self.names = pst.idmap(self.raw); self.n0s = pst.reliability(self.raw); self.cache = {}

    def get(self, year, c):
        if (year, c) not in self.cache:
            t0, t1 = self.T.season_bounds(year); asof = t0 + (t1 - t0) * c
            self.cache[(year, c)] = pst.stuff_features(self.raw, year, asof, self.names, self.n0s)
        return self.cache[(year, c)]


def pitcher_rows(T, stuff, consts, B, year, c, min_n=30, nu=1.0, terms=True, **fk):
    t0, t1 = T.season_bounds(year); asof = t0 + (t1 - t0) * c
    f, lg = pr.features(T, year, asof, min_ros_bf=min_n, **fk)
    base = pr.predict(f, lg, consts, nu=nu); pre = pr.predict(f, lg, consts, use_ytd=False)
    out = {}
    sf = stuff.get(year, c).reindex(f.index).fillna(0.0) if terms else None
    for s, (ycol, key) in PSTATS.items():
        cnt = f["k"] if s == "k" else f["bb"]; den = f["n"]; x = np.where(den > 0, cnt / den.replace(0, np.nan), lg[key]); n0 = consts[key]["n0"]
        d = pd.DataFrame({"y": f[ycol].to_numpy(), "n_ros": f["n_ros"].to_numpy(), "year": year, "cut": c, "id": (f.index + "|" + str(year)).to_numpy(), "ytd": den.to_numpy(),
                          "0 league average now": lg[key], "1 preseason only": pre[key].to_numpy(), "2 YTD face value": x, "3 YTD naive shrinkage": lg[key] + den.to_numpy() * (x - lg[key]) / (den.to_numpy() + n0),
                          "4 ROS update": base[key].to_numpy()})
        if terms:
            fade = n0 / (base[f"pa_eff_{key}"] + n0)
            d["5 + stuff/command terms"] = (base[key] + (B[0] * sf["wh_sva"] * fade if s == "k" else B[1] * sf["command_score"] * fade)).clip(0.01, 0.6).to_numpy()
        out[s] = d
    return out


# ---------------------------------------------------------------------------------------------------------------------------- summaries
def stack(rows_list, stat):
    return pd.concat([r[stat] for r in rows_list], ignore_index=True)


def frozen_weights(d, ref="4 ROS update (no Trackman)"):
    ref = ref if ref in d.columns else "4 ROS update"
    w, a, b = ew.weights_from_base(d["y"].to_numpy(), d[ref].to_numpy(), d["n_ros"].to_numpy())
    return w


def cell_gains(d, w, pairs):
    """Per (year, cut) weighted RMSE difference for each (A, B) pair."""
    rows = []
    for (yr, c), g in d.groupby(["year", "cut"]):
        ii = g.index.to_numpy(); r = {"year": yr, "cut": c}
        for A, B in pairs:
            r[f"{A} - {B}"] = wr(g["y"].to_numpy(), g[A].to_numpy(), w[ii]) - wr(g["y"].to_numpy(), g[B].to_numpy(), w[ii])
        rows.append(r)
    return pd.DataFrame(rows)


def block_boot(cells, col, n=2000):
    """Resample cutoffs with replacement WITHIN each season and average (cells are serially dependent: blocks, not points)."""
    seasons = [g[col].to_numpy() for _, g in cells.groupby("year")]
    out = np.empty(n)
    for i in range(n):
        out[i] = np.mean(np.concatenate([rng.choice(s, len(s)) for s in seasons]))
    return out


def player_boot(d, w, A, B, n=500):
    ids = d["id"].to_numpy(); ug = pd.Series(ids).unique(); by = pd.Series(np.arange(len(d))).groupby(ids).apply(np.array); y = d["y"].to_numpy(); a = d[A].to_numpy(); b = d[B].to_numpy(); out = np.empty(n)
    for i in range(n):
        ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); out[i] = wr(y[ii], a[ii], w[ii]) - wr(y[ii], b[ii], w[ii])
    return out


def interval(x, lo=5, hi=95):
    return f"[{np.percentile(x, lo):+.5f}, {np.percentile(x, hi):+.5f}]"


def report_cutoffs(name, d, stat, pairs):
    w = frozen_weights(d)
    cells = cell_gains(d, w, pairs)
    print(f"\n  {name} {stat}: {len(d)} rows, {len(cells)} season x cutoff cells")
    rows = []
    for A, B in pairs:
        col = f"{A} - {B}"; bb = block_boot(cells, col); pb_ = player_boot(d, w, A, B)
        rows.append({"comparison (negative = A better)": col, "pooled": wr(d['y'].to_numpy(), d[A].to_numpy(), w) - wr(d['y'].to_numpy(), d[B].to_numpy(), w), "mean over cells": cells[col].mean(), "cells < 0": f"{int((cells[col] < 0).sum())}/{len(cells)}",
                     "worst / best cell": f"{cells[col].max():+.4f} / {cells[col].min():+.4f}", "2025 mean": cells[cells.year == 2025][col].mean(), "2026 mean": cells[cells.year == 2026][col].mean(),
                     "cutoff-block 90%": interval(bb), "player 90%": interval(pb_)})
    print(pd.DataFrame(rows).round(5).to_string(index=False))


# ---------------------------------------------------------------------------------------------------------------------------- main
def main():
    T_h = hr.Tables(); T_p = pr.Tables(); stuff = PitcherStuff(T_p)
    consts = {2025: pr.fit_constants(2024), 2026: pr.fit_constants(2025)}; Bk = {2025: rp.STUFF_B["clean"], 2026: rp.STUFF_B["clean"]}
    B = rp.STUFF_B["clean"]
    hp = [("4 ROS update (no Trackman)", "1 preseason only"), ("4 ROS update (no Trackman)", "3 YTD naive shrinkage"), ("5 + pitch-level terms", "4 ROS update (no Trackman)")]
    pp = [("4 ROS update", "1 preseason only"), ("4 ROS update", "3 YTD naive shrinkage"), ("5 + stuff/command terms", "4 ROS update")]

    # ------------------------------------------------------------------ 1. cutoffs
    print("=" * 110 + "\n1. CUTOFF ROBUSTNESS: 16 cutoffs, 10%-85% of the season\n" + "=" * 110)
    H = [hitter_rows(T_h, y, c) for y in (2025, 2026) for c in C16]
    for s in HSTATS:
        report_cutoffs("HITTERS", stack(H, s), s, hp)
    P16 = [pitcher_rows(T_p, stuff, consts[y], B, y, c, terms=False) for y in (2025, 2026) for c in C16]
    for s in PSTATS:
        report_cutoffs("PITCHERS (update only)", stack(P16, s), s, pp[:2])
    P8 = {(y, c): pitcher_rows(T_p, stuff, consts[y], B, y, c) for y in (2025, 2026) for c in C8}
    P8l = list(P8.values())
    for s in PSTATS:
        report_cutoffs("PITCHERS (8 cutoffs, with stuff / command terms)", stack(P8l, s), s, pp)

    # ------------------------------------------------------------------ 2. horizons
    print("\n" + "=" * 110 + "\n2. FORECAST HORIZON: next 14 / 28 / 56 days vs the rest of the season (8 cutoffs, min 15 PA / BF in the window)\n" + "=" * 110)
    for tag, T, mk in (("HITTERS", T_h, "h"), ("PITCHERS", T_p, "p")):
        for H_days in (14, 28, 56, None):
            rows_ = {}
            for y in (2025, 2026):
                for c in C8:
                    rows_[(y, c)] = hitter_rows(T_h, y, c, min_n=15 if H_days else 30, horizon_days=H_days) if mk == "h" else pitcher_rows(T_p, stuff, consts[y], B, y, c, min_n=15 if H_days else 30, horizon_days=H_days)
            for s in (HSTATS if mk == "h" else PSTATS):
                d = stack(list(rows_.values()), s); w = frozen_weights(d)
                A, Bm, Cc = ("4 ROS update (no Trackman)", "1 preseason only", "5 + pitch-level terms") if mk == "h" else ("4 ROS update", "1 preseason only", "5 + stuff/command terms")
                pre = wr(d["y"].to_numpy(), d[Bm].to_numpy(), w); upd = wr(d["y"].to_numpy(), d[A].to_numpy(), w); trm = wr(d["y"].to_numpy(), d[Cc].to_numpy(), w)
                bs = player_boot(d, w, A, Bm, 300); bt_ = player_boot(d, w, Cc, A, 300)
                print(f"  {tag:8s} {s:5s} horizon {str(H_days or 'ROS'):>4s}: n={len(d):5d}  preseason {pre:.4f} -> update {upd:.4f} ({upd-pre:+.4f}, {100*(upd-pre)/pre:+.1f}% {interval(bs,5,95)}) -> + terms {trm:.4f} ({trm-upd:+.4f}, {100*(trm-upd)/upd:+.1f}% {interval(bt_,5,95)})")

    # ------------------------------------------------------------------ 3. recency of the in-season evidence
    print("\n" + "=" * 110 + "\n3. IN-SEASON RECENCY: full season to date vs recency-weighted / trailing window (rest-of-season target, 8 cutoffs)\n" + "=" * 110)
    variants = {"full season to date": {}, "half-life 90d": {"half_life": 90}, "half-life 45d": {"half_life": 45}, "half-life 21d": {"half_life": 21},
                "last 90d": {"window_days": 90}, "last 60d": {"window_days": 60}, "last 30d": {"window_days": 30}}
    for tag, mk in (("HITTERS", "h"), ("PITCHERS", "p")):
        res = {}
        for vname, fk in variants.items():
            rows_ = [hitter_rows(T_h, y, c, **fk) if mk == "h" else pitcher_rows(T_p, stuff, consts[y], B, y, c, **fk) for y in (2025, 2026) for c in C8]
            res[vname] = rows_
        for s in (HSTATS if mk == "h" else PSTATS):
            key = "5 + pitch-level terms" if mk == "h" else "5 + stuff/command terms"
            ref = res["full season to date"]; d0 = stack(ref, s); w0 = frozen_weights(d0)       # one weight set for every variant (same target rows)
            tab = []
            for vname, rows_ in res.items():
                d = stack(rows_, s); assert len(d) == len(d0)
                r = {"variant": vname}
                for y in (2025, 2026):
                    mk_ = (d["year"] == y).to_numpy(); r[str(y)] = wr(d["y"].to_numpy()[mk_], d[key].to_numpy()[mk_], w0[mk_])
                r["all"] = wr(d["y"].to_numpy(), d[key].to_numpy(), w0)
                tab.append(r)
            T_ = pd.DataFrame(tab).set_index("variant")
            print(f"\n  {tag} {s}: RMSE of the full model (update + terms) by recency variant"); print(T_.round(5).to_string())
            for tr, te in ((2025, 2026), (2026, 2025)):
                best = T_[str(tr)].idxmin(); d_b = stack(res[best], s); mk_ = (d0["year"] == te).to_numpy()
                dd = []
                ids = d0["id"].to_numpy()[mk_]; ug = pd.Series(ids).unique(); by = pd.Series(np.arange(mk_.sum())).groupby(ids).apply(np.array)
                yy = d0["y"].to_numpy()[mk_]; ab = d_b[key].to_numpy()[mk_]; a0 = d0[key].to_numpy()[mk_]; ww = w0[mk_]
                for _ in range(300):
                    ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(wr(yy[ii], ab[ii], ww[ii]) - wr(yy[ii], a0[ii], ww[ii]))
                print(f"    select on {tr} -> '{best}'; scored on {te}: change vs full season {np.mean(dd):+.5f} {interval(dd)}")

    # ------------------------------------------------------------------ 4. in-season weight nu, bootstrapped
    print("\n" + "=" * 110 + "\n4. IN-SEASON WEIGHT nu (1 = production): error of the full model by nu, and a player bootstrap of the best nu\n" + "=" * 110)
    NUS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    for tag, mk in (("HITTERS", "h"), ("PITCHERS", "p")):
        res = {nu: [hitter_rows(T_h, y, c, nu=nu) if mk == "h" else pitcher_rows(T_p, stuff, consts[y], B, y, c, nu=nu) for y in (2025, 2026) for c in C8] for nu in NUS}
        for s in (HSTATS if mk == "h" else PSTATS):
            key = "5 + pitch-level terms" if mk == "h" else "5 + stuff/command terms"
            d1 = stack(res[1.0], s); w = frozen_weights(d1); y = d1["y"].to_numpy()
            P_ = {nu: stack(res[nu], s)[key].to_numpy() for nu in NUS}
            err = {nu: wr(y, P_[nu], w) for nu in NUS}
            ids = d1["id"].to_numpy(); ug = pd.Series(ids).unique(); by = pd.Series(np.arange(len(d1))).groupby(ids).apply(np.array)
            best = []
            for _ in range(300):
                ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); e = [wr(y[ii], P_[nu][ii], w[ii]) for nu in NUS]; best.append(NUS[int(np.argmin(e))])
            share = pd.Series(best).value_counts(normalize=True).sort_index()
            print(f"  {tag:8s} {s:5s}: " + "  ".join(f"nu={nu}: {err[nu]:.5f}" for nu in NUS) + f" | bootstrap best nu: " + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))


if __name__ == "__main__":
    main()
