"""
fl_pitcher_ros.py -- rest-of-season (ROS) pitcher K% / BB% update, the pitcher twin of fl_hitter_ros.py.

At an as-of date the current season is the NEWEST partial season in a Marcel-style regression of K% and BB% (per batter faced) toward the league:
    rate = lg_now + [ nu n_ytd (x_ytd - lg_now) + sum_j d^j n_j (x_j - lg_j) ] / [ nu n_ytd + sum_j d^j n_j + n0 ]
with (d, n0) fit on the pitcher season-to-season transitions (2022 -> 2026, leave-one-transition-out, standard inverse-variance weights) -- the pitcher analogue of the hitters' constants.
History: 2022-24 StatsCrew box scores (name-keyed), 2025-26 from the pitch data via war_pitchers (fl_innings_projection.build_panel).  Pitch-level terms are added in fl_pitcher_ros_terms.py.

Inputs per as-of date: dated PA / pitch tables from Supabase (output/pitcher_pa_ros_2025_2026.parquet, output/pitcher_pitches_dated_2025_2026.parquet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_hitter_baseline_backtest as bt
import fl_innings_projection as ip

PA_PATH = "output/pitcher_pa_ros_2025_2026.parquet"
PITCH_PATH = "output/pitcher_pitches_dated_2025_2026.parquet"
KEYS = {"k": ("k", "bf"), "bbr": ("bbr", "bf")}


def history_panel():
    p = ip.build_panel()
    p = p[(p["bf"] > 0)].copy()
    p["k"] = p["so"] / p["bf"]; p["bbr"] = p["bb"] / p["bf"]
    return p


def fit_constants(max_target_year: int = 2026, min_target_year: int = 2023):
    """(delta, n0) per stat from the pitcher season transitions (target year <= max_target_year, so a backtest never sees its own season), LOTO, standard weights.  Re-uses the hitter backtest machinery with the denominator switched to batters faced."""
    p = history_panel(); q = p.copy()
    saved = dict(bt.DEN); saved_min = bt.MIN_NEXT
    bt.DEN["k"] = "bf"; bt.DEN["bbr"] = "bf"; bt.MIN_NEXT = 60
    out = {}
    try:
        for s in ("k", "bbr"):
            d = bt.transitions(q, s); d = d[((d["t"] + 1) <= max_target_year) & ((d["t"] + 1) >= min_target_year)]
            best = None
            for delta in (0.25, 0.5, 0.65, 0.8, 1.0):
                E, _ = bt.loto(d, delta)
                r = bt.rmse(E, E["pred"])
                if best is None or r < best[0]:
                    best = (r, delta)
            delta = best[1]
            out[s] = {"delta": delta, "n0": float(bt.fit_n0(d, delta)), "rmse": best[0], "rmse_league": bt.rmse(d, d["lg_next"]), "rmse_last": bt.rmse(d, d["x_t"]), "n": int(len(d)),
                      "y/y": float(np.corrcoef(d["x_t"] - d["lg_t"], d["x_next"] - d["lg_next"])[0, 1])}
    finally:
        bt.DEN.clear(); bt.DEN.update(saved); bt.MIN_NEXT = saved_min
    return out


class Tables:
    def __init__(self):
        pa = pd.read_parquet(PA_PATH); pa["name_clean"] = pa["pitcher"].apply(ip.lastfirst_to_clean)
        pa["k"] = (pa["kind"] == "strikeout").astype(int); pa["bb"] = (pa["kind"] == "walk").astype(int)
        self.pa = pa
        pi = pd.read_parquet(PITCH_PATH); pi["name_clean"] = pi["pitcher"].apply(ip.lastfirst_to_clean); self.pi = pi
        self.panel = history_panel()
        self.by = {y: self.panel[self.panel["year"] == y].set_index("name_clean") for y in range(2022, 2027)}
        self.lg = {s: self.panel.groupby("year").apply(lambda g, s=s: np.average(g[KEYS[s][0]], weights=g["bf"])) for s in KEYS}

    def season_bounds(self, year):
        d = self.pa[self.pa["year"] == year]["date"]
        return d.min(), d.max()


def features(T: Tables, year: int, asof, min_ros_bf: int = 30, population: str = "ros", horizon_days=None, half_life=None, window_days=None):
    pa = T.pa[T.pa["year"] == year]
    y_pa = pa[pa["date"] < asof]
    if window_days:
        y_pa = y_pa[y_pa["date"] >= asof - pd.Timedelta(days=window_days)]
    if half_life:
        wgt = 0.5 ** ((asof - y_pa["date"]).dt.days.to_numpy() / half_life)
        y_pa = y_pa.assign(_w=wgt, k_w=y_pa["k"] * wgt, bb_w=y_pa["bb"] * wgt)
        ytd = y_pa.groupby("name_clean").agg(n=("_w", "sum"), k=("k_w", "sum"), bb=("bb_w", "sum"))
    else:
        ytd = y_pa.groupby("name_clean").agg(n=("k", "size"), k=("k", "sum"), bb=("bb", "sum"))
    r_pa = pa[pa["date"] >= asof]
    if horizon_days:
        r_pa = r_pa[r_pa["date"] < asof + pd.Timedelta(days=horizon_days)]
    ros = r_pa.groupby("name_clean").agg(n_ros=("k", "size"), k_ros=("k", "mean"), bb_ros=("bb", "mean"))
    if population == "ytd":
        f = ytd.join(ros, how="left"); f = f[f["n"] > 0].copy(); f["n_ros"] = f["n_ros"].fillna(0)
    else:
        f = ros.join(ytd, how="left"); f = f[f["n_ros"] >= min_ros_bf].copy()
    for c in ("n", "k", "bb"):
        f[c] = f[c].fillna(0)
    for s in KEYS:
        for j in (1, 2, 3, 4):
            y = year - j
            if y >= 2022 and y in T.by:
                b = T.by[y]; b = b[~b.index.duplicated(keep="first")]
                x = b[KEYS[s][0]].reindex(f.index); n = b["bf"].reindex(f.index).fillna(0.0)
                f[f"h_{s}_x{j}"] = x.fillna(0.0); f[f"h_{s}_n{j}"] = np.where(x.isna(), 0.0, n); f[f"h_{s}_l{j}"] = T.lg[s].get(y, 0.0)
            else:
                f[f"h_{s}_x{j}"] = 0.0; f[f"h_{s}_n{j}"] = 0.0; f[f"h_{s}_l{j}"] = 0.0
    lg_now = {"k": f["k"].sum() / f["n"].sum(), "bbr": f["bb"].sum() / f["n"].sum()}
    return f, lg_now


def predict(f, lg_now, consts, nu=1.0, use_ytd=True, use_hist=True, n0_scale=1.0, max_hist=4):
    out = pd.DataFrame(index=f.index)
    for s in KEYS:
        delta, n0 = consts[s]["delta"], consts[s]["n0"]
        cnt = f["k"] if s == "k" else f["bb"]; den = f["n"]
        x_ytd = np.where(den > 0, cnt / den.replace(0, np.nan), 0.0)
        num = nu * den * (x_ytd - lg_now[s]) if use_ytd else 0.0 * den
        w = nu * den if use_ytd else 0.0 * den
        if use_hist:
            for j in range(1, max_hist + 1):
                wj = delta ** j
                num = num + wj * f[f"h_{s}_n{j}"] * (f[f"h_{s}_x{j}"] - f[f"h_{s}_l{j}"]) * (f[f"h_{s}_n{j}"] > 0)
                w = w + wj * f[f"h_{s}_n{j}"]
        out[s] = lg_now[s] + num / (w + n0 * n0_scale)
        out[f"pa_eff_{s}"] = w
    return out
