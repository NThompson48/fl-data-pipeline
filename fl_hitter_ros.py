"""
fl_hitter_ros.py -- rest-of-season (ROS) hitter projections: update the preseason model with the current season's plate appearances AND its pitch-level skills.

At an as-of date inside a season, the current season is treated as the newest, partial season in the same regression the preseason model uses:
    rate = lg_now + [ nu n_ytd (x_ytd - lg_now) + sum_j d^j n_j (x_j - lg_j) ] / [ nu n_ytd + sum_j d^j n_j + n0 ]          j = 1 (last season), 2, ...
(nu = weight of in-season PA relative to a PA from last season's baseline; nu = 1 is the production weighting), for wOBA, K%, BB%, HR/PA, BABIP; wOBA = 50/50 of the regressed wOBA
and wOBA rebuilt from the regressed components (frozen production weights); then the pitch-level terms, computed on this season's pitches (pooled with last season's where they exist):
    wOBA   + gamma (shrunk xwOBA - blend) + B_EV90 x shrunk EV90 deviation x fade        K%  + B_WHIFF x shrunk Whiff% deviation x fade        BB%  - B_SWING ... Swing%
with fade = n0 / (PA_eff + n0), PA_eff = nu n_ytd + d n_1.  All constants are the production constants (fl_hitter_projection.py) unless a backtest refits them.

Inputs per as-of date come from the dated tables built from Supabase `pitches` (output/hitter_pa_ros_2025_2026.parquet, output/hitter_pitches_dated_2025_2026.parquet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev
import fl_innings_projection as ip

PA_PATH = "output/hitter_pa_ros_2025_2026.parquet"
PITCH_PATH = "output/hitter_pitches_dated_2025_2026.parquet"
SPEC = {"woba": (0.75, 440.0), "k": (0.75, 79.3), "bbr": (0.5, 146.6), "hrr": (0.5, 136.8), "babip": (0.5, 851.6)}
BETA = {"k": -0.177, "bbr": 0.516, "hrr": 1.414}          # component weights of the preseason fit without BABIP (2026-10-08; was K -.157, BB .496, HR 1.378, BABIP .426); ROS effect +-.00002 (fl_ros_babip_drop_test.py)
HIT = ("single", "double", "triple")
BIP = ("single", "double", "triple", "out", "fielders_choice", "error")
SWING = {"strike_swinging", "foul_not_fieldable", "foul_fieldable", "foul_ball", "in_play"}
PROD = dict(gamma=0.485, n0x=200.0, b_ev=0.00377, n0_ev=14.26, b_whiff=0.605, n0_whiff=52.5, b_swing=-0.387, n0_swing=92.1, nu=1.0, nu_bbr=0.4, trackman_scale=1.0)   # nu_bbr: in-season weight for BB% only (fit on 2025 ROS rows, scored on 2026: -.0006; DECISION_LOG)


class Tables:
    """Loads the dated PA / pitch tables and the history panel once."""
    def __init__(self):
        pa = pd.read_parquet(PA_PATH); pa["name_clean"] = pa["batter"].apply(ip.lastfirst_to_clean)
        pa["k"] = (pa["kind"] == "strikeout").astype(int); pa["bb"] = (pa["kind"] == "walk").astype(int); pa["hr"] = (pa["kind"] == "home_run").astype(int)
        pa["bip"] = pa["kind"].isin(BIP).astype(int); pa["hit"] = pa["kind"].isin(HIT).astype(int)
        pa["ev"] = np.where((pa["kind"].isin(BIP + ("home_run",))) & pa["exit_speed"].notna() & pa["angle"].notna() & (pa["exit_speed"] <= 120), pa["exit_speed"], np.nan)
        self.pa = pa
        pi = pd.read_parquet(PITCH_PATH); pi = pi[pi["outside_zone"].notna()].copy(); pi["name_clean"] = pi["batter"].apply(ip.lastfirst_to_clean)
        pi["swing"] = pi["pitch_call"].isin(SWING).astype(int); pi["whiff"] = pi["pitch_call"].eq("strike_swinging").astype(int); pi["pitches"] = 1
        pi["o_pitch"] = (pi["outside_zone"] == 1).astype(int); pi["o_swing"] = ((pi["outside_zone"] == 1) & (pi["swing"] == 1)).astype(int)
        self.pi = pi
        p = dev.load(); self.panel = p; q = p.copy(); q["woba"] = q["woba_n"]; self.q = q
        self.lg = {s: bt.league(q, s) for s in SPEC}
        self.by = {y: q[q["year"] == y].set_index("name_clean") for y in range(2022, 2027)}

    def season_bounds(self, year):
        d = self.pa[self.pa["year"] == year]["date"]
        return d.min(), d.max()


def _counts(pa, pi, w=None):
    """Per-hitter counts.  `w` (optional, one weight per PA row) makes the RATE counts (n, wOBA, K, BB, HR, balls in play, hits) recency-weighted; the skill counts (xwOBA, EV90, pitch counts) stay unweighted."""
    if w is not None:
        pa = pa.assign(_w=w, woba_w=pa["woba"] * w, k_w=pa["k"] * w, bb_w=pa["bb"] * w, hr_w=pa["hr"] * w, bip_w=pa["bip"] * w, hit_w=pa["hit"] * w)
        g = pa.groupby("name_clean").agg(n=("_w", "sum"), woba_sum=("woba_w", "sum"), k=("k_w", "sum"), bb=("bb_w", "sum"), hr=("hr_w", "sum"), bip=("bip_w", "sum"), hit=("hit_w", "sum"),
                                         xw_sum=("xw", "sum"), n_ev=("ev", "count"), n_raw=("woba", "size"))
    else:
        g = pa.groupby("name_clean").agg(n=("woba", "size"), woba_sum=("woba", "sum"), k=("k", "sum"), bb=("bb", "sum"), hr=("hr", "sum"), bip=("bip", "sum"), hit=("hit", "sum"),
                                         xw_sum=("xw", "sum"), n_ev=("ev", "count"), n_raw=("woba", "size"))
    q90 = pa.dropna(subset=["ev"]).groupby("name_clean")["ev"].quantile(0.9).rename("ev90")
    p = pi.groupby("name_clean").agg(pitches=("pitches", "sum"), swings=("swing", "sum"), whiffs=("whiff", "sum"), o_pitches=("o_pitch", "sum"), o_swings=("o_swing", "sum"))
    return g.join(q90, how="left").join(p, how="outer").fillna({"n": 0})


def features(T: Tables, year: int, asof, min_ros_pa: int = 30, population: str = "ros", horizon_days=None, half_life=None, window_days=None):
    """One row per hitter: YTD counts (PA before `asof`), prior-season pitch-level counts, history lists, and the ROS outcome (PA on/after `asof`)."""
    pa, pi = T.pa[T.pa["year"] == year], T.pi[T.pi["year"] == year]
    ytd_pa = pa[pa["date"] < asof]
    if window_days:                                       # only the last `window_days` days of this season count as in-season evidence for the RATES
        ytd_pa = ytd_pa[ytd_pa["date"] >= asof - pd.Timedelta(days=window_days)]
    w = (0.5 ** ((asof - ytd_pa["date"]).dt.days.to_numpy() / half_life)) if half_life else None      # exponential recency weights (half-life in days)
    ytd = _counts(ytd_pa, pi[pi["date"] < asof], w)
    ros_pa = pa[pa["date"] >= asof]
    if horizon_days:                                      # forecast only the next `horizon_days` days instead of the rest of the season
        ros_pa = ros_pa[ros_pa["date"] < asof + pd.Timedelta(days=horizon_days)]
    ros = ros_pa.groupby("name_clean").agg(n_ros=("woba", "size"), woba_ros=("woba", "mean"), k_ros=("k", "mean"), bb_ros=("bb", "mean"))
    if population == "ytd":          # production: everyone who has appeared so far this season; ROS columns are the realised outcome where it exists
        f = ytd.join(ros, how="left"); f = f[f["n"] > 0].copy(); f["n_ros"] = f["n_ros"].fillna(0)
    else:
        f = ros.join(ytd, how="left"); f = f[f["n_ros"] >= min_ros_pa].copy()
    for c in ("n", "woba_sum", "k", "bb", "hr", "bip", "hit", "xw_sum", "n_ev", "n_raw", "pitches", "swings", "whiffs", "o_pitches", "o_swings"):
        f[c] = f[c].fillna(0)
    # prior season's pitch-level counts (full season) and the pooled EV90
    prev = year - 1
    if prev >= 2025:
        pp, pq = T.pa[T.pa["year"] == prev], T.pi[T.pi["year"] == prev]; prior_c = _counts(pp, pq).add_prefix("p_"); f = f.join(prior_c, how="left")
        both = pd.concat([pa[pa["date"] < asof], pp]); f["ev90_pool"] = both.dropna(subset=["ev"]).groupby("name_clean")["ev"].quantile(0.9).reindex(f.index); f["n_ev_pool"] = both.groupby("name_clean")["ev"].count().reindex(f.index)
    else:
        f["ev90_pool"] = f["ev90"]; f["n_ev_pool"] = f["n_ev"]
    for c in [c for c in f.columns if c.startswith("p_")]:
        f[c] = f[c].fillna(0)
    for c in ("p_n", "p_xw_sum", "p_swings", "p_pitches", "p_whiffs", "p_o_swings", "p_o_pitches"):
        if c not in f.columns:
            f[c] = 0.0
    # history (panel seasons before `year`): per stat lists of (x, n, lg)
    for s, (delta, n0) in SPEC.items():
        den = bt.DEN[s]
        for j in (1, 2, 3):
            y = year - j
            if y >= 2022 and y in T.by:
                b = T.by[y]; x = b[s].reindex(f.index); n = b[den].reindex(f.index).fillna(0.0)
                f[f"h_{s}_x{j}"] = x.fillna(0.0); f[f"h_{s}_n{j}"] = np.where(x.isna(), 0.0, n); f[f"h_{s}_l{j}"] = T.lg[s].get(y, 0.0)
            else:
                f[f"h_{s}_x{j}"] = 0.0; f[f"h_{s}_n{j}"] = 0.0; f[f"h_{s}_l{j}"] = 0.0
    lg_now = {"woba": f["woba_sum"].sum() / f["n"].sum(), "k": f["k"].sum() / f["n"].sum(), "bbr": f["bb"].sum() / f["n"].sum(), "hrr": f["hr"].sum() / f["n"].sum(), "babip": f["hit"].sum() / f["bip"].sum()}
    return f, lg_now


def _ytd_rate(f, s):
    if s == "woba":
        return f["woba_sum"], f["n"]
    if s == "k":
        return f["k"], f["n"]
    if s == "bbr":
        return f["bb"], f["n"]
    if s == "hrr":
        return f["hr"], f["n"]
    return f["hit"], f["bip"]


def predict(f, lg_now, params=None, trackman=True, use_ytd=True, use_hist=True, nu=None, n0_scale=1.0, spec=None, max_hist=3):
    """Returns a DataFrame of ROS predictions: woba, k, bbr, hrr, babip (regressed), blend, and the final trackman-adjusted woba / k / bbr."""
    P = dict(PROD); P.update(params or {}); nu_in = nu; nu = P["nu"] if nu is None else nu
    out = pd.DataFrame(index=f.index)
    pa_eff = {}
    SPEC_ = spec or SPEC
    for s, (delta, n0) in SPEC_.items():
        num_cnt, den = _ytd_rate(f, s)
        x_ytd = np.where(den > 0, num_cnt / den.replace(0, np.nan), 0.0)
        nu_s = P["nu_bbr"] if (s == "bbr" and nu_in is None) else nu         # BB% has its own in-season weight unless nu was passed explicitly
        num = nu_s * den * (x_ytd - lg_now[s]) if use_ytd else 0.0 * den
        w_den = nu_s * den if use_ytd else 0.0 * den
        if use_hist:
            for j in range(1, max_hist + 1):
                wj = delta ** j
                num = num + wj * f[f"h_{s}_n{j}"] * (f[f"h_{s}_x{j}"] - f[f"h_{s}_l{j}"]) * (f[f"h_{s}_n{j}"] > 0)
                w_den = w_den + wj * f[f"h_{s}_n{j}"]
        out[s] = lg_now[s] + num / (w_den + n0 * n0_scale)
        pa_eff[s] = w_den
    comp = sum(BETA[s] * (out[s] - lg_now[s]) for s in BETA)
    out["blend"] = 0.5 * out["woba"] + 0.5 * (lg_now["woba"] + comp)
    out["pa_eff_woba"] = pa_eff["woba"]
    out["woba_final"], out["k_final"], out["bbr_final"] = out["blend"], out["k"], out["bbr"]
    if trackman:
        sc = P["trackman_scale"]
        # xwOBA nudge: pooled this-season + last-season (weight delta) xwOBA, shrunk
        n_x = f["n_raw"] + 0.75 * f["p_n"]; xw = (f["xw_sum"] + 0.75 * f["p_xw_sum"])
        lgx = (f["xw_sum"].sum() / f["n_raw"].sum())
        xs = (xw + P["n0x"] * lgx) / (n_x + P["n0x"])
        # EV90 term
        lg_ev = np.nanmean(f["ev90_pool"]) if f["ev90_pool"].notna().any() else 100.1
        ev_dev = np.where(f["ev90_pool"].isna(), 0.0, (f["ev90_pool"] - 100.1) * f["n_ev_pool"] / (f["n_ev_pool"] + P["n0_ev"]))
        fade_w = SPEC_["woba"][1] / (out["pa_eff_woba"] + SPEC_["woba"][1])
        out["woba_final"] = out["blend"] + sc * P["gamma"] * (xs - out["blend"]) + sc * P["b_ev"] * ev_dev * fade_w
        # Whiff% term for K%
        sw = f["swings"] + f["p_swings"]; wh = f["whiffs"] + f["p_whiffs"]; lgw = f["whiffs"].sum() / max(f["swings"].sum(), 1)
        wdev = np.where(sw > 0, (wh / sw.replace(0, np.nan) - lgw).fillna(0.0) * sw / (sw + P["n0_whiff"]), 0.0)
        fade_k = SPEC_["k"][1] / (out["pa_eff_woba"] * 0 + (nu * f["n"] + 0.75 * f["p_n"]) + SPEC_["k"][1])
        out["k_final"] = (out["k"] + sc * P["b_whiff"] * wdev * fade_k).clip(0.03, 0.55)
        # Swing% term for BB%
        pi_ = f["pitches"] + f["p_pitches"]; sp = f["swings"] + f["p_swings"]; lgs = f["swings"].sum() / max(f["pitches"].sum(), 1)
        sdev = np.where(pi_ > 0, (sp / pi_.replace(0, np.nan) - lgs).fillna(0.0) * pi_ / (pi_ + P["n0_swing"]), 0.0)
        fade_b = SPEC_["bbr"][1] / ((P["nu_bbr"] if nu_in is None else nu) * f["n"] + 0.5 * f["p_n"] + SPEC_["bbr"][1])
        out["bbr_final"] = (out["bbr"] + sc * P["b_swing"] * sdev * fade_b).clip(0.0, 0.4)
    return out
