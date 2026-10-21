"""
fl_error_analysis_common.py -- shared machinery for the projection ERROR ANALYSIS (user 2026-10-13: "deep dive error analysis on our projections looking for variables to improve the
model": (1) tendencies / traits / locations / circumstances correlated with the errors, (2) whether they are sticky and stable, (3) whether they improve the model overall and for
rookies / low-history players).  Used by fl_error_analysis_hitters.py and fl_error_analysis_pitchers.py.

PRE-SPECIFIED DESIGN (written before looking at any result):
  * Residual = actual - projected, out of sample, LEVEL-FREE (the sample's weighted mean residual removed: production recentres to the league level), frozen standard
    inverse-variance weights (fl_eval_weights.weights_from_base on the production prediction).
  * Candidate traits are measured ONLY in the feature window of each sample (before the outcome period), shrunk toward the population mean by their own reliability
    (n / (n + n0), n0 from the split-half reliability below).
  * STICKINESS: split-half reliability (alternating games within a season, players with >= MIN_HALF units in both halves), implied n0, and year-to-year correlation
    2025 -> 2026 disattenuated by the two seasons' reliabilities (~1 = a stable trait, low = the trait itself changes).
  * SCREEN: weighted regression residual ~ z(trait) + centred prediction + log(1 + history), player-bootstrap SE; Benjamini-Hochberg q within each (sample, target).
    Replication: 2025 and 2026 samples are disjoint in outcomes; a trait is SHORTLISTED only if the inverse-variance pooled z across the two seasons has BH q <= .10
    across all traits for that target, the two seasons agree in sign, split-half reliability >= .40 and disattenuated y/y >= .50.
  * IMPROVEMENT (shortlist only): new = production + b x (trait_shrunk - mean) x fade, fade = n0_target / (history + n0_target) (the form of the existing EV90 / Whiff% /
    Swing% priors; constant b reported as the alternative); b fit on one season, scored on the other (and transferred to the preseason sample); weighted RMSE and MAE
    with frozen weights, player bootstrap (1,000), overall, LOW-HISTORY (< 150 PA / BF of FL history incl. this season to date, rookies included) and ROOKIES (no prior FL season).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew

TRACK = "2025_2026_pitches.parquet"
COLS = ["game_id", "pa_id", "pa_pitch_count", "year", "date", "inning", "batter", "batter_side", "batter_team", "pitcher", "pitcher_id", "pitcher_team", "pitcher_throws",
        "balls", "strikes", "pitch_call", "play_result", "pitch_type", "effective_velo", "ivb", "horz_break", "spin_rate", "extension", "rel_height", "arm_angle",
        "vert_appr_angle", "outside_zone", "csoe", "batspeed", "exit_speed", "angle", "direction", "barrel", "xwoba", "oaa", "is_terminal_pitch", "k_or_bb", "game_type"]
BALL = {"ball_called", "ball_in_dirt", "ball_intentional", "automatic_ball"}
SWING = {"strike_swinging", "foul_not_fieldable", "foul_fieldable", "foul_ball", "in_play"}
FOUL = {"foul_not_fieldable", "foul_fieldable", "foul_ball"}
BIP = {"out", "single", "double", "triple", "error", "fielders_choice", "sacrifice", "home_run"}
FB_TYPES = {"four_seam", "sinker"}; BRK = {"slider", "curveball", "sweeper", "cutter", "knuckleball"}; OFF = {"changeup", "splitter"}
MIN_HALF = 30


def _clean(s):
    import fl_innings_projection as ip
    u = pd.Series(s.unique()); m = dict(zip(u, u.map(lambda x: ip.lastfirst_to_clean(x) if isinstance(x, str) else None))); return s.map(m)


def load_tracking():
    t = pd.read_parquet(TRACK, columns=COLS); t = t[t["game_type"] == "reg_season"].reset_index(drop=True)          # regular season only (user 2026-10-20)
    t["date"] = pd.to_datetime(t["date"]); t["bat"] = _clean(t["batter"]); t["pit"] = _clean(t["pitcher"]); t["pitcher_id"] = t["pitcher_id"].astype(str)
    pc = t["pitch_call"].astype(str)
    t["swing"] = pc.isin(SWING).astype(float); t["whiff"] = (pc == "strike_swinging").astype(float); t["foul"] = pc.isin(FOUL).astype(float)
    t["cstr"] = (pc == "strike_called").astype(float); t["strike"] = (~pc.isin(BALL | {"hit_by_pitch", "undefined"})).astype(float)
    t["oz"] = (t["outside_zone"] == 1).astype(float); t["iz"] = (t["outside_zone"] == 0).astype(float)
    t["first"] = ((t["balls"] == 0) & (t["strikes"] == 0)).astype(float); t["two"] = (t["strikes"] == 2).astype(float)
    t["take"] = pc.isin({"ball_called", "strike_called", "ball_in_dirt"}).astype(float)
    t["is_bip"] = t["play_result"].isin(BIP) & t["exit_speed"].notna() & t["angle"].notna() & (t["exit_speed"] <= 125)
    side = t["batter_side"].astype(str); thr = t["pitcher_throws"].astype(str)
    t["left_bat"] = (side == "left").astype(float); t["same_hand"] = ((side == thr) & side.isin(["left", "right"])).astype(float)
    t["hb_as"] = t["horz_break"] * np.where(thr == "left", -1.0, 1.0)
    d = t["direction"]; t["pull"] = (((side == "right") & (d < -15)) | ((side == "left") & (d > 15))).astype(float); t["oppo"] = (((side == "right") & (d > 15)) | ((side == "left") & (d < -15))).astype(float)
    return t


def _ratio(g, num, den, mask=None):
    """per-player ratio sum(num)/sum(den) over rows where mask; returns (value, n=sum(den))."""
    x = g if mask is None else g[mask]
    a = x.groupby("key")[[num, den]].sum() if den != "_one" else x.assign(_one=1.0).groupby("key")[[num, "_one"]].sum()
    return a.iloc[:, 0] / a.iloc[:, 1].replace(0, np.nan), a.iloc[:, 1]


def _mean(g, col, mask=None):
    x = g if mask is None else g[mask]; x = x[x[col].notna()]
    a = x.groupby("key")[col].agg(["mean", "size"]); return a["mean"], a["size"].astype(float)


def hitter_traits(t):
    """t = tracking rows of the window, with t['key'] = hitter.  Returns {trait: (value Series, n Series)}."""
    g = t; bip = g[g["is_bip"]].copy(); fb = g[g["pitch_type"].isin(FB_TYPES)]
    sw = g[g["swing"] == 1]; tw = g[g["two"] == 1]
    out = {}
    out["swing_pct"] = _ratio(g, "swing", "_one")
    out["chase_pct"] = _ratio(g.assign(osw=g["swing"] * g["oz"]), "osw", "oz")
    out["zswing_pct"] = _ratio(g.assign(zsw=g["swing"] * g["iz"]), "zsw", "iz")
    zs = g[(g["iz"] == 1) & (g["swing"] == 1)]; out["zcontact_pct"] = _ratio(zs.assign(c=1 - zs["whiff"]), "c", "_one")
    out["whiff_pct"] = _ratio(sw, "whiff", "_one")
    out["first_pitch_swing"] = _ratio(g[g["first"] == 1], "swing", "_one")
    out["two_strike_whiff"] = _ratio(tw[tw["swing"] == 1], "whiff", "_one")
    out["foul_per_swing"] = _ratio(sw, "foul", "_one")
    out["called_strike_pct"] = _ratio(g, "cstr", "_one")
    out["bat_speed"] = _mean(sw, "batspeed")
    out["breaking_seen"] = _ratio(g.assign(b=g["pitch_type"].isin(BRK).astype(float)), "b", "_one")
    out["fb_velo_faced"] = _mean(fb, "effective_velo")
    out["same_hand_share"] = _ratio(g, "same_hand", "_one")
    bip["hard"] = (bip["exit_speed"] >= 95).astype(float); bip["ss"] = bip["angle"].between(8, 32).astype(float)
    bip["gb"] = (bip["angle"] < 10).astype(float); bip["ld"] = bip["angle"].between(10, 25, inclusive="left").astype(float); bip["fbb"] = bip["angle"].between(25, 50, inclusive="left").astype(float); bip["pu"] = (bip["angle"] >= 50).astype(float)
    bip["pull_air"] = bip["pull"] * (bip["angle"] >= 10); bip["hr"] = (bip["play_result"] == "home_run").astype(float); bip["air"] = (bip["angle"] >= 10).astype(float)
    out["avg_ev"] = _mean(bip, "exit_speed"); out["hardhit_pct"] = _ratio(bip, "hard", "_one"); out["barrel_pct"] = _ratio(bip.assign(br=bip["barrel"].fillna(0)), "br", "_one")
    out["sweetspot_pct"] = _ratio(bip, "ss", "_one"); out["la_mean"] = _mean(bip, "angle")
    sd = bip.groupby("key")["angle"].agg(["std", "size"]); out["la_sd"] = (sd["std"], sd["size"].astype(float))
    out["gb_pct"] = _ratio(bip, "gb", "_one"); out["ld_pct"] = _ratio(bip, "ld", "_one"); out["fb_pct"] = _ratio(bip, "fbb", "_one"); out["pu_pct"] = _ratio(bip, "pu", "_one")
    out["pull_pct"] = _ratio(bip, "pull", "_one"); out["oppo_pct"] = _ratio(bip, "oppo", "_one"); out["pull_air_pct"] = _ratio(bip, "pull_air", "_one")
    out["xwobacon"] = _mean(bip, "xwoba"); out["hr_per_air"] = _ratio(bip, "hr", "air")
    nh = bip[bip["hr"] == 0].assign(hit=bip["play_result"].isin(["single", "double", "triple"]).astype(float)); out["babip_tracked"] = _ratio(nh, "hit", "_one")
    return out


def pitcher_traits(t):
    """t = tracking rows of the window, with t['key'] = pitcher."""
    g = t; fb = g[g["pitch_type"].isin(FB_TYPES)]; bip = g[g["is_bip"]].copy(); out = {}
    out["fb_velo"] = _mean(fb, "effective_velo"); out["fb_ivb"] = _mean(fb, "ivb"); out["fb_hb_armside"] = _mean(fb, "hb_as"); out["fb_vaa"] = _mean(fb, "vert_appr_angle")
    out["fb_spin"] = _mean(fb, "spin_rate"); out["extension"] = _mean(g, "extension"); out["rel_height"] = _mean(g, "rel_height"); out["arm_angle"] = _mean(g, "arm_angle")
    pt = g.assign(fbx=g["pitch_type"].isin(FB_TYPES).astype(float), brk=g["pitch_type"].isin(BRK).astype(float), off=g["pitch_type"].isin(OFF).astype(float))
    out["fb_usage"] = _ratio(pt, "fbx", "_one"); out["breaking_usage"] = _ratio(pt, "brk", "_one"); out["offspeed_usage"] = _ratio(pt, "off", "_one")
    sh = g.groupby(["key", "pitch_type"]).size().unstack(fill_value=0); shr = sh.div(sh.sum(1), axis=0); out["n_pitch_types"] = ((shr >= 0.10).sum(1).astype(float), sh.sum(1).astype(float))
    vo = g[g["pitch_type"].isin(OFF)].groupby("key")["effective_velo"].mean(); vf = fb.groupby("key")["effective_velo"].mean(); gap = (vf - vo).dropna()
    out["velo_gap_fb_off"] = (gap, g[g["pitch_type"].isin(OFF)].groupby("key").size().reindex(gap.index).astype(float))
    out["zone_pct"] = _ratio(g, "iz", "_one"); out["first_pitch_strike"] = _ratio(g[g["first"] == 1], "strike", "_one")
    out["csoe_takes"] = _mean(g, "csoe"); out["chase_induced"] = _ratio(g.assign(osw=g["swing"] * g["oz"]), "osw", "oz")
    zs = g[(g["iz"] == 1) & (g["swing"] == 1)]; out["zcontact_allowed"] = _ratio(zs.assign(c=1 - zs["whiff"]), "c", "_one")
    out["swstr_pct"] = _ratio(g, "whiff", "_one"); out["called_strike_pct"] = _ratio(g, "cstr", "_one")
    tw = g[g["two"] == 1]; tw = tw.assign(kk=((tw["k_or_bb"].astype(str) == "strikeout") & tw["is_terminal_pitch"].astype(str).isin(["1", "True", "true"])).astype(float))
    out["two_strike_putaway"] = _ratio(tw, "kk", "_one")
    bip["hard"] = (bip["exit_speed"] >= 95).astype(float); bip["gb"] = (bip["angle"] < 10).astype(float); bip["fbb"] = bip["angle"].between(25, 50, inclusive="left").astype(float)
    bip["pu"] = (bip["angle"] >= 50).astype(float); bip["hr"] = (bip["play_result"] == "home_run").astype(float); bip["air"] = (bip["angle"] >= 10).astype(float)
    out["gb_allowed"] = _ratio(bip, "gb", "_one"); out["fb_allowed"] = _ratio(bip, "fbb", "_one"); out["pu_allowed"] = _ratio(bip, "pu", "_one")
    out["ev_allowed"] = _mean(bip, "exit_speed"); out["hardhit_allowed"] = _ratio(bip, "hard", "_one"); out["barrel_allowed"] = _ratio(bip.assign(br=bip["barrel"].fillna(0)), "br", "_one")
    out["xwobacon_allowed"] = _mean(bip, "xwoba"); out["hr_per_air_allowed"] = _ratio(bip, "hr", "air")
    nh = bip[bip["hr"] == 0].assign(hit=bip["play_result"].isin(["single", "double", "triple"]).astype(float)); out["babip_allowed"] = _ratio(nh, "hit", "_one")
    out["lhb_share"] = _ratio(g, "left_bat", "_one")
    term = g[g["is_terminal_pitch"].astype(str).isin(["1", "True", "true"])].copy()
    gp = term.groupby(["key", "game_id"]).size(); bfg = gp.groupby("key").agg(["mean", "size"]); out["bf_per_game"] = (bfg["mean"], bfg["size"].astype(float))
    first = term.sort_values(["game_id", "pa_id"]).groupby(["game_id", "pitcher_team"]).head(1)[["game_id", "key"]].assign(start=1.0)
    gm = term[["key", "game_id"]].drop_duplicates().merge(first, on=["game_id", "key"], how="left").fillna({"start": 0.0}); out["start_share"] = _ratio(gm, "start", "_one")
    term["bf_idx"] = term.sort_values(["game_id", "pa_id"]).groupby(["key", "game_id"]).cumcount(); out["tto3_share"] = _ratio(term.assign(t3=(term["bf_idx"] >= 18).astype(float)), "t3", "_one")
    of = g[g["oaa"].notna()]; team = of.groupby(["pitcher_team"])["oaa"].mean(); gg = g.drop_duplicates(["key", "game_id"])[["key", "pitcher_team"]]
    tv = gg.assign(d=gg["pitcher_team"].map(team)).groupby("key")["d"].agg(["mean", "size"]); out["team_of_defense"] = (tv["mean"], tv["size"].astype(float) * 20)
    return out


def build(t, kind):
    return hitter_traits(t) if kind == "h" else pitcher_traits(t)


def stickiness(t, kind):
    """Split-half (alternating games) reliability per season, implied n0, and disattenuated year-to-year correlation.  Returns DataFrame indexed by trait."""
    rows = {}
    rel = {}
    for yr in (2025, 2026):
        ty = t[t["year"] == yr]; gm = ty[["key", "game_id", "date"]].drop_duplicates(["key", "game_id"]).sort_values("date")
        gm["half"] = gm.groupby("key").cumcount() % 2; ty = ty.merge(gm[["key", "game_id", "half"]], on=["key", "game_id"])
        A, B = build(ty[ty["half"] == 0], kind), build(ty[ty["half"] == 1], kind)
        for k in A:
            va, na = A[k]; vb, nb = B[k]; d = pd.DataFrame({"a": va, "na": na, "b": vb, "nb": nb}).dropna(); d = d[(d["na"] >= MIN_HALF) & (d["nb"] >= MIN_HALF)]
            if len(d) < 30:
                continue
            r = np.corrcoef(d["a"], d["b"])[0, 1]; nh = float(((d["na"] + d["nb"]) / 2).mean()); n0 = nh * (1 - r) / r if r > 0.02 else np.inf
            rel.setdefault(k, []).append((r, nh, n0, len(d)))
    full = {yr: build(t[t["year"] == yr], kind) for yr in (2025, 2026)}
    for k, lst in rel.items():
        r = float(np.mean([x[0] for x in lst])); n0 = float(np.nanmedian([x[2] for x in lst])); nh = float(np.mean([x[1] for x in lst]))
        v5, n5 = full[2025][k]; v6, n6 = full[2026][k]; d = pd.DataFrame({"a": v5, "na": n5, "b": v6, "nb": n6}).dropna(); d = d[(d["na"] >= 2 * MIN_HALF) & (d["nb"] >= 2 * MIN_HALF)]
        ryy = np.corrcoef(d["a"], d["b"])[0, 1] if len(d) >= 30 else np.nan
        ra = np.mean(d["na"] / (d["na"] + n0)) if np.isfinite(n0) else 0; rb = np.mean(d["nb"] / (d["nb"] + n0)) if np.isfinite(n0) else 0
        rows[k] = {"split-half r": r, "units per half": nh, "n0": n0, "y/y r": ryy, "y/y n": len(d), "y/y disattenuated": ryy / np.sqrt(ra * rb) if ra > 0 and rb > 0 else np.nan}
    return pd.DataFrame(rows).T


def shrink(v, n, n0):
    d = pd.DataFrame({"v": v, "n": n}).dropna()
    if d.empty or not np.isfinite(n0):
        return pd.Series(dtype=float)
    mu = np.average(d["v"], weights=d["n"]); return mu + (d["v"] - mu) * d["n"] / (d["n"] + n0)


def standardise(x, w=None):
    x = pd.Series(x, dtype=float); m = np.average(x, weights=w) if w is not None else x.mean(); s = np.sqrt(np.average((x - m) ** 2, weights=w)) if w is not None else x.std()
    return (x - m) / s if s > 0 else x * 0


def screen(S, traits, n0s, B=400, seed=1, min_cov=0.3):
    """S: DataFrame indexed by player key with columns e (level-free residual), w (weights), pc (centred prediction), lh (log1p history).  traits: {name: (value, n)}.
    Returns DataFrame: trait, n_with, coef per SD, se (bootstrap), t, r (weighted corr), p (normal), plus the share of rows with the trait."""
    rng = np.random.default_rng(seed); out = []
    for k, (v, n) in traits.items():
        xs = shrink(v, n, n0s.get(k, np.inf)); d = S.join(xs.rename("x"), how="left")
        mu = np.average(d["x"].dropna(), weights=d.loc[d["x"].notna(), "w"]) if d["x"].notna().any() else 0.0; has = d["x"].notna().mean(); d["x"] = d["x"].fillna(mu)
        if d["x"].std() == 0 or has < min_cov:
            continue
        d["z"] = standardise(d["x"], d["w"]); X = np.column_stack([np.ones(len(d)), d["z"], d["pc"], d["lh"]]); y = d["e"].to_numpy(); w = d["w"].to_numpy()

        def fit(ii):
            Xi, yi, wi = X[ii], y[ii], w[ii]; return np.linalg.lstsq(Xi * np.sqrt(wi)[:, None], yi * np.sqrt(wi), rcond=None)[0][1]
        b = fit(np.arange(len(d))); bs = np.array([fit(rng.integers(0, len(d), len(d))) for _ in range(B)]); se = bs.std()
        r = np.average((d["z"] - np.average(d["z"], weights=w)) * (y - np.average(y, weights=w)), weights=w) / np.sqrt(np.average((y - np.average(y, weights=w)) ** 2, weights=w))
        from scipy.stats import norm
        out.append({"trait": k, "coverage": has, "coef per SD": b, "se": se, "t": b / se, "r": r, "p": 2 * norm.sf(abs(b / se))})
    return pd.DataFrame(out, columns=["trait", "coverage", "coef per SD", "se", "t", "r", "p"]).set_index("trait")


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q


def improvement(train, test, x_tr, x_te, n0_target, fade=True, B=1000, seed=3):
    """train / test: DataFrames with e, w, hist (and rookie flag in test).  x_*: shrunk trait (aligned, NaN -> mean).  Fit b on train (weighted LS of e on the term),
    apply to test.  Returns dict of metrics (overall / low-history / rookies) and b."""
    def term(df, x):
        mu = np.average(x, weights=df["w"]); f = n0_target / (df["hist"] + n0_target) if fade else 1.0; return (x - mu) * f
    ttr = term(train, x_tr.to_numpy()); b = float(np.sum(train["w"] * ttr * train["e"]) / np.sum(train["w"] * ttr ** 2))
    tte = term(test, x_te.to_numpy()); e0 = test["e"].to_numpy(); e1 = e0 - b * tte; w = test["w"].to_numpy()
    res = {"b": b}
    for lab, m in (("all", np.ones(len(test), bool)), ("low-history", (test["hist"] < 150).to_numpy()), ("rookies", test["rookie"].to_numpy().astype(bool))):
        if m.sum() < 15:
            continue
        y0 = e0[m] - np.average(e0[m], weights=w[m]); y1 = e1[m] - np.average(e1[m], weights=w[m])
        rep = er.compare(np.zeros(m.sum()), -y0, -y1, w[m], test.index.to_numpy()[m], n=B)
        res[lab] = {"n": int(m.sum()), "RMSE old": rep.get("RMSE old", np.sqrt(np.average(y0 ** 2, weights=w[m]))), "RMSE new": rep.get("RMSE new", np.sqrt(np.average(y1 ** 2, weights=w[m]))),
                    "dRMSE": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "MAE old": np.average(np.abs(y0), weights=w[m]),
                    "MAE new": np.average(np.abs(y1), weights=w[m]), "dMAE": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]}
    return res


def frame(e, pred, n, hist, rookie, idx):
    """Residual frame: level-free residual, frozen standard weights, centred prediction, log history."""
    e = np.asarray(e, float); pred = np.asarray(pred, float); n = np.asarray(n, float)
    w = ew.weights_from_base(pred + e, pred, n)[0]; ec = e - np.average(e, weights=w)
    return pd.DataFrame({"e": ec, "w": w, "pc": pred - np.average(pred, weights=w), "lh": np.log1p(np.asarray(hist, float)), "hist": np.asarray(hist, float),
                         "rookie": np.asarray(rookie, bool), "n": n}, index=pd.Index(idx))
