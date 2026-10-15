"""
fl_hitter_projection.py -- 2027 Frontier League HITTER projections (production pipeline).

What a hitter gets (everything park-neutral unless it says Mississippi):
  K%, BB%, HR/PA (park-neutral)   regression to the league level (Marcel-style, three seasons at weights 1 / delta / delta^2), constants fit leave-one-
                                  transition-out; K% + Whiff% prior, BB% + Swing% prior (both fade with PA)
  wOBA                            50/50 blend of (a) the same regression applied to wOBA itself and (b) wOBA rebuilt from the regressed K%, BB%, HR/PA;
                                  plus gamma x (shrunk xwOBA - blend) for contact quality and an EV90 power prior (fades with PA)
  PA                              per team game: a + b x PA_t/G_t + c x projected wOBA quality, at 102 games, if he plays
  WAR parts                       PRODUCTION: batting (wOBA over league / wOBA scale) + position (shrunk own rate toward his position's) + replacement (user 2026-10-13)
                                  + baserunning = his position's average incl. DH (fl_bsr_proxy.py, user 2026-10-14).  Fielding (OF / C only) is computed and stored but
                                  left out (WAR_INCLUDE_FIELDING flag).
  WAR distribution                Monte Carlo over PA, wOBA error (a + b/PA law), fielding, position and baserunning error; playing time CORRELATED with
                                  the season's performance (rho, fl_informative_pt_test.py: bad seasons lose PA)
  Mississippi                     POST-PROJECTION, half his PA at TrustmarkPark: wOBA surface (exit speed x launch angle x side, the surface's own level) + the
                                  wOBA consequence of the global K% / BB% effects; MIS K% / BB% / HR/PA alongside; "park fit" WAR = his shift minus the average hitter's.
                                  History is neutralised with the same effects (wOBA, HR/PA by the HR surface, K%, BB%) for the Trustmark share of his PA

Pre-2025 seasons come from StatsCrew batting lines (no Trackman, no WAR parts); 2025-26 from Supabase. Park-neutralisation only
removes the TrustmarkPark share of a hitter's history (Mississippi hitters ~48% of PA, visitors ~3%); other parks are left alone.

Usage:  python fl_hitter_projection.py [--publish]     (--publish also copies the outputs to ../projections-dashboard/data)
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dev
import fl_hitter_ev as hev
import fl_hitter_whiff as hwf
import fl_hitter_swing as hsw
import fl_hitter_depth3_test as d3
import fl_hitter_park as park
import fl_bsr_proxy as bsr_proxy

warnings.filterwarnings("ignore")

OUT_PROJ = "output/fl_2027_hitter_projections.parquet"
OUT_HIST = "output/fl_hitter_history.parquet"
OUT_COEF = "output/fl_hitter_model_coefs.json"
DASH_DATA = "../projections-dashboard/data"

import fl_season_config as cfg
TARGET_YEAR = cfg.TARGET_YEAR
LATEST = cfg.LATEST_SEASON
# regular-season length by year (fl_season_config: 2025 = 96, 2026 = 102, 2027 = 102; 2022-24 taken as 96: max games played by a hitter on a team was 96 each year)
GAMES = {y: g for y, g in cfg.GAMES.items() if y <= cfg.LATEST_SEASON}
TARGET_GAMES = cfg.TARGET_GAMES
SC = park.WOBA_SCALE
# per-stat: weight on the prior season (delta), gap multiplier on n0 for hitters whose last season is 2025 (n=19 pairs: noisy)
SPEC = {"woba": (0.75, 2.0), "k": (0.75, 1.0), "bbr": (0.5, 1.7), "hrr": (0.5, 2.0), "babip": (0.5, 1.0), "hbpr": (0.5, 1.0)}
# xwOBA add-on (2025 -> 2026 only, n=111, 10-fold CV, standard inverse-variance weights: RMSE .0369 -> .0360; gamma .57 at n0x 200; was .63 under the
# old n/(n+150) placeholder weights). Provenance: fl_hitter_model_dev.py part D.  0.485 is the JOINT fit with the EV90 term below (fl_hitter_ev90_fit.py, SE .12).
# EV90 "power prior" (user decision 2026-10-07; provisional until 2027 data confirm): + B_EV90 x (shrunk EV90 - league EV90, mph) x n0w / (PA_eff + n0w), where PA_eff = PA_t + delta x PA_{t-1}
# and n0w = 440 (880 for hitters with no 2026 PA).  Fit within-season (369 hitter-seasons, 34,650 draws, m = 30-200 PA): .00377 wOBA per mph (SE .00080), stable across m (.0035-.0041); one SD of
# hitter EV90 (3.2 mph) = +.012 wOBA before the fade.  The year-to-year tests could not confirm it (n ~ 100, coefficient +.003 to +.006 per SD, no RMSE gain), see DECISION_LOG.md.
B_EV90 = 0.00377
EV90_DELTA = 0.75
# Whiff% "contact-skill prior" for K% (user: update the K% model; provisional like EV90): proj_K += B_WHIFF x (shrunk Whiff% - league Whiff%) x n0k / (PA_eff + n0k), n0k = the K% regression constant (79.3).
# Fit within-season (375 hitter-seasons, 35,275 draws, m = 30-200 PA, fair intercept): B = 0.605 K% per unit of Whiff% (SE .046; by m .52-.63); one SD of hitter Whiff% (.069) = +4.2 K% points at zero
# sample.  Within-season K% RMSE gain 2-7% in thin samples (m = 30-120), 2% at 200 PA; year-to-year (2025 -> 2026, reverse) could NOT confirm it (forward -.0003 faded / -.0012 unfaded, interval
# includes 0; reverse +.0008 faded, slightly worse) -- see DECISION_LOG.md.
B_WHIFF = 0.605
# Swing% "plate-discipline prior" for BB% (user go-ahead 2026-10-08): proj_BB += B_SWING x (shrunk Swing% - league Swing%) x n0bb / (PA_eff + n0bb), n0bb = the BB% regression constant (146.6), Swing% shrunk by n0 = 92.1 pitches.
# Fit within-season (fl_hitter_bb_study.py): B = -0.387 BB% per unit of Swing% (SE .030); forward 2025 -> 2026 (this path, production constants, term from 2025 pitches only): BB% RMSE -.0010 [-.0021, -.0001], P(better) .96
# (fl_hitter_swing_preseason_check.py).  Feeds the wOBA components like the Whiff term (accuracy-neutral on wOBA: -.00002).
B_SWING = -0.387
GAMMA_XWOBA, N0_XWOBA = 0.485, 200.0
# wOBA error law Var = a + b/PA (fl_hitter_model_dev.py part E; 4 held-out transitions, standardised SD 1.00, coverage .50/.80/.91)
LUCK_A, LUCK_B = 0.000266, 0.4545
# fielding: own per-PA rate x PA/(PA + 200) (10-fold optimum, slope ~1); residual SD 6.2 runs / 600 PA. Position: shrink own rate toward his position's.
FLD_N0, FLD_RESID_PER_PA = 200.0, 6.2 / 600
# infield fielding is not measured (2025: none; 2026: 0 for ~70-90% of infield regulars, the rest tiny = innings at OF / C).  User 2026-10-12: keep OF / C fielding
# (dropping it costs WAR RMSE +.11, fl_fielding_drop_report.py), infielders get 0 with a flag and the realised season spread of MEASURED outfielders as their
# fielding uncertainty (computed in build_projections from 2026 OF regulars, ~12 runs / 600 PA) instead of the 6.2 residual of a measured player.
FLD_MEASURED_POS = ("LF", "CF", "RF", "C")
POS_N0, POS_RESID_PER_PA = 100.0, 6.0 / 600
# WAR COMPONENTS IN PRODUCTION: batting + positional adjustment + replacement (user 2026-10-13) + BASERUNNING PROXY (user 2026-10-14).  Fielding stays out (infield not
# measured; outfield double-counts CF difficulty): fld_rate is computed and stored only.  Baserunning = the position average incl. DH (>= 50% of PA at DH) from 2026
# war_players.bsr_runs (fl_bsr_proxy.py), NOT the player's own record; WAR backtest -> 2026: RMSE 1.374 -> 1.348, MAE 1.124 -> 1.098 (fl_bsr_proxy_validation.py).
WAR_INCLUDE_FIELDING = False
WAR_INCLUDE_BASERUNNING = True
GAP_PA, GAP_P_RETURN = 166.0, 0.065         # hitters whose last season is 2025: median PA of the 26 who came back (of 402), return rate
N_SIM = 3000
# informative playing time (user 2026-10-12): corr(standardised season wOBA error, PA residual) in the 2025 -> 2026 backtest, 2-fold cross-fit (fl_informative_pt_test.py,
# n = 156): Brier of P(WAR>0) .1877 -> .1857 (P(better) .98), WAR RMSE 1.201 -> 1.188, MAE .979 -> .970.  Marginals of PA and wOBA error unchanged.
RHO_PT = 0.223
# K% BRIDGE PRIOR (user 2026-10-13, fl_hitter_k_bridge_prior_validation.py): hitters who debuted in the FL in 2023+ with a MiLB record before the debut shrink their K% regression toward
# their MiLB bridge prediction instead of the league mean: + n0_k x bridge_dev / (history weight + n0_k).  Validation (56 transitions, 45 hitters, LOTO n0, out-of-fold bridges):
# RMSE .04636 -> .04274 (P(better) .998), MAE P .996, all three transitions, permutation p .002, full production path 2025 -> 2026 P .993.  Switched off in backtest builds
# (fl_hitter_pre2026.build_shifted), whose shifted years do not match the real debut years.
K_BRIDGE_PRIOR = True


def k_bridge_dev():
    """Series name_clean -> bridge prediction of the FL K% deviation from the league, from MiLB seasons before the hitter's FL debut (published K% bridge coefficients)."""
    import fl_hitter_bridge_test as hb
    c = json.load(open("output/fl_hitter_bridge_coefs.json"))["coefs"]["k"]
    m = hb.milb_table(); d = hb.fl_debuts()
    U, has = hb.features(d, m, c["n0"], c["decay"], "k"); u = np.clip(U.sum(axis=1), c["u_min"], c["u_max"])
    dev = pd.Series(c["intercept"] + c["slope"] * u, index=d["nm"].to_numpy())[has]
    names = pd.read_parquet("output/fl_hitter_panel.parquet")["name_clean"].drop_duplicates()
    return names.set_axis(names.map(hb.norm)).pipe(lambda x: pd.Series(dev.reindex(x.index).to_numpy(), index=x.to_numpy())).dropna()
MIS = "Mississippi Mud Monsters"


# --------------------------------------------------------------------------- #
#  fitting
# --------------------------------------------------------------------------- #
def league_levels(q):
    return {s: bt.league(q, s) for s in SPEC if s in q.columns}


def fit_all(p):
    """n0 per stat, component->wOBA weights, PA model. Returns a dict of frozen parameters."""
    q = p.copy(); q["woba"] = q["woba_n"]
    out = {"n0": {}, "n0_loto": {}, "rmse": {}}
    loto_pred = {}
    for s, (delta, _) in SPEC.items():
        d = d3.frame(q, s)                                          # 3-season history: latest + previous (delta) + the one before (delta^2); fl_hitter_depth3_test.py
        out["n0"][s] = float(d3.fit_n0(d, delta, 3))
        Ls, n0s = [], {}
        for t in sorted(d["t"].unique()):
            tr, te = d[d["t"] != t], d[d["t"] == t]; n0s[t] = d3.fit_n0(tr, delta, 3); Ls.append(te.assign(pred=d3.pred(te, n0s[t], delta, 3)))
        E = pd.concat(Ls)
        out["n0_loto"][s] = [float(min(n0s.values())), float(max(n0s.values()))]
        out["rmse"][s] = {"league_mean": bt.rmse(d, d["lg_next"]), "last_year": bt.rmse(d, d["x_t"]), "model": bt.rmse(E, E["pred"])}
        out.setdefault("law", {})[s] = {"a": float(d["law_a"].iloc[0]), "b": float(d["law_b"].iloc[0])}
        loto_pred[s] = E.set_index(["name_clean", "t"])
    # component weights: next-year wOBA deviation on the DEVIATIONS of the regressed K%, BB%, HR/PA, BABIP predictions from their league levels
    comp = ["k", "bbr", "hrr"]                                     # BABIP dropped from the components 2026-10-08 (fl_hitter_babip_ablation.py: blend RMSE -.0002, MAE -.00025 without it)
    W = loto_pred["woba"][["x_next", "n_next", "lg_next", "w", "pred"]].rename(columns={"pred": "woba_pred"})
    for s in comp:
        W = W.join(loto_pred[s][["pred", "lg_next"]].rename(columns={"pred": f"{s}_pred", "lg_next": f"{s}_lg"}), how="inner")
    W = W.dropna()
    y = (W["x_next"] - W["lg_next"]).to_numpy()
    X = np.column_stack([(W[f"{s}_pred"] - W[f"{s}_lg"]).to_numpy() for s in comp])
    w = W["w"].to_numpy()                                   # standard inverse-variance weights (fl_eval_weights), frozen per stat
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    out["comp_beta"] = dict(zip(comp, map(float, beta)))
    pred_c = X @ beta
    out["rmse"]["woba_components"] = float(np.sqrt(np.average((y - pred_c) ** 2, weights=w)))
    out["rmse"]["woba_direct_same_rows"] = float(np.sqrt(np.average((y - (W["woba_pred"] - W["lg_next"]).to_numpy()) ** 2, weights=w)))
    out["rmse"]["woba_blend_same_rows"] = float(np.sqrt(np.average((y - 0.5 * (pred_c + (W["woba_pred"] - W["lg_next"]).to_numpy())) ** 2, weights=w)))
    out["n_component_rows"] = int(len(W))
    # playing time
    d = dev.pa_model(p); r = d.dropna(subset=["pa_next"]).copy()
    # season-length-aware playing time (2026-10-08): PA per team game, projected at TARGET_GAMES (fl_hitter_pa_games_test.py: LOTO RMSE -1.1 PA, MAE -0.7)
    r["x_pg"] = r["pa_t"] / r["t"].map(GAMES); r["y_pg"] = r["pa_next"] / r["t"].map(lambda t: GAMES[t + 1])
    # quality-aware (2026-10-08, fl_hitter_pa_quality_test.py): better hitters play more.  q = his projected wOBA deviation from the direct three-season regression (all returners, n0 = the production wOBA n0)
    saved = bt.MIN_NEXT; bt.MIN_NEXT = 1; fr = d3.frame(q, "woba"); bt.MIN_NEXT = saved
    fr["q"] = d3.pred(fr, out["n0"]["woba"], SPEC["woba"][0], 3) - fr["lg_next"]; r = r.merge(fr[["name_clean", "t", "q"]], on=["name_clean", "t"], how="inner")
    m = smf.ols("y_pg ~ x_pg + q", data=r).fit(); ml = smf.logit("ret ~ lnpa", data=d).fit(disp=0)
    out["pa"] = {"a": float(m.params["Intercept"]), "b": float(m.params["x_pg"]), "c_quality": float(m.params["q"]), "resid_sd": float(np.sqrt(m.scale)) * TARGET_GAMES, "per_game": True, "target_games": TARGET_GAMES,
                 "ret_a": float(ml.params["Intercept"]), "ret_b": float(ml.params["lnpa"])}
    return out


# --------------------------------------------------------------------------- #
#  projection
# --------------------------------------------------------------------------- #
def reg_rate(x_t, n_t, x_p, n_p, lg_t, lg_p, lg_next, n0, delta, x_pp=np.nan, n_pp=0.0, lg_pp=np.nan):
    """lg_next + (n_t dev_t + delta n_p dev_p + delta^2 n_pp dev_pp) / (n_t + delta n_p + delta^2 n_pp + n0); a missing season -> weight 0."""
    n_p = np.where(np.isnan(x_p), 0.0, n_p)
    dev_p = np.where(np.isnan(x_p), 0.0, x_p - lg_p)
    x_pp = np.asarray(x_pp, dtype=float); n_pp = np.where(np.isnan(x_pp) | np.isnan(lg_pp), 0.0, n_pp); dev_pp = np.where(np.isnan(x_pp) | np.isnan(lg_pp), 0.0, x_pp - lg_pp)
    return lg_next + (n_t * (x_t - lg_t) + delta * n_p * dev_p + delta ** 2 * n_pp * dev_pp) / (n_t + delta * n_p + delta ** 2 * n_pp + n0)


def build_projections(p, fit):
    q = p.copy(); q["woba"] = q["woba_n"]
    lg = {s: bt.league(q if s != "hrr" else q, s) for s in SPEC}
    t26 = p[p["year"] == 2026].set_index("name_clean"); t25 = p[p["year"] == 2025].set_index("name_clean"); t24 = p[p["year"] == 2024].set_index("name_clean")
    qi = q.set_index(["year", "name_clean"])
    names = sorted(set(t26.index) | set(t25.index))
    rows = []
    for nm in names:
        has26 = nm in t26.index
        yr = 2026 if has26 else 2025
        prev_yr = yr - 1
        cur = qi.loc[(yr, nm)]; cur = cur.iloc[0] if isinstance(cur, pd.DataFrame) else cur
        prv = qi.loc[(prev_yr, nm)] if (prev_yr, nm) in qi.index else None
        if isinstance(prv, pd.DataFrame):
            prv = prv.iloc[0]
        prv2 = qi.loc[(prev_yr - 1, nm)] if (prev_yr - 1, nm) in qi.index else None
        if isinstance(prv2, pd.DataFrame):
            prv2 = prv2.iloc[0]
        rows.append((nm, yr, cur, prv, prv2))
    recs = []
    for nm, yr, cur, prv, prv2 in rows:
        gap = (yr == 2025)
        r = {"name_clean": nm, "last_season": yr, "gap": gap}
        for s, (delta, gm) in SPEC.items():
            n0 = fit["n0"][s] * (gm if gap else 1.0)
            den = bt.DEN[s]
            x_t, n_t = cur[s], cur[den]
            x_p = prv[s] if prv is not None else np.nan
            n_p = prv[den] if prv is not None else 0.0
            if pd.isna(x_t) or pd.isna(n_t) or n_t <= 0:
                r[f"proj_{s}"] = np.nan; continue
            x_pp = prv2[s] if prv2 is not None else np.nan
            n_pp = prv2[den] if prv2 is not None else 0.0
            if s == "k":
                r["k_nh"] = float(n_t) + delta * (float(n_p) if pd.notna(x_p) and pd.notna(n_p) else 0.0) + delta ** 2 * (float(n_pp) if pd.notna(x_pp) and pd.notna(n_pp) and pd.notna(lg[s].get(prev_yr - 1, np.nan)) else 0.0); r["k_n0"] = n0
            r[f"proj_{s}"] = float(reg_rate(x_t, float(n_t), np.array(x_p, dtype=float), float(n_p) if pd.notna(n_p) else 0.0,
                                           lg[s][yr], lg[s].get(prev_yr, np.nan), lg[s][LATEST], n0, delta,
                                           x_pp=np.array(x_pp, dtype=float), n_pp=float(n_pp) if pd.notna(n_pp) else 0.0, lg_pp=lg[s].get(prev_yr - 1, np.nan)))
        r["pa_t"] = float(cur["pa"])
        r["pa_prev"] = float(prv["pa"]) if prv is not None and pd.notna(prv["pa"]) else 0.0
        for c in ("xwoba", "fielding_runs", "pos_adj_runs", "avg_ev", "ev90", "max_ev", "barrel_pct", "hardhit_pct", "sweetspot_pct", "pull_air_pct", "gb_pct", "whiff_pct",
                  "chase_pct", "zswing_pct", "zcontact_pct", "swing_pct", "xdamage", "bbe", "air_n", "air_hr", "primary_pos", "team", "player", "side", "xba", "xslg", "woba_raw", "woba_n"):
            r[c] = cur[c] if c in cur.index else np.nan
        recs.append(r)
    d = pd.DataFrame(recs)

    # --- K%: contact-skill (Whiff%) prior --------------------------------------------------------------------------------
    wh, lg_wh = hwf.whiff_table()
    d = d.merge(wh[["name_clean", "swings", "whiff_swing_pct", "whiff_dev_shrunk"]], on="name_clean", how="left")
    pa_eff_k = d["pa_t"] + SPEC["k"][0] * d["pa_prev"]
    d["whiff_term"] = (B_WHIFF * d["whiff_dev_shrunk"] * fit["n0"]["k"] / (pa_eff_k + fit["n0"]["k"])).fillna(0.0)
    d["proj_k_reg"] = d["proj_k"]                                                          # the history regression alone (displayed as his own regressed K%)
    d["k_bridge_dev"] = d["name_clean"].map(k_bridge_dev()) if K_BRIDGE_PRIOR else np.nan
    d["k_bridge_term"] = (d["k_n0"] * d["k_bridge_dev"] / (d["k_nh"] + d["k_n0"])).fillna(0.0)
    d["proj_k"] = d["proj_k"] + d["k_bridge_term"]
    d["proj_k"] = (d["proj_k"] + d["whiff_term"]).clip(0.03, 0.55)
    d["lg_whiff"] = lg_wh

    # --- BB%: plate-discipline (Swing%) prior ---------------------------------------------------------------------------
    sw, lg_sw = hsw.swing_table()
    d = d.merge(sw[["name_clean", "pitches", "swing_pct_pool", "swing_dev_shrunk"]].rename(columns={"pitches": "pitches_seen"}), on="name_clean", how="left")
    pa_eff_bb = d["pa_t"] + SPEC["bbr"][0] * d["pa_prev"]
    d["swing_term"] = (B_SWING * d["swing_dev_shrunk"] * fit["n0"]["bbr"] / (pa_eff_bb + fit["n0"]["bbr"])).fillna(0.0)
    d["proj_bbr_reg"] = d["proj_bbr"]
    d["proj_bbr"] = (d["proj_bbr"] + d["swing_term"]).clip(0.0, 0.4)
    d["lg_swing"] = lg_sw

    # --- wOBA: direct, components, blend, xwOBA term ------------------------------------------------------------------
    lgw = lg["woba"][LATEST]
    comp_dev = sum(fit["comp_beta"][s] * (d[f"proj_{s}"] - lg[s][LATEST]).fillna(0.0) for s in ("k", "bbr", "hrr"))   # missing component (e.g. no AB) -> league level
    d["woba_direct"] = d["proj_woba"]
    d["woba_comp"] = lgw + comp_dev
    blend = 0.5 * (d["woba_direct"] + d["woba_comp"])
    lgx = np.average(d.loc[d["xwoba"].notna(), "xwoba"], weights=d.loc[d["xwoba"].notna(), "pa_t"])
    xs = (d["pa_t"] * d["xwoba"] + N0_XWOBA * lgx) / (d["pa_t"] + N0_XWOBA)
    d["xwoba_shrunk"] = xs
    d["proj_woba_xw"] = blend + GAMMA_XWOBA * (xs - blend).fillna(0.0)
    ev, lg_ev = hev.ev90_table()
    d = d.merge(ev.rename(columns={"ev90": "ev90_pool", "bbe": "bbe_pool"}), on="name_clean", how="left")
    pa_eff = d["pa_t"] + EV90_DELTA * d["pa_prev"]
    n0w = fit["n0"]["woba"] * np.where(d["gap"], SPEC["woba"][1], 1.0)
    d["ev90_term"] = (B_EV90 * d["ev90_dev_shrunk"] * n0w / (pa_eff + n0w)).fillna(0.0)
    d["proj_woba"] = d["proj_woba_xw"] + d["ev90_term"]
    d["proj_woba_base"] = blend
    d["lg_ev90"] = lg_ev
    d["lg_woba"] = lgw
    # --- playing time ------------------------------------------------------------------------------------------------
    pm = fit["pa"]
    g_latest = d["last_season"].map(GAMES)                                               # games in the season his PA total came from
    q_dev = (d["woba_direct"] - lgw).fillna(0.0)                                         # projected wOBA deviation (direct regression), the quality signal behind playing time
    d["proj_pa"] = np.where(d["gap"], GAP_PA, np.maximum(TARGET_GAMES * (pm["a"] + pm["b"] * d["pa_t"] / g_latest + pm["c_quality"] * q_dev), 20.0))
    d["p_return"] = np.where(d["gap"], GAP_P_RETURN, 1 / (1 + np.exp(-(pm["ret_a"] + pm["ret_b"] * np.log(d["pa_t"])))))
    # --- WAR parts per PA --------------------------------------------------------------------------------------------
    t26w = p[(p["year"] == 2026) & (p["source"] == "trackman")]
    rpw = float((t26w["rar"] / t26w["war"].replace(0, np.nan)).median()); repl = float((t26w["repl_runs"] / t26w["pa"]).median())
    pos_group = (t26w.groupby("primary_pos")["pos_adj_runs"].sum() / t26w.groupby("primary_pos")["pa"].sum())
    d["pos_group_rate"] = d["primary_pos"].map(pos_group).fillna(0.0)
    fld_rate = d["fielding_runs"].fillna(0) / d["pa_t"]
    d["fld_rate"] = fld_rate * d["pa_t"] / (d["pa_t"] + FLD_N0)
    d["fld_measured"] = d["primary_pos"].isin(FLD_MEASURED_POS)                           # OF / C (catcher framing); infielders and unknown positions: not measured
    of = t26w[t26w["primary_pos"].isin(("LF", "CF", "RF")) & (t26w["pa"] >= 150)]; r_of = of["fielding_runs"].fillna(0) / of["pa"]
    d["fld_sd_rate"] = np.where(d["fld_measured"], FLD_RESID_PER_PA, float(np.sqrt(np.average((r_of - np.average(r_of, weights=of["pa"])) ** 2, weights=of["pa"]))))
    own_pos = d["pos_adj_runs"] / d["pa_t"]
    d["pos_rate"] = np.where(d["gap"], d["pos_group_rate"], (d["pa_t"] * own_pos + POS_N0 * d["pos_group_rate"]) / (d["pa_t"] + POS_N0))
    d["bat_rate"] = (d["proj_woba"] - lgw) / SC
    d["repl_rate"] = repl
    bm = bsr_proxy.production_model()                                                    # baserunning runs per PA from position incl. DH (2026 bsr_runs)
    d["bsr_pos"] = bsr_proxy.player_positions(d["name_clean"], d["primary_pos"])
    d["bsr_rate"] = bsr_proxy.predict(bm, d["bsr_pos"], d["name_clean"].map(lambda _: np.nan), d["name_clean"].map(lambda _: np.nan))
    d["bsr_tau2"], d["bsr_noise"] = bm["tau2"], bm["noise"]                              # residual true-talent variance per PA^2 and one season's noise per PA (WAR simulation)
    d["rar_rate"] = d["bat_rate"] + d["pos_rate"] + repl + (d["fld_rate"] if WAR_INCLUDE_FIELDING else 0.0) + (d["bsr_rate"] if WAR_INCLUDE_BASERUNNING else 0.0)   # runs above replacement per PA, park-neutral (production: bat + position + baserunning + replacement)
    d["rar_per600"] = 600 * d["rar_rate"]
    d["war_per600"] = d["rar_per600"] / rpw
    d["rpw"], d["repl_per_pa"] = rpw, repl
    # --- wRC+ (park-neutral) -----------------------------------------------------------------------------------------
    lg_rpa = float(read_league_rpa())
    d["proj_wrc_plus"] = 100 * (1 + (d["proj_woba"] - lgw) / (SC * lg_rpa))
    d["lg_runs_per_pa"] = lg_rpa
    # --- Mississippi translation ---------------------------------------------------------------------------------------
    bbe_tot = d["bbe"].sum(min_count=1)
    air_lg = d["air_n"].sum() / d["bbe"].sum()
    air_s = (d["air_n"].fillna(0) + 89.0 * air_lg) / (d["bbe"].fillna(0) + 89.0)          # n0 = 89 batted balls (2025->26 persistence)
    d["air_pct_shrunk"] = air_s
    contact = 1 - d["proj_k"] - d["proj_bbr"] - d["proj_hbpr"].fillna(0)
    air_pa = (contact * air_s).clip(lower=0)
    d["proj_air_per_pa"] = air_pa
    d["mis_woba_shift"] = park.translate(d["name_clean"], share=0.5)
    d["proj_k_mis"] = d["proj_k"] + 0.5 * park.K_EFFECT; d["proj_bbr_mis"] = d["proj_bbr"] + 0.5 * park.BB_EFFECT; d["proj_hrr_mis"] = d["proj_hrr"] + 0.5 * park.effect_hr(d["name_clean"])
    wts = d["proj_pa"]
    d["mis_shift_avg"] = float(np.average(d["mis_woba_shift"], weights=wts))
    d["mis_park_fit_runs"] = (d["mis_woba_shift"] - d["mis_shift_avg"]) / SC * d["proj_pa"]       # runs per projected PA total, vs the average hitter
    d["proj_woba_mis"] = d["proj_woba"] + d["mis_woba_shift"]
    return d, dict(lg=lgw, rpw=rpw, repl=repl, lg_rpa=lg_rpa, lg_k=lg["k"][LATEST], lg_bb=lg["bbr"][LATEST], lg_hr=lg["hrr"][LATEST])


def read_league_rpa():
    from sqlalchemy import text
    from fl_supabase import read_sql_df
    v = read_sql_df(text("select value from war_context where scope='bsr' and key='runs' and year=2026 and game_type='reg_season'"))["value"].iloc[0]
    pa = read_sql_df(text("select sum(pa) pa from lineup_stats where game_type='reg_season' and vs='All' and role='batter' and year_label='2026'"))["pa"].iloc[0]
    return float(v) / float(pa)


# --------------------------------------------------------------------------- #
#  WAR distribution
# --------------------------------------------------------------------------- #
def simulate_war(d, c, seed=7):
    rng = np.random.default_rng(seed)
    n = len(d)
    pa_hat = d["proj_pa"].to_numpy()[:, None]
    z = rng.standard_normal((n, N_SIM))                                                  # the season's standardised wOBA error; PA residual correlated with it (RHO_PT)
    pa = np.clip(pa_hat + dev_sd_pa * (RHO_PT * z + np.sqrt(1 - RHO_PT ** 2) * rng.standard_normal((n, N_SIM))), 10, 700)
    woba = d["proj_woba"].to_numpy()[:, None] + np.sqrt(LUCK_A + LUCK_B / pa) * z
    bat = (woba - c["lg"]) / SC
    fld = (d["fld_rate"].to_numpy()[:, None] + d["fld_sd_rate"].to_numpy()[:, None] * rng.standard_normal((n, N_SIM))) if WAR_INCLUDE_FIELDING else 0.0
    pos = d["pos_rate"].to_numpy()[:, None] + POS_RESID_PER_PA * rng.standard_normal((n, N_SIM))
    bsr_mean = d["bsr_rate"].to_numpy()[:, None] * pa; bsr_sd = np.sqrt(pa ** 2 * d["bsr_tau2"].to_numpy()[:, None] + pa * d["bsr_noise"].to_numpy()[:, None])   # his true rate around the position proxy + one season's baserunning noise (runs)
    runs = pa * (bat + fld + pos + c["repl"]) + ((bsr_mean + bsr_sd * rng.standard_normal((n, N_SIM))) if WAR_INCLUDE_BASERUNNING else 0.0)
    war = runs / c["rpw"]
    fit_adj = (d["mis_woba_shift"].to_numpy()[:, None] - d["mis_shift_avg"].iloc[0]) / SC * pa / c["rpw"]
    war_mis = war + fit_adj
    # point WAR = projected rate x projected PA ("if he plays his projected role"), NOT the simulation mean: with correlated playing time the mean picks up
    # cov(PA, rate) (+.07 WAR on average), which made the per-player bias worse in the backtest (-.09 -> -.16 WAR, fl_informative_pt_test.py); the
    # correlated simulation is used for the range and P(WAR>0) only (user 2026-10-12)
    pt = d["proj_pa"].to_numpy(); point = pt * d["rar_rate"].to_numpy() / c["rpw"]
    point_mis = point + (d["mis_woba_shift"].to_numpy() - d["mis_shift_avg"].iloc[0]) / SC * pt / c["rpw"]
    out = pd.DataFrame({"proj_war": point, "proj_war_sim_mean": war.mean(1), "proj_war_p10": np.percentile(war, 10, axis=1), "proj_war_p50": np.percentile(war, 50, axis=1),
                        "proj_war_p90": np.percentile(war, 90, axis=1), "p_war_pos": (war > 0).mean(1),
                        "proj_war_mis": point_mis, "p_war_pos_mis": (war_mis > 0).mean(1),
                        "proj_war_mis_p10": np.percentile(war_mis, 10, axis=1), "proj_war_mis_p90": np.percentile(war_mis, 90, axis=1)}, index=d.index)
    return out


dev_sd_pa = 129.0


def main(publish: bool):
    global dev_sd_pa
    p = dev.load()
    fit = fit_all(p)
    dev_sd_pa = fit["pa"]["resid_sd"]
    d, c = build_projections(p, fit)
    d = d.join(simulate_war(d, c))
    d["pitcher_like_id"] = d["name_clean"]
    d["hitter"] = d["player"]
    d["wrc_plus_mis"] = 100 * (1 + (d["proj_woba_mis"] - c["lg"]) / (SC * c["lg_runs_per_pa"] if "lg_runs_per_pa" in c else SC * c["lg_rpa"]))
    d["proj_season"] = TARGET_YEAR

    keep = ["name_clean", "hitter", "team", "primary_pos", "side", "last_season", "gap", "pa_t", "proj_pa", "p_return",
            "proj_k", "proj_k_reg", "k_bridge_dev", "k_bridge_term", "whiff_term", "whiff_swing_pct", "swings", "proj_bbr", "proj_bbr_reg", "swing_term", "swing_pct_pool", "pitches_seen", "proj_hrr", "proj_babip", "proj_hbpr", "woba_direct", "woba_comp", "xwoba", "xwoba_shrunk", "proj_woba", "proj_woba_base", "proj_woba_xw", "ev90_term", "ev90_pool", "bbe_pool", "lg_woba", "proj_wrc_plus",
            "proj_woba_mis", "wrc_plus_mis", "mis_woba_shift", "proj_k_mis", "proj_bbr_mis", "proj_hrr_mis", "mis_park_fit_runs", "proj_air_per_pa", "air_pct_shrunk",
            "bat_rate", "fld_rate", "fld_measured", "pos_rate", "bsr_rate", "bsr_pos", "repl_rate", "rar_per600", "war_per600", "rpw", "repl_per_pa",
            "proj_war", "proj_war_p10", "proj_war_p50", "proj_war_p90", "p_war_pos", "proj_war_mis", "proj_war_mis_p10", "proj_war_mis_p90", "p_war_pos_mis",
            "avg_ev", "ev90", "max_ev", "barrel_pct", "hardhit_pct", "sweetspot_pct", "pull_air_pct", "gb_pct", "whiff_pct", "chase_pct", "zswing_pct", "zcontact_pct", "swing_pct", "xdamage",
            "woba_raw", "woba_n"]
    out = d[keep].copy()
    out.to_parquet(OUT_PROJ, index=False)
    hist = p[["name_clean", "year", "team", "source", "pa", "ab", "h", "hr", "bb", "so", "hbp", "k_raw", "bbr_raw", "hrr_raw", "iso", "babip", "woba_raw", "woba_n", "xwoba",
              "war", "batting_runs", "pos_adj_runs", "fielding_runs", "rar", "tm_share"]].rename(columns={"k_raw": "k", "bbr_raw": "bbr", "hrr_raw": "hrr"}).copy()     # observed rates for display (the model uses park-neutral ones)
    # wRC+ for the history table (user request 2026-10-08): park-neutral, computed from park-neutral wOBA the same way as the projected wRC+ (100 x (1 + (wOBA - league) / (1.0175 x league runs per PA))); league runs per PA by year:
    # 2026 = war_context (reg_season runs / PA); 2022-24 = StatsCrew league runs / PA; 2025 backed out of the published wRC+ (slope of wRC+ on wOBA deviation).  Published wRC+ (the pipeline's park factor) for 2025-26 alongside.
    import fl_innings_projection as ip
    from sqlalchemy import text as _t
    from fl_supabase import read_sql_df as _rd
    qq = p.copy(); qq["woba"] = qq["woba_n"]; lg_y = {y: float(bt.league(qq, "woba")[y]) for y in range(2022, 2027)}
    pub = _rd(_t("select player, year, wrc_plus from war_players where game_type='reg_season' and year in (2025, 2026)")); pub["name_clean"] = pub["player"].apply(ip.lastfirst_to_clean); pub = pub.drop_duplicates(["name_clean", "year"])
    sb = pd.read_parquet("output/fl_statscrew_batting_2022_2024.parquet"); rpa = {y: float(g["r"].sum() / g["pa"].sum()) for y, g in sb.groupby("year")}; rpa[2026] = float(read_league_rpa())
    t25 = hist[hist["year"] == 2025].merge(pub[pub["year"] == 2025][["name_clean", "wrc_plus"]], on="name_clean"); t25 = t25[t25["pa"] >= 100]
    slope = np.polyfit(t25["woba_n"] - lg_y[2025], t25["wrc_plus"], 1)[0]; rpa[2025] = 100.0 / (SC * slope)
    hist["wrc_plus_neutral"] = [100 * (1 + (w - lg_y[int(y)]) / (SC * rpa[int(y)])) if pd.notna(w) else np.nan for w, y in zip(hist["woba_n"], hist["year"])]
    hist = hist.merge(pub[["name_clean", "year", "wrc_plus"]].rename(columns={"wrc_plus": "wrc_plus_published"}), on=["name_clean", "year"], how="left")
    print("league runs per PA used for wRC+:", {y: round(v, 4) for y, v in sorted(rpa.items())})
    hist.to_parquet(OUT_HIST, index=False)
    coefs = {"target_year": TARGET_YEAR, "spec": {k: {"delta": v[0], "gap_n0_mult": v[1], "n0": fit["n0"][k]} for k, v in SPEC.items()}, "comp_beta": fit["comp_beta"],
             "gamma_xwoba": GAMMA_XWOBA, "n0_xwoba": N0_XWOBA, "b_ev90_per_mph": B_EV90, "b_whiff": B_WHIFF, "whiff_n0_swings": hwf.N0_WHIFF, "b_swing": B_SWING, "swing_n0_pitches": hsw.N0_SWING, "ev90_n0_balls": hev.N0_EV90, "ev90_delta": EV90_DELTA, "luck_a": LUCK_A, "luck_b": LUCK_B, "pa_model": fit["pa"], "league": c,
             "fld_n0": FLD_N0, "pos_n0": POS_N0, "trustmark": {"model": "post-projection surface (wOBA, HR) not centred + global K% / BB%", "k_effect_full": park.K_EFFECT, "bb_effect_full": park.BB_EFFECT, "kbb_woba_full": park.kbb_woba()}, "rmse": fit["rmse"]}
    with open(OUT_COEF, "w") as f:
        json.dump(coefs, f, indent=1, default=float)
    print(f"saved {OUT_PROJ} ({len(out)} hitters), {OUT_HIST} ({len(hist)} hitter-seasons), {OUT_COEF}")
    print("fit:", {k: round(v, 1) for k, v in fit["n0"].items()}, "| comp beta", {k: round(v, 3) for k, v in fit["comp_beta"].items()})
    print("RMSE (LOTO): wOBA league mean %.4f / last year %.4f / regressed %.4f; components %.4f vs direct %.4f vs blend %.4f on %d rows" % (
        fit["rmse"]["woba"]["league_mean"], fit["rmse"]["woba"]["last_year"], fit["rmse"]["woba"]["model"], fit["rmse"]["woba_components"],
        fit["rmse"]["woba_direct_same_rows"], fit["rmse"]["woba_blend_same_rows"], fit["n_component_rows"]))
    print("league 2026 (park-neutral): wOBA %.4f, K%% %.4f, BB%% %.4f, HR/PA %.4f; runs/PA %.4f; runs per win %.3f; replacement %.4f runs/PA" % (
        c["lg"], c["lg_k"], c["lg_bb"], c["lg_hr"], c["lg_rpa"], c["rpw"], c["repl"]))
    reg = out[(out["pa_t"] >= 150) & (~out["gap"])]
    print(f"\nhitters with 2026 PA>=150 (n={len(reg)}): proj wOBA mean {reg['proj_woba'].mean():.4f} SD {reg['proj_woba'].std():.4f}; proj WAR mean {reg['proj_war'].mean():.2f}; P(WAR>0) mean {reg['p_war_pos'].mean():.2f}; "
          f"Mississippi shift mean {reg['mis_woba_shift'].mean():+.4f} SD {reg['mis_woba_shift'].std():.4f}; park-fit runs SD {reg['mis_park_fit_runs'].std():.2f}")
    show = ["hitter", "team", "primary_pos", "pa_t", "proj_pa", "proj_k", "proj_bbr", "proj_woba", "proj_wrc_plus", "proj_woba_mis", "proj_war", "proj_war_p10", "proj_war_p90", "p_war_pos"]
    print(out[~out["gap"]].sort_values("proj_war", ascending=False)[show].head(12).round(3).to_string(index=False))
    if publish:
        for f in (OUT_PROJ, OUT_HIST, OUT_COEF):
            shutil.copy(f, DASH_DATA)
        print(f"published to {DASH_DATA}")


if __name__ == "__main__":
    main("--publish" in sys.argv)
