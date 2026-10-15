"""
fl_ros_production.py -- rest-of-season (ROS) snapshots for hitters and pitchers as of any date inside a season.

HITTERS   rates   fl_hitter_ros.predict: the season to date is the newest partial season in the production regression + the pitch-level terms (xwOBA nudge, EV90, Whiff% -> K%, Swing% -> BB%);
                  wOBA gets the seasonal league-level correction (ROS minus YTD league wOBA by fraction of the season, from the seasons NOT being scored; provisional)
          work    PA over the rest of the season = PA per team game (blended with last season's) x team games remaining x a recent-activity rule (fl_ros_workload)
          WAR     bat (wOBA vs the ROS league level / wOBA scale) + fielding + position (preseason rates) + replacement runs, over ROS PA, with an 80% range (simulation)
PITCHERS  rates   fl_pitcher_ros: Marcel-style update + wh_sva_level (K%) and command_score (BB%) priors rebuilt from the pitches before the date (fl_pitcher_ros_stuff)
          RA9     the production park-neutral baseline (RA9 ~ K% + BB% + IP/appearance, frozen) on the ROS K% / BB%, plus the Mississippi (half games at TrustmarkPark) shift
          work    BF over the rest of the season (same workload model) -> innings at 4.512 BF/IP
          WAR     (replacement RA9 - RA9) x ROS IP / 9 / runs-per-win; P(WAR>0) from the luck law 185.16 / IP
Every snapshot carries the realised ROS outcome (K%, BB%, wOBA, PA / BF, and RA9 / IP from the play-by-play for 2026) when the season has already been played, so the dashboard can show forecast vs result.

Usage:  python fl_ros_production.py            (self-check on one 2026 as-of date)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import text

import fl_hitter_ros as hr
import fl_innings_projection as ip
import fl_pitcher_features as fp
import fl_pitcher_ros as pr
import fl_pitcher_ros_stuff as pst
import fl_ra9_trustmark_projection as ra9
import fl_ros_workload as wl
from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
SC = 1.0175
BF_PER_IP = ra9.LEAGUE_BF_PER_IP
CUTS = wl.CUTS
STUFF_B = {"clean": (0.929, 0.030), "pooled": (0.958, 0.0345)}       # (K% per wh_sva unit, BB% per command_score inch): 2025-only fit (for scoring 2026) / pooled 2025-26 fit (live use)
LUCK_A, LUCK_B = 0.000266, 0.4545                                      # hitter wOBA error law
FLD_N0, POS_N0, FLD_RESID, POS_RESID = 200.0, 100.0, 6.2 / 600, 6.0 / 600
HIT_PA_SD, PIT_BF_SD = 70.0, 68.0                                      # workload RMSE from the backtest (activity rule)


def seasonal_delta(T_h, seasons, frac, stat="woba"):
    """ROS-minus-YTD league level of `stat` at season fraction `frac`, averaged over `seasons` and linearly interpolated across the five backtest cutoffs (flat outside)."""
    vals = []
    for y in seasons:
        t0, t1 = T_h.season_bounds(y); d = T_h.pa[T_h.pa["year"] == y]
        pts = []
        for c in CUTS:
            a = t0 + (t1 - t0) * c; ytd, r = d[d["date"] < a], d[d["date"] >= a]
            pts.append(r["woba"].mean() - ytd["woba"].mean())
        vals.append(np.interp(frac, CUTS, pts))
    return float(np.mean(vals))


def workload_fit(D, tag, years):
    d = D[tag]; d = d[d["year"].isin(years)]
    best = None
    for k in (2, 5, 10, 20, 40):
        p, _, _ = wl.predict_workload(d, k, d["lg_rate"]); e = float(np.sqrt(np.mean((d["n_ros"] - p) ** 2)))
        if best is None or e < best[0]:
            best = (e, k)
    k = best[1]; c, im = wl.fit_activity(d, k)
    return {"k": k, "c": c, "inactive": im}


def display_names(pa, col):
    return pa.drop_duplicates("name_clean").set_index("name_clean")[col]


# ---------------------------------------------------------------------------------------------------------------------------------- hitters
def hitter_snapshot(T_h, year, asof, wfit, delta_seasons, mis_share=0.5):
    f, lg = hr.features(T_h, year, asof, population="ytd")
    pred = hr.predict(f, lg, trackman=True); pre = hr.predict(f, lg, trackman=False, use_ytd=False)
    t0, t1 = T_h.season_bounds(year); frac = float((asof - t0) / (t1 - t0))
    dw = seasonal_delta(T_h, delta_seasons, frac) if delta_seasons else 0.0
    tg_cur, tg_prev = wl.team_games(year), wl.team_games(year - 1)
    wf = wl.workload_frame(T_h.pa, "name_clean", "batter_team", year, asof, tg_cur, tg_prev).reindex(f.index)
    wf["lg_rate"] = wf["n_ytd"].sum() / max(wf["g_eff"].sum(), 1)
    pa_ros = pd.Series(wl.predict_final(wf, wfit["k"], wfit["c"], wfit["inactive"]), index=f.index)
    lg_level = lg["woba"] + dw
    woba_ros = pred["woba_final"] + dw
    # WAR parts: preseason fielding / position rates (prior season), replacement and runs-per-win from the latest published WAR table
    prev = T_h.panel[T_h.panel["year"] == year - 1].drop_duplicates("name_clean").set_index("name_clean").reindex(f.index)
    cur_pos = T_h.panel[(T_h.panel["year"] == 2026) & (T_h.panel["source"] == "trackman")].drop_duplicates("name_clean").set_index("name_clean")
    grp = cur_pos.groupby("primary_pos")["pos_adj_runs"].sum() / cur_pos.groupby("primary_pos")["pa"].sum()
    pos_lab = cur_pos["primary_pos"].reindex(f.index)
    fld = (prev["fielding_runs"].fillna(0) / prev["pa"].replace(0, np.nan)).fillna(0) * prev["pa"].fillna(0) / (prev["pa"].fillna(0) + FLD_N0)
    import fl_bsr_proxy as _bp0, fl_hitter_projection as _hp0
    pt = _bp0.positions(2026); pt = pt[pt["pos_adj"].notna()]; glab = pt["pos"] if _hp0.POS_DH_GROUP else pt["field_pos"]
    grp = pt.groupby(glab)["pos_adj"].sum() / pt.groupby(glab)["pa"].sum()                 # 2026 group rates, DH its own group (as the preseason projection)
    grp_rate = pd.Series(f.index, index=f.index).map(glab).fillna(pos_lab).map(grp).fillna(0.0)
    own_pa = pd.Series(f.index, index=f.index).map(pt["pa"]); own = pd.Series(f.index, index=f.index).map(pt["pos_adj"]) / own_pa
    pos = ((own_pa * own + _hp0.POS_N0 * grp_rate) / (own_pa + _hp0.POS_N0)).where(own.notna(), grp_rate)     # own 2026 rate shrunk toward the group, as the preseason projection
    t26 = cur_pos; rpw = float((t26["rar"] / t26["war"].replace(0, np.nan)).median()); repl = float((t26["repl_runs"] / t26["pa"]).median())
    import fl_hitter_projection as _hp
    fld_w = fld if _hp.WAR_INCLUDE_FIELDING else fld * 0.0                     # production WAR = batting + position + baserunning proxy + replacement (user 2026-10-13 / 14)
    import fl_bsr_proxy as _bp
    bm = _bp.production_model(); bsr = pd.Series(_bp.predict(bm, _bp.player_positions(f.index, pos_lab.to_numpy()), [np.nan] * len(f), [np.nan] * len(f)), index=f.index)
    bsr_w = bsr if _hp.WAR_INCLUDE_BASERUNNING else bsr * 0.0                  # position average incl. DH, fitted on FULL-season 2026 bsr_runs (in-season 2026 snapshots see it: display only)
    rate = (woba_ros - lg_level) / SC + fld_w + pos + bsr_w + repl
    rng = np.random.default_rng(1)
    n = len(f); S = 800
    pa_d = np.clip(pa_ros.to_numpy()[:, None] + HIT_PA_SD * rng.standard_normal((n, S)) * np.minimum(1.0, pa_ros.to_numpy()[:, None] / 40 + 0.0), 0, None)
    wsd = np.sqrt(LUCK_A + LUCK_B / np.maximum(pa_d, 5))
    r_d = (woba_ros.to_numpy()[:, None] + wsd * rng.standard_normal((n, S)) - lg_level) / SC + ((fld.to_numpy()[:, None] + FLD_RESID * rng.standard_normal((n, S))) if _hp.WAR_INCLUDE_FIELDING else 0.0) + pos.to_numpy()[:, None] + POS_RESID * rng.standard_normal((n, S)) + repl
    war = pa_d * r_d / rpw
    if _hp.WAR_INCLUDE_BASERUNNING:                                               # baserunning runs: proxy x PA + his true rate around it + the remaining season's noise
        war = war + (pa_d * bsr.to_numpy()[:, None] + np.sqrt(pa_d ** 2 * bm["tau2"] + pa_d * bm["noise"]) * rng.standard_normal((n, S))) / rpw
    out = pd.DataFrame(index=f.index)
    out["team"] = wf["team"]; out["ytd_pa"] = f["n"]; out["days_since"] = wf["days_since"]; out["g_rem"] = wf["g_rem"]
    out["ytd_woba"] = np.where(f["n"] > 0, f["woba_sum"] / f["n"].replace(0, np.nan), np.nan); out["ytd_k"] = f["k"] / f["n"]; out["ytd_bb"] = f["bb"] / f["n"]
    out["pre_woba"] = pre["blend"]; out["pre_k"] = pre["k"]; out["pre_bb"] = pre["bbr"]
    out["ros_woba"] = woba_ros; out["ros_k"] = pred["k_final"]; out["ros_bb"] = pred["bbr_final"]
    out["ros_wrc_plus"] = 100 * (1 + (woba_ros - lg_level) / (SC * 0.1529))
    out["ros_pa"] = pa_ros
    out["bat_rate"] = (woba_ros - lg_level) / SC; out["fld_rate"] = fld; out["pos_rate"] = pos; out["bsr_rate"] = bsr_w; out["repl_rate"] = repl
    out["ros_war"] = war.mean(1); out["ros_war_p10"] = np.percentile(war, 10, axis=1); out["ros_war_p90"] = np.percentile(war, 90, axis=1); out["p_war_pos"] = (war > 0).mean(1)
    # Mississippi: wOBA shift for his projected HR rate and air balls, half his PA at TrustmarkPark, applied to the ROS wOBA (display)
    out["act_woba"] = f["woba_ros"]; out["act_k"] = f["k_ros"]; out["act_bb"] = f["bb_ros"]; out["act_pa"] = f["n_ros"]
    out["lg_woba_ytd"] = lg["woba"]; out["seasonal_delta"] = dw; out["frac"] = frac
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- pitchers
class PitcherContext:
    def __init__(self, T_p):
        self.raw = fp.load_fl_pitches_cached(); self.names = pst.idmap(self.raw)
        self.war = read_sql_df(text("select player, year, bf, ip, so, bb, ra, ra9, rar, ppf, g from war_pitchers where game_type='reg_season'"))
        self.war["name_clean"] = self.war["player"].apply(ip.lastfirst_to_clean)
        self.base_model, self.train_ranges = ra9.fit_ra9_baseline_model(self.war)
        self.rpw = 10.706
        air = pd.read_parquet("../projections-dashboard/data/fl_ra9_trustmark_projection.parquet"); air["name_clean"] = air["pitcher"].apply(ip.lastfirst_to_clean)
        self.air = air.drop_duplicates("name_clean").set_index("name_clean")["proj_air_pct"]
        try:
            self.runs = pd.read_parquet("output/pitcher_runs_dated_2026.parquet")
        except Exception:
            self.runs = None


def pitcher_snapshot(T_p, ctx, year, asof, consts, n0s, B, wfit, pub=None, term_scale=1.0):
    f, lg = pr.features(T_p, year, asof, population="ytd")
    base = pr.predict(f, lg, consts); pre = pr.predict(f, lg, consts, use_ytd=False)
    sf = pst.stuff_features(ctx.raw, year, asof, ctx.names, n0s).reindex(f.index).fillna(0.0)
    fade_k = consts["k"]["n0"] / (base["pa_eff_k"] + consts["k"]["n0"]); fade_b = consts["bbr"]["n0"] / (base["pa_eff_bbr"] + consts["bbr"]["n0"])
    k_base, bb_base = base["k"], base["bbr"]
    if pub is not None:
        # PUBLISHED preseason projection as the prior mean (fl_ros_published_prior.py): recentred to the current league level, prior strength = the Marcel history weight + n0 (phi = 1);
        # pitchers without a published projection keep the Marcel history prior.  The published model already contains last season's stuff / command, so the stuff terms are halved.
        hist = pr.predict(f, lg, consts, use_ytd=False); res = []
        for key, col, ycol in (("k", "pre_k", "k"), ("bbr", "pre_bb", "bb")):
            m0 = pub[col].reindex(f.index); has = m0.notna().to_numpy(); n = f["n"].to_numpy(); n_p = (hist[f"pa_eff_{key}"] + consts[key]["n0"]).to_numpy()
            dev0 = (m0 - np.average(m0[has], weights=np.maximum(f["n"][has], 1))).fillna(0.0).to_numpy(); x_ytd = np.where(n > 0, f[ycol].to_numpy() / np.where(n > 0, n, 1), 0.0)
            est = lg[key] + (n_p * dev0 + n * (x_ytd - lg[key])) / (n_p + n)
            res.append(pd.Series(np.where(has, est, base[key].to_numpy()), index=f.index))
        k_base, bb_base = res
        fade_k = consts["k"]["n0"] / (hist["pa_eff_k"] + f["n"] + consts["k"]["n0"]); fade_b = consts["bbr"]["n0"] / (hist["pa_eff_bbr"] + f["n"] + consts["bbr"]["n0"])
        B = (B[0] * term_scale, B[1] * term_scale)
    k_ros = (k_base + B[0] * sf["wh_sva"] * fade_k).clip(0.03, 0.6); bb_ros = (bb_base + B[1] * sf["command_score"] * fade_b).clip(0.01, 0.4)
    t0, t1 = T_p.season_bounds(year); frac = float((asof - t0) / (t1 - t0))
    tg_cur, tg_prev = wl.team_games(year), wl.team_games(year - 1)
    wf = wl.workload_frame(T_p.pa, "name_clean", "pitcher_team", year, asof, tg_cur, tg_prev).reindex(f.index); wf["lg_rate"] = wf["n_ytd"].sum() / max(wf["g_eff"].sum(), 1)
    bf_ros = pd.Series(wl.predict_final(wf, wfit["k"], wfit["c"], wfit["inactive"]), index=f.index); ip_ros = bf_ros / BF_PER_IP
    # innings per appearance: last season (EB) + this season to date (approximate IP = BF / league BF per IP)
    pa = T_p.pa[(T_p.pa["year"] == year) & (T_p.pa["date"] < asof)]; g_ytd = pa.groupby("name_clean")["game_id"].nunique().reindex(f.index).fillna(0); ip_ytd = f["n"] / BF_PER_IP
    w_prev = ctx.war[ctx.war["year"] == year - 1].groupby("name_clean").agg(ip=("ip", "sum"), g=("g", "sum")).reindex(f.index).fillna(0)
    ipg = (w_prev["ip"] + ip_ytd + ra9.N0_IPG * ra9.LG_IPG) / (w_prev["g"] + g_ytd + ra9.N0_IPG)
    X = pd.DataFrame({"K_pred": k_ros, "BB_pred": bb_ros, "ip_per_g_prior": ipg}, index=f.index); X = ra9.clip_to_training_range(X, ctx.train_ranges)
    X0 = pd.DataFrame({"K_pred": pre["k"], "BB_pred": pre["bbr"], "ip_per_g_prior": ipg}, index=f.index); X0 = ra9.clip_to_training_range(X0, ctx.train_ranges)
    ra9_n = ctx.base_model.predict(X); ra9_pre = ctx.base_model.predict(X0)
    shift_in = pd.DataFrame({"name_clean": f.index.to_numpy(), "K_pred": k_ros.to_numpy(), "BB_pred": bb_ros.to_numpy(), "proj_air_pct": [ctx.air.get(n, ra9.LEAGUE_AVG_AIR_PCT) for n in f.index], "proj_bf": 100 * BF_PER_IP, "proj_ip": 100.0}, index=f.index)
    shift = ra9.build_trustmark_adjustment(shift_in, 0.5)["trustmark_ra9_shift"]
    repl = ra9.LEAGUE_RA9_REPLACEMENT; repl_mis = repl + ra9.build_replacement_trustmark_shift(0.5)
    ip_eff = np.maximum(ip_ros, ra9.UNCERTAINTY_MIN_IP)
    sd_ra9 = np.sqrt(ra9.RA9_LUCK_VAR_COEF / ip_eff)
    war = (repl - ra9_n) * ip_ros / 9 / ctx.rpw
    sd_ip = PIT_BF_SD / BF_PER_IP
    war_sd = np.sqrt((ip_ros / 9 / ctx.rpw) ** 2 * sd_ra9 ** 2 + ((repl - ra9_n) / 9 / ctx.rpw) ** 2 * sd_ip ** 2)
    out = pd.DataFrame(index=f.index)
    out["team"] = wf["team"]; out["ytd_bf"] = f["n"]; out["days_since"] = wf["days_since"]; out["g_rem"] = wf["g_rem"]
    out["ytd_k"] = f["k"] / f["n"]; out["ytd_bb"] = f["bb"] / f["n"]; out["pre_k"] = pre["k"]; out["pre_bb"] = pre["bbr"]
    out["ros_k"] = k_ros; out["ros_bb"] = bb_ros; out["ros_kbb"] = k_ros - bb_ros; out["pre_ra9"] = ra9_pre
    out["ros_ra9"] = ra9_n; out["ros_ra9_mis"] = ra9_n + shift; out["ros_bf"] = bf_ros; out["ros_ip"] = ip_ros; out["ip_per_app"] = ipg
    out["ros_war"] = war; out["ros_war_p10"] = war - 1.2816 * war_sd; out["ros_war_p90"] = war + 1.2816 * war_sd
    out["p_war_pos"] = norm.cdf((repl - ra9_n) / sd_ra9); out["p_war_pos_mis"] = norm.cdf((repl_mis - (ra9_n + shift)) / sd_ra9)
    out["act_k"] = f["k_ros"]; out["act_bb"] = f["bb_ros"]; out["act_bf"] = f["n_ros"]
    if ctx.runs is not None and year == 2026:
        r = ctx.runs[ctx.runs["date"] >= asof].groupby("name_clean")[["outs", "runs"]].sum().reindex(f.index).fillna(0)
        out["act_ip"] = r["outs"] / 3; out["act_ra9"] = np.where(r["outs"] > 0, 9 * r["runs"] / (r["outs"] / 3), np.nan)
        y = ctx.runs[ctx.runs["date"] < asof].groupby("name_clean")[["outs", "runs"]].sum().reindex(f.index).fillna(0); out["ytd_ip"] = y["outs"] / 3
        out["ytd_ra9"] = np.where(y["outs"] > 0, 9 * y["runs"] / (y["outs"] / 3), np.nan)
    out["frac"] = frac
    return out


def main():
    T_h = hr.Tables(); T_p = pr.Tables(); ctx = PitcherContext(T_p)
    D = wl.collect(T_h, T_p)
    wf_h = workload_fit(D, "hit", (2025,)); wf_p = workload_fit(D, "pit", (2025,))
    print("workload fits (2025 only):", wf_h, wf_p)
    asof = pd.Timestamp("2026-07-15")
    h = hitter_snapshot(T_h, 2026, asof, wf_h, (2025,))
    print("\nHITTERS as of", asof.date(), len(h)); print(h[["team", "ytd_pa", "pre_woba", "ros_woba", "ros_pa", "ros_war", "ros_war_p10", "ros_war_p90", "act_woba", "act_pa"]].sort_values("ros_war", ascending=False).head(6).round(3).to_string())
    consts = pr.fit_constants(2025); n0s = pst.reliability(ctx.raw)
    p = pitcher_snapshot(T_p, ctx, 2026, asof, consts, n0s, STUFF_B["clean"], wf_p)
    print("\nPITCHERS as of", asof.date(), len(p)); print(p[["team", "ytd_bf", "pre_ra9", "ros_ra9", "ros_ra9_mis", "ros_ip", "ros_war", "p_war_pos", "act_ra9", "act_ip"]].sort_values("ros_war", ascending=False).head(6).round(3).to_string())


if __name__ == "__main__":
    main()
