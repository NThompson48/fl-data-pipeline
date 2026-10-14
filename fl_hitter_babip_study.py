"""
fl_hitter_babip_study.py -- is hitter BABIP predictable, and from what?

Data: every ball in play (HR excluded) by a Frontier League hitter in 2025-26 regular-season games (pitch-level Trackman,
output/hitter_bip_2025_2026.parquet, cached from Supabase `pitches`).  BIP = single / double / triple / out / fielders_choice / error;
hit = single / double / triple.  Sacrifices are excluded (the official BABIP denominator has sac flies, which cannot be told from bunts here).

A  Skill vs luck inside one season: variance decomposition (observed variance of hitter BABIP minus binomial noise) and split-half
   reliability, for BABIP, for xBABIP (hit probability from exit velocity + launch angle + spray, league-wide model, out-of-fold) and
   for the part of BABIP that exit velocity / launch angle do NOT explain ("finishing").
B  Year to year (2025 -> 2026): correlations and a cross-validated horse race for next-year BABIP.
C  How much of BABIP is the batted ball vs context (team / park / defense): team-level variation of the residual.
D  What it means for the projection model.

Usage:  python fl_hitter_babip_study.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import fl_eval_weights as ew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, KFold
from sqlalchemy import text

from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
CACHE = "output/hitter_bip_2025_2026.parquet"
BIP_RESULTS = ("single", "double", "triple", "out", "fielders_choice", "error")
HITS = ("single", "double", "triple")


def load_bip() -> pd.DataFrame:
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    q = text("""select batter, batter_id, year, game_id, pa_id, play_result, exit_speed, angle, direction, batted_ball_type, tagged_hit_type,
                       batter_side, batter_team, home_team, away_team, batspeed
                from pitches where game_type = 'reg_season' and year in (2025, 2026)
                  and play_result in ('single','double','triple','out','fielders_choice','error')""")
    d = read_sql_df(q)
    d["year"] = d["year"].astype(int)
    d["hit"] = d["play_result"].isin(HITS).astype(int)
    d["venue"] = d["game_id"].str.split("-").str[1]
    d["is_trustmark"] = d["venue"] == "TrustmarkPark"
    d["tracked"] = d["exit_speed"].notna() & d["angle"].notna()
    d.to_parquet(CACHE, index=False)
    return d


def expected_hit_prob(d: pd.DataFrame) -> pd.Series:
    """Out-of-fold P(hit | EV, LA, spray) from a league-wide gradient-boosted model; grouped by hitter-season so a hitter's own
    batted balls never train the model that scores them."""
    t = d[d["tracked"]].copy()
    side = t["batter_side"].map({"right": 1, "left": -1}).fillna(0)
    t["pull_dir"] = t["direction"] * side                       # + = pulled (for a right-handed hitter direction < 0 is the left side; sign flips by side)
    X = t[["exit_speed", "angle", "pull_dir", "direction"]].fillna(0).to_numpy()
    y = t["hit"].to_numpy()
    groups = (t["batter"] + "|" + t["year"].astype(str)).to_numpy()
    p = np.zeros(len(t))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, min_samples_leaf=60, random_state=0).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return pd.Series(p, index=t.index)


def season_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["batter", "year"])
    s = g.agg(bip=("hit", "size"), hits=("hit", "sum")).reset_index()
    tr = d[d["tracked"]].groupby(["batter", "year"]).agg(bip_t=("hit", "size"), hits_t=("hit", "sum"), exp_hits=("xhit", "sum"),
                                                         ev=("exit_speed", "mean"), la=("angle", "mean")).reset_index()
    s = s.merge(tr, on=["batter", "year"], how="left")
    s["babip"] = s["hits"] / s["bip"]
    s["xbabip"] = s["exp_hits"] / s["bip_t"]
    s["babip_t"] = s["hits_t"] / s["bip_t"]
    s["finish"] = s["babip_t"] - s["xbabip"]          # actual minus expected hit rate on the SAME tracked balls
    return s


def partA(d, s):
    print("A. SKILL vs LUCK inside a season")
    lg = d["hit"].mean()
    print(f"   {len(d):,} balls in play, league BABIP {lg:.4f}; tracked (EV & LA) {d['tracked'].mean():.1%}")
    rows = []
    for y in (2025, 2026):
        e = s[(s["year"] == y) & (s["bip_t"] >= 80)]
        for lab, col, nn in (("BABIP (tracked balls)", "babip_t", "bip_t"), ("xBABIP (EV/LA/spray)", "xbabip", "bip_t"), ("finish = BABIP - xBABIP", "finish", "bip_t")):
            x = e[col]; n = e[nn]
            p = e["babip_t"].mean() if col != "xbabip" else e["xbabip"].mean()
            noise = (p * (1 - p) / n).mean() if col != "xbabip" else np.nan
            if col == "finish":
                # actual minus expected: variance of a sum of Bernoullis with their own (known) probabilities
                pass
            rows.append({"year": y, "measure": lab, "hitters (>=80 tracked BIP)": len(e), "mean": x.mean(), "observed SD": x.std(), "noise SD (binomial)": np.sqrt(noise) if col == "babip_t" else np.nan})
    # variance decomposition for BABIP proper
    out = []
    for y in (2025, 2026):
        e = s[(s["year"] == y) & (s["bip_t"] >= 80)]
        p = e["babip_t"].mean(); n = e["bip_t"]
        obs = e["babip_t"].var(); noise = (p * (1 - p) / n).mean()
        out.append({"year": y, "n": len(e), "observed SD": np.sqrt(obs), "noise SD": np.sqrt(noise), "talent SD (BABIP)": np.sqrt(max(obs - noise, 0)),
                    "share of variance that is skill": max(obs - noise, 0) / obs})
    print(pd.DataFrame(out).round(4).to_string(index=False))
    # xBABIP's own spread: its variance is almost all 'real' (batted-ball quality), noise only from the batted-ball distribution itself
    for y in (2025, 2026):
        e = s[(s["year"] == y) & (s["bip_t"] >= 80)]
        print(f"   {y}: SD of xBABIP across hitters {e['xbabip'].std():.4f}; SD of finish {e['finish'].std():.4f}; corr(BABIP, xBABIP) {e['babip_t'].corr(e['xbabip']):.3f}")
    # split-half reliability, random halves of each hitter-season's tracked BIP
    rng = np.random.default_rng(0)
    t = d[d["tracked"]].copy()
    res = {"BABIP": [], "xBABIP": [], "finish": []}
    for rep in range(30):
        t["half"] = rng.integers(0, 2, len(t))
        g = t.groupby(["batter", "year", "half"]).agg(n=("hit", "size"), h=("hit", "sum"), xh=("xhit", "sum")).reset_index()
        w = g.pivot_table(index=["batter", "year"], columns="half", values=["n", "h", "xh"]).dropna()
        w = w[(w[("n", 0)] >= 30) & (w[("n", 1)] >= 30)]
        b0, b1 = w[("h", 0)] / w[("n", 0)], w[("h", 1)] / w[("n", 1)]
        x0, x1 = w[("xh", 0)] / w[("n", 0)], w[("xh", 1)] / w[("n", 1)]
        res["BABIP"].append(b0.corr(b1)); res["xBABIP"].append(x0.corr(x1)); res["finish"].append((b0 - x0).corr(b1 - x1))
    rows = []
    for k, v in res.items():
        r = float(np.mean(v)); sb = 2 * r / (1 + r)
        rows.append({"measure": k, "split-half r (halves >= 30 BIP)": r, "Spearman-Brown full-season reliability": sb})
    print("   split-half reliability (30 random splits, hitter-seasons with >=60 tracked BIP):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def partB(s):
    print("\nB. YEAR TO YEAR, 2025 -> 2026")
    a = s[(s["year"] == 2025) & (s["bip_t"] >= 50)].set_index("batter"); b = s[(s["year"] == 2026) & (s["bip_t"] >= 50)].set_index("batter")
    j = a.join(b, lsuffix="_25", rsuffix="_26", how="inner")
    print(f"   n = {len(j)} hitters with >= 50 tracked BIP in both years")
    rows = []
    for lab, c in (("BABIP", "babip_t"), ("xBABIP", "xbabip"), ("finish (BABIP - xBABIP)", "finish")):
        rows.append({"measure": lab, "corr 2025 -> 2026 (same measure)": j[f"{c}_25"].corr(j[f"{c}_26"])})
    rows.append({"measure": "xBABIP 2025 -> BABIP 2026", "corr 2025 -> 2026 (same measure)": j["xbabip_25"].corr(j["babip_t_26"])})
    rows.append({"measure": "BABIP 2025 -> BABIP 2026", "corr 2025 -> 2026 (same measure)": j["babip_t_25"].corr(j["babip_t_26"])})
    rows.append({"measure": "xBABIP 2025 -> xBABIP 2026", "corr 2025 -> 2026 (same measure)": j["xbabip_25"].corr(j["xbabip_26"])})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    lgB = (j["hits_t_26"].sum() / j["bip_t_26"].sum()); lg25 = (j["hits_t_25"].sum() / j["bip_t_25"].sum())
    y = j["babip_t_26"].to_numpy()
    _ref = lgB + ((j["bip_t_25"] * j["babip_t_25"] + 800 * lg25) / (j["bip_t_25"] + 800) - lg25).to_numpy()
    w, _a, _b = ew.weights_from_base(y, _ref, j["bip_t_26"].to_numpy())
    print(f"   variance law for next-year BABIP: a={_a:.5f}, b={_b:.3f} (n0_eval = {_b/max(_a,1e-12):.0f})")

    def shr(x, n, n0, lg):
        return (n * x + n0 * lg) / (n + n0)
    cands = {}
    cands["league mean"] = np.full(len(j), lgB)
    cands["last year BABIP (raw)"] = j["babip_t_25"].to_numpy() + (lgB - lg25)
    for n0 in (100, 200, 400, 800):
        cands[f"BABIP shrunk n0={n0}"] = lgB + (shr(j["babip_t_25"], j["bip_t_25"], n0, lg25) - lg25).to_numpy()
    for n0 in (50, 100, 200, 400):
        cands[f"xBABIP shrunk n0={n0}"] = lgB + (shr(j["xbabip_25"], j["bip_t_25"], n0, j["xbabip_25"].mean()) - j["xbabip_25"].mean()).to_numpy()
    rm = lambda p: float(np.sqrt(np.average((y - p) ** 2, weights=w)))
    res = [{"predictor of 2026 BABIP": k, "RMSE": rm(v)} for k, v in cands.items()]
    # cross-validated blends (weights fit on training folds)
    xs = j["xbabip_25"].fillna(j["xbabip_25"].mean()); bs = j["babip_t_25"]
    from sklearn.linear_model import LinearRegression
    for lab, cols in (("fit: BABIP + xBABIP", ["babip_t_25", "xbabip_25"]), ("fit: xBABIP only", ["xbabip_25"]), ("fit: BABIP only", ["babip_t_25"]),
                      ("fit: xBABIP + K% proxy (ev)", ["xbabip_25", "ev_25"])):
        X = j[cols].fillna(j[cols].mean()).to_numpy(); pr = np.zeros(len(y))
        for rep in range(20):
            for tri, tei in KFold(10, shuffle=True, random_state=rep).split(X):
                m = LinearRegression().fit(X[tri], y[tri], sample_weight=w[tri]); pr[tei] += m.predict(X[tei]) / 20
        res.append({"predictor of 2026 BABIP": lab + " (10-fold)", "RMSE": rm(pr)})
    print(pd.DataFrame(res).round(4).to_string(index=False))
    m = LinearRegression().fit(j[["babip_t_25", "xbabip_25"]], y, sample_weight=w)
    print(f"   full-sample weights: BABIP {m.coef_[0]:+.3f}, xBABIP {m.coef_[1]:+.3f}, intercept {m.intercept_:+.3f}")


def partC(d, s):
    print("\nC. CONTEXT: is the part of BABIP the batted ball does not explain a team / park / defense effect?")
    t = d[d["tracked"]].copy()
    t["resid"] = t["hit"] - t["xhit"]
    # by hitter's team (home-park defense and park half the time) and by venue
    for lab, key in (("hitter's team", "batter_team"), ("venue", "venue")):
        g = t.groupby(["year", key]).agg(n=("resid", "size"), mean=("resid", "mean")).reset_index()
        g = g[g["n"] >= 400]
        obs = g.groupby("year")["mean"].var()
        noise = (t["resid"].var() / g["n"]).groupby(g["year"]).mean()
        print(f"   {lab}: SD of mean residual across {g.groupby('year').size().to_dict()} groups = {np.sqrt(obs).round(4).to_dict()}  vs binomial noise SD {np.sqrt(noise).round(4).to_dict()}")
    mt = t.groupby("is_trustmark").agg(n=("resid", "size"), bab=("hit", "mean"), xb=("xhit", "mean"), res=("resid", "mean"))
    print("   Trustmark vs elsewhere (all tracked non-HR balls in play):"); print(mt.round(4).to_string())
    # same hitters at home vs road for Mississippi hitters
    m = t[t["batter_team"] == "Mississippi Mud Monsters"]
    g = m.groupby("is_trustmark").agg(n=("resid", "size"), babip=("hit", "mean"), xbabip=("xhit", "mean"))
    print("   Mississippi hitters, Trustmark vs road:"); print(g.round(4).to_string())


def partD(s):
    print("\nD. IMPLICATION FOR THE PROJECTION MODEL")
    a = s[(s["year"] == 2025) & (s["bip_t"] >= 50)].set_index("batter"); b = s[(s["year"] == 2026) & (s["bip_t"] >= 50)].set_index("batter")
    j = a.join(b, lsuffix="_25", rsuffix="_26", how="inner")
    print("   talent SD of BABIP ~ sqrt(cov(2025, 2026)) when the skill is stable: %.4f" % np.sqrt(max(np.cov(j["babip_t_25"], j["babip_t_26"])[0, 1], 0)))
    print("   talent SD of xBABIP ~ %.4f; of finishing ~ %.4f" % (np.sqrt(max(np.cov(j["xbabip_25"], j["xbabip_26"])[0, 1], 0)), np.sqrt(max(np.cov(j["finish_25"], j["finish_26"])[0, 1], 0))))


if __name__ == "__main__":
    d = load_bip()
    d["xhit"] = np.nan
    d.loc[d["tracked"], "xhit"] = expected_hit_prob(d)
    s = season_table(d)
    s.to_parquet("output/hitter_babip_seasons.parquet", index=False)
    partA(d, s)
    partB(s)
    partC(d, s)
    partD(s)
