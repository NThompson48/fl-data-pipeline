"""
fl_hitter_babip_venue.py -- follow-up to fl_hitter_babip_study.py: the venue (park + home defense) shows a BABIP effect the batted ball does not
explain (SD of venue mean residual 0.018-0.020 vs 0.0075 binomial noise). Is it stable, and does removing it make hitter BABIP predictable?

1  Stability: correlation of each venue's residual (hit - expected hit) between 2025 and 2026; which venues.
2  Park-adjusted BABIP: remove the venue effect (estimated from 2025 balls only, then applied to both years, so the 2026 target is never used to
   estimate it) and redo the year-to-year correlation and the predictive horse race.
3  Home/road split inside the same hitters (removes "different hitters play there").

Usage:  python fl_hitter_babip_venue.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import fl_eval_weights as ew
import fl_hitter_babip_study as bs

warnings.filterwarnings("ignore")


def main():
    d = bs.load_bip()
    d["xhit"] = np.nan
    d.loc[d["tracked"], "xhit"] = bs.expected_hit_prob(d)
    t = d[d["tracked"]].copy()
    t["resid"] = t["hit"] - t["xhit"]

    print("1. VENUE EFFECT STABILITY (mean of hit - expected hit on tracked balls in play, per venue)")
    v = t.groupby(["venue", "year"]).agg(n=("resid", "size"), res=("resid", "mean")).reset_index().pivot(index="venue", columns="year", values=["n", "res"])
    v.columns = [f"{a}_{b}" for a, b in v.columns]
    v = v.dropna()
    sd_noise = t["resid"].std() / np.sqrt(v[["n_2025", "n_2026"]].mean().mean())
    r = v["res_2025"].corr(v["res_2026"])
    cov = np.cov(v["res_2025"], v["res_2026"])[0, 1]
    print(f"   {len(v)} venues; corr(2025, 2026) = {r:.2f}; SD of venue effect (sqrt of cross-year covariance) = {np.sqrt(max(cov, 0)):.4f}; per-year binomial noise SD {sd_noise:.4f}")
    show = v.assign(avg=(v["res_2025"] + v["res_2026"]) / 2).sort_values("avg")
    print(show[["n_2025", "n_2026", "res_2025", "res_2026", "avg"]].round(4).to_string())

    # venue effects from 2025 balls only, shrunk by reliability (cross-year covariance / observed variance)
    e25 = t[t["year"] == 2025].groupby("venue").agg(n=("resid", "size"), res=("resid", "mean"))
    tau2 = max(np.cov(v["res_2025"], v["res_2026"])[0, 1], 1e-6)
    noise = t["resid"].var() / e25["n"]
    w = tau2 / (tau2 + noise)
    eff = (e25["res"] * w).rename("venue_eff")
    t = t.merge(eff, left_on="venue", right_index=True, how="left")
    t["venue_eff"] = t["venue_eff"].fillna(0.0)

    print("\n2. PARK-ADJUSTED HITTER BABIP, 2025 -> 2026 (venue effects estimated from 2025 only, shrunk; applied to both seasons)")
    rows = {}
    for lab, adj in (("raw", 0.0), ("venue-adjusted", t["venue_eff"])):
        z = t.assign(h=t["hit"] - adj, xh=t["xhit"] - (adj if isinstance(adj, float) else 0))
        g = z.groupby(["batter", "year"]).agg(n=("hit", "size"), h=("h", "sum"), xh=("xhit", "sum")).reset_index()
        g["babip"] = g["h"] / g["n"]; g["xbabip"] = g["xh"] / g["n"]
        a = g[(g["year"] == 2025) & (g["n"] >= 50)].set_index("batter"); b = g[(g["year"] == 2026) & (g["n"] >= 50)].set_index("batter")
        j = a.join(b, lsuffix="_25", rsuffix="_26", how="inner")
        rows[lab] = j
        print(f"   {lab:15s}: n={len(j)}  corr BABIP 25->26 {j['babip_25'].corr(j['babip_26']):.3f}   corr xBABIP25 -> BABIP26 {j['xbabip_25'].corr(j['babip_26']):.3f}")
    j = rows["venue-adjusted"]
    y = j["babip_26"].to_numpy()
    lg26 = (j["h_26"].sum() / j["n_26"].sum()); lg25 = (j["h_25"].sum() / j["n_25"].sum())
    _ref = lg26 + ((j["n_25"] * j["babip_25"] + 800 * lg25) / (j["n_25"] + 800) - lg25).to_numpy()
    wts, _a, _b = ew.weights_from_base(y, _ref, j["n_26"].to_numpy())
    rm = lambda p: float(np.sqrt(np.average((y - p) ** 2, weights=wts)))
    res = [{"predictor of venue-adjusted 2026 BABIP": "league mean", "RMSE": rm(np.full(len(y), lg26))}]
    for n0 in (100, 200, 400, 800):
        res.append({"predictor of venue-adjusted 2026 BABIP": f"BABIP shrunk n0={n0}", "RMSE": rm(lg26 + ((j["n_25"] * j["babip_25"] + n0 * lg25) / (j["n_25"] + n0) - lg25).to_numpy())})
    for n0 in (100, 200, 400):
        m = j["xbabip_25"].mean()
        res.append({"predictor of venue-adjusted 2026 BABIP": f"xBABIP shrunk n0={n0}", "RMSE": rm(lg26 + ((j["n_25"] * j["xbabip_25"] + n0 * m) / (j["n_25"] + n0) - m).to_numpy())})
    print(pd.DataFrame(res).round(4).to_string(index=False))

    print("\n3. SAME HITTERS, HOME vs ROAD (does the venue effect show inside a hitter's own balls?)")
    t["is_home"] = t["batter_team"] == t["home_team"]
    g = t.groupby(["batter", "year", "is_home"]).agg(n=("hit", "size"), h=("hit", "sum"), xh=("xhit", "sum")).reset_index()
    w_ = g.pivot_table(index=["batter", "year"], columns="is_home", values=["n", "h", "xh"]).dropna()
    w_ = w_[(w_[("n", True)] >= 25) & (w_[("n", False)] >= 25)]
    home_b = w_[("h", True)].sum() / w_[("n", True)].sum(); road_b = w_[("h", False)].sum() / w_[("n", False)].sum()
    home_x = w_[("xh", True)].sum() / w_[("n", True)].sum(); road_x = w_[("xh", False)].sum() / w_[("n", False)].sum()
    print(f"   {len(w_)} hitter-seasons with >=25 tracked BIP home and road: BABIP home {home_b:.4f} vs road {road_b:.4f}; xBABIP home {home_x:.4f} vs road {road_x:.4f}")
    # team-level (the hitter's own team's park): does a hitter's home/road BABIP gap track his park's venue effect?
    tm = t[t["is_home"]].groupby(["batter_team", "year"]).agg(res=("resid", "mean"), n=("resid", "size")).reset_index()
    print("   home-team mean residual (hit - xhit) on the hitter's OWN home balls: SD across teams %.4f" % tm["res"].std())


if __name__ == "__main__":
    main()
