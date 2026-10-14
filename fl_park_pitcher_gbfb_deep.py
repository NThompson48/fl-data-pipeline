"""
fl_park_pitcher_gbfb_deep.py -- fly-ball share and ground-ball share of PITCHERS at Trustmark, controlling for K% and BB% allowed  (user 2026-10-09: "Fly-ball share and ground-ball share are the ones I'm most interested in")
Same design as fl_park_pitcher_profile_screen.py (visiting pitchers only; year + hitter + pitcher + [pitcher x home] + park FE; game-cluster SEs; profiles from NON-Trustmark balls, shrunk n0 89).
(1) What a +1 SD means in percentage points of balls in play (league mean and pitcher-level SD of the shrunk share).
(2) Specification: GB alone | FB alone | GB + FB | GB + FB + LD (pop-ups the base) | conventional FB >= 25 deg incl. pop-ups | Air% (>= 10 deg, the production term) -- all with K% and BB% controlled.
(3) Shape: terciles of GB share and of FB share (lowest tercile = base), K% and BB% controlled -- is it linear?
(4) By year (2025 / 2026) for the GB + FB model.
(5) WHICH OUTCOMES carry it: HR, singles+doubles+triples, K, BB, and balls in play per PA, GB + FB model.
(6) Structural check: the ball-level park effects (fl_park_play_the_park.py: ground ~0, line drive -.05, fly -.07 to -.18, pop-up ~0) predict a much smaller pitcher-level shift than the regression estimates -- put the two next to each other.
"""
import warnings

import numpy as np
import pandas as pd

import fl_park_pitcher_final as pf
import fl_park_pitcher_profile_screen as sc
from fl_park_profile_fe import cluster_ols, partial_out

warnings.filterwarnings("ignore")


def share_series(bb, lo, hi):
    return ((bb["angle"] >= lo) & (bb["angle"] < hi)).astype(float)


def main():
    est, prof, stdz = pf.build(); est["b_home"] = (est["batter_team"] == est["home_team"]).astype(int); C = sc.build_candidates(est)
    bb = pf.pitcher_bbe(); bb["pid"] = bb["pitcher_id"].astype(str); bb["park"] = bb["game_id"].str.split("-").str[1]; bb = bb[bb["park"] != "TrustmarkPark"]
    extra = {}
    defs = {"GB": (-91, 10), "LD": (10, 25), "FB": (25, 40), "PU": (40, 91), "FB>=25": (25, 91), "AIR>=10": (10, 91)}; sdinfo = []
    for k, (lo, hi) in defs.items():
        s = share_series(bb, lo, hi).groupby(bb["pid"]); m, n = s.mean(), s.size(); lg = np.average(m, weights=n); d = (m - lg) * n / (n + 89); v = est["pid"].map(d); extra[k] = (d - v.mean()) / v.std(); sdinfo.append({"share": k, "league mean": lg, "SD of shrunk share across the PA pool": v.std(), "SD raw (pitchers >= 150 balls)": m[n >= 150].std()})
    print("(1) SCALE OF A +1 SD (percentage points of balls in play)\n" + pd.DataFrame(sdinfo).round(4).to_string(index=False))
    C2 = pd.DataFrame(extra); C2["K"] = C["K% allowed"]; C2["BB"] = C["BB% allowed"]
    for c in C2.columns:
        est["z_" + c] = est["pid"].map(C2[c])
    est = est.dropna(subset=["z_K", "z_BB", "z_GB"]).reset_index(drop=True)
    for c in C2.columns:
        est["z_" + c] = est["z_" + c].fillna(0.0)
    est["woba_y"] = est["woba"]; kind = est["kind"].astype(str); est["is_hit_nohr"] = kind.isin(["single", "double", "triple"]).astype(float); est["is_bb_"] = (kind == "walk").astype(float); est["is_bip_"] = kind.isin(["single", "double", "triple", "out", "fielders_choice", "error", "sacrifice"]).astype(float)
    print(f"\nestimation sample {len(est):,} PA, {est['pid'].nunique()} pitchers; visiting-pitcher PA at Trustmark {int(((est['tm'] == 1) & (est['b_home'] == 1)).sum()):,}")
    home_batter = (est["b_home"] == 1); g = [pd.factorize(est["year"])[0], pd.factorize(est["name_clean"])[0], pd.factorize(est["pid"])[0], pd.factorize(np.where(~home_batter, est["pid"], "AWAY"))[0], pd.factorize(est["park"])[0]]
    vis = home_batter.to_numpy(float); tmv = est["tm"].to_numpy(float) * vis; cl = pd.factorize(est["game_id"])[0]; R9 = pf.TO_R9
    cache = {}

    def resid_y(col):
        if col not in cache:
            cache[col] = partial_out(est[col].to_numpy(float)[:, None], g, iters=30)[:, 0]
        return cache[col]
    Xcache = {}

    def xr(term, mask=None):
        key = (term, None if mask is None else mask.tobytes()[:16])
        if key not in Xcache:
            v = tmv * est[term].to_numpy(float) if mask is None else tmv * mask * est[term].to_numpy(float); Xcache[key] = partial_out(v[:, None], g, iters=30)[:, 0]
        return Xcache[key]

    def run(terms, ycol="woba_y"):
        X = np.column_stack([xr(t) for t in terms]); return cluster_ols(X, resid_y(ycol), cl)
    # (2) specifications
    specs = {"GB alone": ["z_GB"], "FB alone": ["z_FB"], "GB + FB": ["z_GB", "z_FB"], "GB + FB + LD (pop-up base)": ["z_GB", "z_FB", "z_LD"], "FB>=25 (incl. pop-ups) alone": ["z_FB>=25"], "Air>=10 (production term) alone": ["z_AIR>=10"]}; rows = []
    for lab, ts in specs.items():
        b, se = run(["z_K", "z_BB"] + ts)
        for i, t in enumerate(ts):
            rows.append({"specification": lab, "term": t[2:], "coef (wOBA per +1 SD)": b[2 + i], "SE": se[2 + i], "t": b[2 + i] / se[2 + i], "runs/9": b[2 + i] * R9})
    print("\n(2) SPECIFICATIONS (K% and BB% allowed controlled in every row; negative = better for the pitcher at Trustmark)\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    # (3) shape by terciles
    pit = est.drop_duplicates("pid").set_index("pid"); rows = []
    for nm in ("GB", "FB"):
        q = pd.qcut(pit["z_" + nm], 3, labels=False); est["t_" + nm] = est["pid"].map(q.to_dict() if hasattr(q, "to_dict") else q)
        d1, d2 = (est["t_" + nm] == 1).astype(float), (est["t_" + nm] == 2).astype(float); X = np.column_stack([xr("z_K"), xr("z_BB"), partial_out((tmv * d1.to_numpy())[:, None], g, iters=30)[:, 0], partial_out((tmv * d2.to_numpy())[:, None], g, iters=30)[:, 0]]); b, se = cluster_ols(X, resid_y("woba_y"), cl)
        rows.append({"profile": nm + " share", "middle vs lowest tercile": f"{b[2]:+.4f} ({se[2]:.4f})", "highest vs lowest tercile": f"{b[3]:+.4f} ({se[3]:.4f})", "t high vs low": b[3] / se[3], "mean share low / mid / high": " / ".join(f"{bb_:.3f}" for bb_ in [C2[nm].groupby(q).mean().get(i, np.nan) for i in (0, 1, 2)])})
    print("\n(3) SHAPE: terciles of pitchers (K%, BB% controlled)\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    # (4) by year
    y25, y26 = (est["year"] == 2025).to_numpy(float), (est["year"] == 2026).to_numpy(float); cols = {}
    for t in ("z_K", "z_BB", "z_GB", "z_FB"):
        for yy, m in (("2025", y25), ("2026", y26)):
            cols[(t, yy)] = partial_out((tmv * m * est[t].to_numpy(float))[:, None], g, iters=30)[:, 0]
    X = np.column_stack([cols[(t, yy)] for t in ("z_K", "z_BB", "z_GB", "z_FB") for yy in ("2025", "2026")]); b, se = cluster_ols(X, resid_y("woba_y"), cl); rows = []
    for i, t in enumerate(("z_K", "z_BB", "z_GB", "z_FB")):
        rows.append({"term": t[2:], "2025": f"{b[2 * i]:+.4f} ({se[2 * i]:.4f})", "2026": f"{b[2 * i + 1]:+.4f} ({se[2 * i + 1]:.4f})", "t of year difference": (b[2 * i] - b[2 * i + 1]) / np.hypot(se[2 * i], se[2 * i + 1])})
    print("\n(4) BY YEAR (GB + FB model, K% and BB% each year)\n" + pd.DataFrame(rows).to_string(index=False))
    # (5) outcomes
    outs = [("wOBA", "woba_y"), ("HR per PA", "is_hr"), ("1B+2B+3B per PA", "is_hit_nohr"), ("K per PA", "is_k"), ("BB per PA", "is_bb_"), ("balls in play per PA", "is_bip_")]; rows = []
    for lab, col in outs:
        b, se = run(["z_K", "z_BB", "z_GB", "z_FB"], col); rows.append({"outcome": lab, "league mean": est[col].mean(), "GB share": f"{b[2]:+.4f} ({se[2]:.4f})", "t GB": b[2] / se[2], "FB share": f"{b[3]:+.4f} ({se[3]:.4f})", "t FB": b[3] / se[3]})
    print("\n(5) WHICH OUTCOMES CARRY IT (per PA, per +1 SD; GB + FB model with K%, BB% controlled)\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    # (6) structural vs regression
    sdgb = [r for r in sdinfo if r["share"] == "GB"][0]["SD of shrunk share across the PA pool"]; sdfb = [r for r in sdinfo if r["share"] == "FB"][0]["SD of shrunk share across the PA pool"]
    pit_cell = {"GB": 0.01, "LD": -0.05, "FB": -0.10, "PU": -0.01}      # rough per-ball park effects from fl_park_play_the_park.py, averaged over spray (ground ~ +.01, line drive ~ -.05, fly ~ -.10 (pull -.18 / middle -.07 / oppo -.17, weighted), pop-up ~ -.01)
    pred_per_ball = sdfb * (pit_cell["FB"] - pit_cell["GB"]) * 0.5 + sdgb * (pit_cell["GB"] - pit_cell["LD"]) * 0.0
    print(f"\n(6) STRUCTURAL CHECK: +1 SD of fly-ball share = {sdfb * 100:.1f} points of balls in play; moving those from average balls to fly balls changes the park effect by about {sdfb * (pit_cell['FB'] - 0.5 * (pit_cell['GB'] + pit_cell['LD'])):+.4f} wOBA per ball in play = "
          f"{sdfb * (pit_cell['FB'] - 0.5 * (pit_cell['GB'] + pit_cell['LD'])) * 0.68:+.4f} per batter faced (balls in play per PA ~ .68); the regression above is about -.02 per PA per SD, i.e. several times larger.")


if __name__ == "__main__":
    main()
