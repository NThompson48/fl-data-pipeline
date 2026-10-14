"""
fl_park_visitors_fe.py -- TrustmarkPark effects measured with VISITING PLAYERS ONLY, plus the Mississippi home-vs-road matched pair (user 2026-10-08: home players can be built / selected for the park, so their home results mix the park
with roster fit and home-field effects).
Design:
 HITTER SIDE: woba = year + hitter + pitcher + [hitter x HOME-batter] + park.  The per-hitter home effect absorbs every home hitter's results at his own park, so the park effects are identified from VISITING hitters (hitter FE
   from all their PA; the home pitchers they face are common to the model).  Profile interactions: Trustmark x profile for visiting hitters only (home hitters' interaction is absorbed).
 PITCHER SIDE: woba allowed = year + hitter + pitcher + [pitcher x HOME-pitcher] + park: identified from VISITING pitchers (facing the home hitters).
 (a) park effect on wOBA for every park, hitter side and pitcher side, with a game-cluster bootstrap (B = 100) for the Trustmark SE;  (b) outcome decomposition at Trustmark (HR, K, BB, singles, 2B+3B);
 (c) profile interactions with visitors only, bootstrap SEs;  (d) MATCHED PAIR for the Mud Monsters: with generic home advantage and the usual FE, the extra home-vs-road wOBA of Mississippi's HITTERS and of Mississippi's PITCHERS
     (same players, home vs road) beyond the league-wide home advantage, next to the same quantity for each of the other 17 teams (so Mississippi can be located in the league distribution).
"""
import warnings

import numpy as np
import pandas as pd

from fl_park_profile_fe import cluster_ols, partial_out
from fl_park_woba_fe import fe_fit

warnings.filterwarnings("ignore")


def load():
    import fl_hitter_ros as hr
    pa = hr.Tables().pa.copy(); pit = pd.read_parquet("output/pitcher_pa_ros_2025_2026.parquet"); assert len(pit) == len(pa)
    pa["pid"] = pit["pitcher_id"].astype(object).where(pit["pitcher_id"].notna(), "unknown").astype(str).values; pa["pitcher_team"] = pit["pitcher_team"].values; pa["park"] = pa["game_id"].str.split("-").str[1]; pa["name_clean"] = pa["name_clean"].fillna("unknown")
    pa = pa[pa["woba"].notna() & pa["park"].notna()].reset_index(drop=True); pa["tm"] = (pa["park"] == "TrustmarkPark").astype(int)
    pa["b_home"] = (pa["batter_team"] == pa["home_team"]).astype(int); kind = pa["kind"].astype(str)
    for c, v in (("is_hr", "home_run"), ("is_k", "strikeout"), ("is_bb", "walk"), ("is_1b", "single")):
        pa[c] = (kind == v).astype(float)
    pa["is_xbh"] = kind.isin(["double", "triple"]).astype(float); return pa


def groups_for(pa, side):
    g = [pd.factorize(pa["year"])[0], pd.factorize(pa["name_clean"])[0], pd.factorize(pa["pid"])[0]]
    if side == "hitter":
        g.append(pd.factorize(np.where(pa["b_home"] == 1, pa["name_clean"], "AWAY_HITTER"))[0])        # per-hitter home effect (home hitters' home PA); the AWAY level is a generic home-advantage dummy
        visitor = (pa["b_home"] == 0)
    else:
        g.append(pd.factorize(np.where(pa["b_home"] == 0, pa["pid"], "AWAY_PITCHER"))[0])             # per-pitcher home effect (home pitchers' home PA)
        visitor = (pa["b_home"] == 1)
    g.append(pd.factorize(pa["park"])[0]); return g, visitor.to_numpy()


def park_effects(pa, side, col="woba", iters=30):
    g, vis = groups_for(pa, side); y = pa[col].to_numpy(float); mu, eff = fe_fit(y, g, iters=iters); parks = list(pd.factorize(pa["park"])[1])
    w = np.bincount(g[-1][vis], minlength=len(parks)); pe = pd.Series(eff[-1], index=parks); return pe - np.average(pe, weights=np.maximum(w, 1e-9))


def main():
    pa = load(); out = {}
    for side in ("hitter", "pitcher"):
        out[side] = park_effects(pa, side)
    full = pd.read_csv("output/park_woba_fe_2025_26.csv", index_col=0)["wOBA effect (points)"]
    t = pd.DataFrame({"visiting HITTERS only": out["hitter"], "visiting PITCHERS only": out["pitcher"], "all PA (earlier estimate)": full}); t["average of the two visitor estimates"] = (t.iloc[:, 0] + t.iloc[:, 1]) / 2
    print("(a) PARK EFFECT ON wOBA, visiting players only (points of wOBA vs the league-average park)\n" + t.sort_values("average of the two visitor estimates").round(4).to_string())
    # bootstrap for Trustmark
    rng = np.random.default_rng(9); games = pa["game_id"].to_numpy(); ug = pd.unique(games); by_game = pd.Series(np.arange(len(pa))).groupby(games).apply(np.array); boots = {"hitter": [], "pitcher": []}
    for i in range(100):
        ii = np.concatenate([by_game[g] for g in rng.choice(ug, len(ug))]); sub = pa.iloc[ii].reset_index(drop=True)
        for side in boots:
            boots[side].append(park_effects(sub, side, iters=20).get("TrustmarkPark", np.nan))
    for side in boots:
        a = np.array(boots[side], float); print(f"  Trustmark, visiting {side}s: {out[side]['TrustmarkPark']:+.4f}  bootstrap SE {np.nanstd(a):.4f}  90% [{np.nanpercentile(a, 5):+.4f}, {np.nanpercentile(a, 95):+.4f}]")
    hs, ps = np.array(boots["hitter"], float), np.array(boots["pitcher"], float); both = (hs + ps) / 2
    print(f"  average of the two: {(out['hitter']['TrustmarkPark'] + out['pitcher']['TrustmarkPark']) / 2:+.4f}, bootstrap SE {np.nanstd(both):.4f}, 90% [{np.nanpercentile(both, 5):+.4f}, {np.nanpercentile(both, 95):+.4f}]  (= {(out['hitter']['TrustmarkPark'] + out['pitcher']['TrustmarkPark']) / 2 * 39.9:+.2f} runs/9 for a full-time pitcher)")
    # (b) outcome decomposition at Trustmark
    rows = []
    for lab, col in (("HR/PA", "is_hr"), ("K%", "is_k"), ("BB%", "is_bb"), ("single/PA", "is_1b"), ("2B+3B/PA", "is_xbh"), ("wOBA", "woba")):
        rows.append({"outcome": lab, "league": pa[col].mean(), "visiting hitters": park_effects(pa, "hitter", col)["TrustmarkPark"], "visiting pitchers": park_effects(pa, "pitcher", col)["TrustmarkPark"]})
    print("\n(b) TRUSTMARK effect by outcome (per PA), visitors only\n" + pd.DataFrame(rows).set_index("outcome").round(4).to_string())
    # (d) matched pair: Mississippi hitters / pitchers at home vs on the road, beyond the league-wide home advantage
    g = [pd.factorize(pa["year"])[0], pd.factorize(pa["name_clean"])[0], pd.factorize(pa["pid"])[0]]; y = pa["woba"].to_numpy(float); teams = sorted(set(pa["home_team"].unique()))
    # team-specific home effects: hitters of team T at home (b_home & batter_team == T) and pitchers of team T at home (not b_home & pitcher_team == T); generic HFA = the common dummies b_home / pitcher home
    hit_home = {T: ((pa["b_home"] == 1) & (pa["batter_team"] == T)).to_numpy(float) for T in teams}; pit_home = {T: ((pa["b_home"] == 0) & (pa["pitcher_team"] == T)).to_numpy(float) for T in teams}
    cols = [pa["b_home"].to_numpy(float)] + [hit_home[T] for T in teams] + [pit_home[T] for T in teams]; Xr = partial_out(np.column_stack(cols), g, iters=30); yr = partial_out(y[:, None], g, iters=30)[:, 0]; cl = pd.factorize(pa["game_id"])[0]
    # drop one hitter-team and one pitcher-team dummy for identification against the generic home dummy (sum-to-zero by construction below): use ridge-free approach with the generic dummy removed
    Xd = Xr[:, 1:]; bcoef, se = cluster_ols(Xd, yr, cl); nT = len(teams); hb, pb = bcoef[:nT], bcoef[nT:]; hse, pse = se[:nT], se[nT:]
    # centre the team home effects on their PA-weighted mean (the generic home dummy absorbs the mean)
    wh = np.array([hit_home[T].sum() for T in teams]); wp = np.array([pit_home[T].sum() for T in teams]); hb_c = hb - np.average(hb, weights=wh); pb_c = pb - np.average(pb, weights=wp)
    mt = pd.DataFrame({"team": teams, "hitters at home vs road (extra, wOBA)": hb_c, "se h": hse, "pitchers at home vs road (wOBA allowed, extra)": pb_c, "se p": pse}).set_index("team"); mt["hitters minus pitchers"] = (mt.iloc[:, 0] - mt.iloc[:, 2]) / 2
    print("\n(d) MATCHED PAIR, every team: extra wOBA of its HITTERS at home vs on the road and extra wOBA ALLOWED by its PITCHERS at home (beyond the league-wide home advantage, same players, opponent / year effects controlled)\n" + mt.sort_values("hitters at home vs road (extra, wOBA)").round(4).to_string())
    ms = mt.loc["Mississippi Mud Monsters"]; print(f"\nMississippi: hitters {ms.iloc[0]:+.4f} (SE {ms['se h']:.4f}), pitchers' wOBA allowed {ms.iloc[2]:+.4f} (SE {ms['se p']:.4f}); average of the two (park + fit, hitters' gain and pitchers' gain) = {(ms.iloc[0] - ms.iloc[2]) / 2:+.4f} wOBA = {(ms.iloc[0] - ms.iloc[2]) / 2 * 39.9:+.2f} runs/9-equivalent per half")
    print(f"  league spread of the team home-road effects: hitters SD {mt.iloc[:, 0].std():.4f}, pitchers SD {mt.iloc[:, 2].std():.4f}")
    mt.to_csv("output/park_matched_pair_by_team.csv")


if __name__ == "__main__":
    main()
