"""
fl_fielding_rda_style_2026.py -- EXPLORATORY: would a Baseball Prospectus RDA-style infield range metric be more reliable than ours on 2026 data?  (user 2026-10-20)
Articles: baseballprospectus.com/news/article/80209 (Introducing Range Defense Added, 2023) and /99395 (All the Range: an update to infield RDA, 2025).

Ideas tested (BP's, as we read them):
  1. NO starting positions: expected outcomes from the batted ball only (exit speed, launch angle, direction, batter side, distance, hang time); the fielder owns his set-up.
     Ours (PART) conditions on his starting angle / depth.            -> FEATURES = "start" vs "ball"
  2. Attempt Range for every infielder + success on fielded plays.    -> components: REACH (fielded_by == pos vs expected), CONVERT (out given fielded vs expected), TOTAL (out by him vs expected)
  3. Keep the ball-physics model SHALLOW: a very accurate model absorbs fielder skill ("accurate but not too accurate").  -> DEPTH in {1, 2, 3, 4} (our PART metric uses 4)
  4. Judge on team-switchers / resistance to team "house effects".    -> (a) team clustering of fielder estimates vs a permutation null, (b) the few 2026 team-switchers.
Data / protocol identical to fl_fielding_part_2026.py: 2026 balls in play joined to pbp_fielded_by (full season after the 2026-10-20 re-pull), GROUNDERS only (launch angle < 10), per-position gradient
boosting cross-fitted by game (250 iterations, 40 per leaf), runs-free rates per opportunity centred within position, reliability = full-time Spearman-Brown from split-half correlation, fielder-positions with
>= 15 opportunities in both halves; two split schemes: ALT (alternating games within a fielder-position) and CHRON (before / from the league's median game date).
Opportunities: REACH and TOTAL = expected probability summed; CONVERT = number of fielded plays.
PRE-STATED: primary comparison = "ball" vs "start" at the SAME depth, all infielders, REACH and TOTAL; "ball" counts as better only if it beats "start" by > .05 full-time reliability under BOTH split
schemes.  The grid is reported in full (no best-config pick); the highest cell is optimistic by selection.  Reliability is not validity: a house-effect or switcher result that contradicts a gain overrides it.
Nothing here enters WAR; a promising spec would be added to the 2027 pre-registration (C9) as an alternative.  Output: output/fielding_rda_style_2026.json.
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_fielding_part_2026 as fp

warnings.filterwarnings("ignore")
IF = fp.IF
BALL = ["ev", "la", "dir", "left", "dist", "hang"]
COMPS = ("reach", "convert", "total")
rng = np.random.default_rng(11)


def game_dates():
    t = pd.read_parquet("2025_2026_pitches.parquet", columns=["game_id", "date", "year"]); t = t[t["year"] == 2026]
    return t.groupby("game_id")["date"].min().astype(str)


def run_config(m, feats, depth, dates):
    fp.KW["max_depth"] = depth
    ms = m[m["set"] == "GSL"].reset_index(drop=True); gid = ms["game_id"].to_numpy(); out = []
    for pos in IF:
        X, y_out = fp.fielder_frame(ms, pos)
        if feats == "ball":
            X = X[BALL]
        touch = (ms["fielded_by"] == pos).astype(int).to_numpy()
        p_reach = fp.crossfit(X, touch, gid); p_tot = fp.crossfit(X, y_out, gid)
        ti = np.nonzero(touch)[0]; p_conv = np.full(len(ms), np.nan); p_conv[ti] = fp.crossfit(X.iloc[ti].reset_index(drop=True), ms["out"].to_numpy()[ti], gid[ti])
        out.append(pd.DataFrame({"pos": pos, "fielder": ms[f"name_{pos.lower()}"].astype(str), "team": ms["pitcher_team"].to_numpy(), "game_id": gid, "date": ms["game_id"].map(dates).to_numpy(),
                                 "reach_d": touch, "reach_p": p_reach, "total_d": y_out, "total_p": p_tot,
                                 "convert_d": np.where(touch == 1, ms["out"], np.nan), "convert_p": p_conv}))
    P = pd.concat(out, ignore_index=True); P = P[~P["fielder"].isin(["nan", "None"])]
    for c in COMPS:
        P[c + "_opp"] = P[c + "_p"] if c != "convert" else P["convert_d"].notna().astype(float)
        P[c + "_res"] = (P[c + "_d"] - P[c + "_p"]).fillna(0.0)
        g = P.groupby("pos"); P[c + "_res"] -= P[c + "_opp"] * g[c + "_res"].transform("sum") / g[c + "_opp"].transform("sum")
    return P


def reliability(P, comp, scheme, B=1000):
    Q = P[P[comp + "_opp"] > 0]
    if scheme == "alt":
        gm = Q[["fielder", "pos", "game_id", "date"]].drop_duplicates().sort_values(["date", "game_id"]); gm["half"] = gm.groupby(["fielder", "pos"]).cumcount() % 2
        Q = Q.merge(gm[["fielder", "pos", "game_id", "half"]], on=["fielder", "pos", "game_id"])
    else:
        mid = np.sort(Q["date"].unique())[len(Q["date"].unique()) // 2]; Q = Q.assign(half=(Q["date"] >= mid).astype(int))
    H = Q.groupby(["fielder", "pos", "half"]).agg(o=(comp + "_opp", "sum"), r=(comp + "_res", "sum")).unstack("half").dropna()
    H = H[(H[("o", 0)] >= 15) & (H[("o", 1)] >= 15)]
    tot = Q.groupby(["fielder", "pos"])[comp + "_opp"].sum(); ft_all = tot.groupby(level="pos").quantile(0.9); res = {}
    for pos, hh in list(H.groupby(level="pos")) + [("all IF", H)]:
        if len(hh) < 8:
            continue
        a, b = (hh[("r", 0)] / hh[("o", 0)]).to_numpy(), (hh[("r", 1)] / hh[("o", 1)]).to_numpy(); nh = float(((hh[("o", 0)] + hh[("o", 1)]) / 2).mean())
        r = float(np.corrcoef(a, b)[0, 1]); ft = float(ft_all.get(pos, ft_all.mean()))
        f = lambda x: 0.0 if x <= 0 else ft / (ft + nh * (1 - x) / x)
        rs = [np.corrcoef(a[i], b[i])[0, 1] for i in (rng.integers(0, len(a), len(a)) for _ in range(B))]
        res[pos] = {"n": len(hh), "r": r, "rel": f(r), "rel_lo": f(float(np.nanpercentile(rs, 5))), "rel_hi": f(float(np.nanpercentile(rs, 95)))}
    return res


def house_effect(P, comp, perms=1000):
    """Do fielders on the same team look alike more than chance?  F = between-team variance of fielder rates (single-team fielders, >= 40 opportunities) vs the same with team labels shuffled within position."""
    tot = P.groupby(["fielder", "pos"]).agg(o=(comp + "_opp", "sum"), r=(comp + "_res", "sum"), teams=("team", lambda t: set(t)))
    tot = tot[(tot["o"] >= 40) & (tot["teams"].map(len) == 1)].reset_index(); tot["team"] = tot["teams"].map(lambda s: next(iter(s))); tot["rate"] = tot["r"] / tot["o"]
    def stat(team):
        g = pd.DataFrame({"t": team, "x": tot["rate"].to_numpy(), "w": tot["o"].to_numpy()}); mu = np.average(g["x"], weights=g["w"]); s = g.groupby("t").apply(lambda q: q["w"].sum() * (np.average(q["x"], weights=q["w"]) - mu) ** 2)
        return float(s.sum() / g["w"].sum())
    obs = stat(tot["team"].to_numpy()); null = []
    for _ in range(perms):
        t = tot["team"].copy()
        for pos, idx in tot.groupby("pos").groups.items():
            t.loc[idx] = rng.permutation(t.loc[idx].to_numpy())
        null.append(stat(t.to_numpy()))
    return {"n": len(tot), "ratio_to_null": obs / float(np.mean(null)), "p": float((1 + sum(x >= obs for x in null)) / (perms + 1))}


def switchers(P, comp):
    """Fielders who played for 2+ teams: correlation of his rate in his first vs second team (need >= 15 opportunities in each)."""
    T = P[P[comp + "_opp"] > 0].groupby(["fielder", "pos", "team"]).agg(o=(comp + "_opp", "sum"), r=(comp + "_res", "sum"), d0=("date", "min")).reset_index()
    rows = []
    for (f, p), g in T.groupby(["fielder", "pos"]):
        g = g.sort_values("d0")
        if len(g) >= 2 and g["o"].iloc[0] >= 15 and g["o"].iloc[1] >= 15:
            rows.append((g["r"].iloc[0] / g["o"].iloc[0], g["r"].iloc[1] / g["o"].iloc[1]))
    if len(rows) < 6:
        return {"n": len(rows), "spearman": None}
    a, b = np.array(rows).T; from scipy.stats import spearmanr
    return {"n": len(rows), "spearman": float(spearmanr(a, b)[0])}


def main():
    m = fp.load(); dates = game_dates()
    print(f"{len(m):,} balls in play joined; grounders {int((m['set'] == 'GSL').sum()):,}; 2026 games {m['game_id'].nunique()}")
    grid = {}
    for feats in ("start", "ball"):
        for depth in (1, 2, 3, 4):
            P = run_config(m, feats, depth, dates); cell = {}
            for c in COMPS:
                cell[c] = {s: reliability(P, c, s) for s in ("alt", "chron")}
            cell["house"] = {c: house_effect(P, c) for c in ("reach", "total")}; cell["switch"] = {c: switchers(P, c) for c in ("reach", "total")}
            grid[f"{feats}|{depth}"] = cell
            f = lambda c, s: cell[c][s].get("all IF", {}).get("rel", float("nan"))
            print(f"{feats:5s} depth {depth}:  reach alt {f('reach', 'alt'):.2f} chron {f('reach', 'chron'):.2f} | convert alt {f('convert', 'alt'):.2f} chron {f('convert', 'chron'):.2f} | "
                  f"total alt {f('total', 'alt'):.2f} chron {f('total', 'chron'):.2f} | house F/null reach {cell['house']['reach']['ratio_to_null']:.2f} (p {cell['house']['reach']['p']:.3f}), "
                  f"total {cell['house']['total']['ratio_to_null']:.2f} (p {cell['house']['total']['p']:.3f}) | switchers reach {cell['switch']['reach']}", flush=True)
    json.dump(grid, open("output/fielding_rda_style_2026.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
