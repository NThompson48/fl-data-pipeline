"""
fl_fielding_part_2026.py -- 2026 INFIELD fielding metric on the SIS PART framework (user 2026-10-14: https://library.fangraphs.com/defensive-runs-saved-2020-update/
"Can we use this framework to build a 2026 fielding metric for infielders?" + "if reliability / stickiness are reasonable ... add a shrunken component to WAR").

PART mapping to our data (2026 tracking + pbp_fielded_by):
  A  P(out | ball only: exit speed, launch angle, spray direction, batter side)                       -> feasible
  B  P(out by fielder i | ball + where fielder i STARTED: his angle and depth)                        -> feasible (starting positions on ~95% of grounders)
  C  P(out | at the fielder's touch: throw distance, time left)                                     -> NOT feasible (no touch time / location): Range and Throwing stay combined
  D  fielder i recorded the out (pbp fielded_by == i and out / fielders' choice / sacrifice; an error = no out)
  per fielder per play:  RANGE+THROW credit = D_i - p_i   (every infielder with a chance is credited / debited: the multiple-fielder evaluation, on every play since we know all starting spots)
  team POSITIONING      = sum_i p_i - A  (credited to the TEAM, not the player, as SIS does; user default 2026-10-14)
  sets: GSL = grounders (launch angle < 10); AIR = infield-range air balls (launch angle >= 10, distance <= 180 ft); each model fitted per position and per set.
  Cross-fitted 5-fold BY GAME (no play's own outcome sets its own expectation).  Batter-speed proxy (shrunk infield-hit rate 2025-26) tested: kept only if it improves
  held-out log loss of A.
Runs: plays x (average run value of the non-out outcomes of that set - run value of an out), FL run values (woba_weights).
Reliability (the pre-stated WAR rule, user 2026-10-14): split halves by ALTERNATING GAMES within 2026, runs per opportunity (opportunity = sum of p_i over his plays + plays he
  could have reached, measured as sum_i p_i), fielders with >= 15 opportunities in both halves; Spearman-Brown full-season reliability; RULE: add a shrunk fielding term to WAR
  if infielders' full-season reliability >= .40 (n0 from the split-half reliability).
Outputs: output/fielding_part_2026_plays.parquet (per play x fielder), output/fielding_part_2026_players.parquet (per fielder x position), output/fielding_part_2026_teams.parquet.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
IF = ("1B", "2B", "SS", "3B")
OUTS = {"OUT", "FC"}                         # pbp batter_result values that are an out on the play (sacrifices map to OUT in pbp)
KW = dict(max_depth=4, learning_rate=0.05, max_iter=250, min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def load():
    from sqlalchemy import text
    from fl_supabase import read_sql_df
    import fl_hitter_bridge_test as hb
    cols = ["game_id", "pa_id", "year", "inning", "batter", "batter_side", "batter_team", "home_team", "pitcher_team", "play_result", "exit_speed", "angle", "direction", "distance", "hang_time",
            "is_terminal_pitch"] + [f"pos_{p}_{a}" for p in ("1b", "2b", "ss", "3b") for a in ("x", "z")] + [f"name_{p}" for p in ("1b", "2b", "ss", "3b")]
    c = pd.read_parquet("2025_2026_pitches.parquet", columns=cols)
    c = c[(c["year"] == 2026) & c["is_terminal_pitch"].astype(str).isin(["1", "True", "true"]) & c["play_result"].isin(["out", "single", "double", "triple", "error", "fielders_choice", "sacrifice"])].copy()
    c = c[c["exit_speed"].notna() & c["angle"].notna() & c["direction"].notna()]
    c["half"] = np.where(c["batter_team"] == c["home_team"], "BOTTOM", "TOP")
    c["bn"] = c["batter"].map(lambda s: hb.norm(" ".join(reversed(str(s).split(", "))) if ", " in str(s) else s))
    f = read_sql_df(text("select game_id, inning, half, batter, batter_result, fielded_by, event_index from pbp_fielded_by")); f["bn"] = f["batter"].map(hb.norm)
    c = c.sort_values(["game_id", "pa_id"]); f = f.sort_values(["game_id", "event_index"])
    c["k"] = c.groupby(["game_id", "inning", "half", "bn"]).cumcount(); f["k"] = f.groupby(["game_id", "inning", "half", "bn"]).cumcount()
    m = c.merge(f[["game_id", "inning", "half", "bn", "k", "fielded_by", "batter_result"]], on=["game_id", "inning", "half", "bn", "k"], how="inner")
    m = m.dropna(subset=[f"pos_{p}_{a}" for p in ("1b", "2b", "ss", "3b") for a in ("x", "z")])
    m["out"] = m["batter_result"].isin(OUTS).astype(int); m["left"] = (m["batter_side"].astype(str) == "left").astype(int)
    m["set"] = np.where(m["angle"] < 10, "GSL", np.where(m["distance"] <= 180, "AIR", "OF"))
    return m[m["set"] != "OF"].reset_index(drop=True)


def fielder_frame(m, pos):
    p = pos.lower(); x, z = m[f"pos_{p}_x"], m[f"pos_{p}_z"]
    ang = np.degrees(np.arctan2(z, x)); dang = m["direction"] - ang; dep = np.hypot(x, z)
    X = pd.DataFrame({"ev": m["exit_speed"], "la": m["angle"], "dir": m["direction"], "f_ang": ang, "d_ang": dang, "f_dep": dep, "perp": dep * np.sin(np.radians(dang)),
                      "left": m["left"], "dist": m["distance"], "hang": m["hang_time"]})
    y = ((m["fielded_by"] == pos) & (m["out"] == 1)).astype(int).to_numpy(); return X, y


def crossfit(X, y, groups, folds=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        if y[tr].sum() < 20:
            continue
        p[te] = HistGradientBoostingClassifier(**KW).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return p


def logloss(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    m = load(); g = m["game_id"].to_numpy()
    print(f"2026 balls in play joined to fielded-by with starting positions: {len(m):,} (GSL {int((m['set'] == 'GSL').sum()):,}, infield-range AIR {int((m['set'] == 'AIR').sum()):,})")
    print("sign check, mean spray direction of grounders fielded by each infielder:", m[m["set"] == "GSL"].groupby("fielded_by")["direction"].mean().reindex(list(IF)).round(1).to_dict())
    # --- A: ball only (team-level expectation), with / without the batter-speed proxy
    A = np.zeros(len(m)); ll = {}
    hist = pd.read_parquet("output/fl_hitter_panel.parquet")
    for s in ("GSL", "AIR"):
        idx = np.nonzero((m["set"] == s).to_numpy())[0]; Xa = m.loc[idx, ["exit_speed", "angle", "direction", "left", "distance", "hang_time"]].reset_index(drop=True); ya = m.loc[idx, "out"].to_numpy()
        A[idx] = crossfit(Xa, ya, g[idx]); ll[s] = logloss(ya, A[idx])
    print("held-out log loss of A (ball only):", {k: round(v, 4) for k, v in ll.items()}, "(the batter-speed proxy needs a sprint / infield-hit measure per batter; tested below)")
    # batter-speed proxy: shrunk infield-hit rate is not stored per batter in the panel -> use stolen-base attempts per time on base (2025-26), shrunk
    ok = hist["year"] >= 2025; sb = hist[ok].groupby("name_clean").agg(sb=("sb", "sum"), cs=("cs", "sum"), on=("h", "sum"), bb=("bb", "sum")).fillna(0)
    rate = (sb["sb"] + sb["cs"]) / (sb["on"] + sb["bb"]).replace(0, np.nan); lg = float((sb["sb"] + sb["cs"]).sum() / (sb["on"] + sb["bb"]).sum()); nn = sb["on"] + sb["bb"]
    spd = lg + (rate.fillna(lg) - lg) * nn / (nn + 60); m["speed"] = m["bn"].map(spd).fillna(lg)
    idx = np.nonzero((m["set"] == "GSL").to_numpy())[0]; Xs = m.loc[idx, ["exit_speed", "angle", "direction", "left", "distance", "hang_time", "speed"]].reset_index(drop=True)
    As = crossfit(Xs, m.loc[idx, "out"].to_numpy(), g[idx]); ll_s = logloss(m.loc[idx, "out"].to_numpy(), As)
    use_speed = ll_s < ll["GSL"] - 0.0005
    print(f"GSL log loss with the speed proxy {ll_s:.4f} vs {ll['GSL']:.4f} without -> {'KEEP' if use_speed else 'drop'} the proxy")
    if use_speed:
        A[idx] = As
    # --- B: per fielder, per set
    rows = []
    for s in ("GSL", "AIR"):
        idx = np.nonzero((m["set"] == s).to_numpy())[0]; ms = m.loc[idx].reset_index(drop=True)
        for pos in IF:
            X, y = fielder_frame(ms, pos)
            if use_speed and s == "GSL":
                X["speed"] = ms["speed"].to_numpy()
            p = crossfit(X, y, ms["game_id"].to_numpy())
            name = ms[f"name_{pos.lower()}"].astype(str)
            rows.append(pd.DataFrame({"row": idx, "set": s, "pos": pos, "fielder": name.to_numpy(), "team": ms["pitcher_team"].to_numpy(), "game_id": ms["game_id"].to_numpy(),
                                      "p": p, "d": y, "A": A[idx]}))
            print(f"  {s} {pos}: plays made {int(y.sum()):,}, mean expected {p.mean():.3f}, held-out log loss {logloss(y, p):.4f}", flush=True)
        # pitcher / catcher outs: modelled so the expected-out total B closes against A (team positioning), never credited to an infielder
        Xo = ms[["exit_speed", "angle", "direction", "left", "distance", "hang_time"]].reset_index(drop=True); yo = (ms["fielded_by"].isin(["P", "C"]) & (ms["out"] == 1)).astype(int).to_numpy()
        po = crossfit(Xo, yo, ms["game_id"].to_numpy())
        rows.append(pd.DataFrame({"row": idx, "set": s, "pos": "P/C", "fielder": "(pitcher / catcher)", "team": ms["pitcher_team"].to_numpy(), "game_id": ms["game_id"].to_numpy(), "p": po, "d": yo, "A": A[idx]}))
    P = pd.concat(rows, ignore_index=True)
    # runs per play: average run value of the non-out outcomes of the set minus an out
    from sqlalchemy import text
    from fl_supabase import read_sql_df
    w = read_sql_df(text("select outcome, rv_above_avg, out_value from woba_weights")); rv = dict(zip(w["outcome"], w["rv_above_avg"])); rv_out = float(w["out_value"].iloc[0])
    val = {}
    for s in ("GSL", "AIR"):
        nonout = m[(m["set"] == s) & (m["out"] == 0)]["play_result"].map({"single": "1B", "double": "2B", "triple": "3B", "error": "1B", "fielders_choice": "1B", "sacrifice": "1B"})
        val[s] = float(nonout.map(rv).mean() - rv_out)
    print("runs per play saved (non-out value - out value):", {k: round(v, 3) for k, v in val.items()})
    P["plays"] = P["d"] - P["p"]; P["runs_raw"] = P["plays"] * P["set"].map(val); P["opp"] = P["p"]
    # centre within position and set: each position's average fielder = 0 per opportunity (cross-fitted expectations run slightly low; also rules out any double count
    # with the positional adjustment)
    mrate = P.groupby(["pos", "set"]).apply(lambda q: q["runs_raw"].sum() / q["opp"].sum()); P["runs"] = P["runs_raw"] - P["opp"] * [mrate[(a_, b_)] for a_, b_ in zip(P["pos"], P["set"])]
    P.to_parquet("output/fielding_part_2026_plays.parquet", index=False)
    # team positioning: sum_i p_i - A per play (per set), credited to the team
    tp = P.groupby(["row", "set", "team"]).agg(B=("p", "sum"), A=("A", "first")).reset_index(); tp["pos_plays"] = tp["B"].clip(upper=1) - tp["A"]; tp["pos_runs"] = tp["pos_plays"] * tp["set"].map(val)
    T = tp.groupby("team").agg(plays=("row", "size"), positioning_plays=("pos_plays", "sum"), positioning_runs=("pos_runs", "sum")).sort_values("positioning_runs"); T.to_parquet("output/fielding_part_2026_teams.parquet")
    print("\nTEAM POSITIONING runs (2026; positive = positioned to make more outs than a ball-only expectation):\n" + T.round(1).to_string())
    # players
    Pl = P.groupby(["fielder", "pos"]).agg(team=("team", lambda t: "/".join(sorted(set(t)))), opps=("opp", "sum"), made=("d", "sum"), plays=("plays", "sum"), runs=("runs", "sum"),
                                         gsl_runs=("runs", lambda r: r[P.loc[r.index, "set"] == "GSL"].sum()), air_runs=("runs", lambda r: r[P.loc[r.index, "set"] == "AIR"].sum()), games=("game_id", "nunique")).reset_index()
    Pl = Pl[Pl["fielder"].ne("nan") & Pl["fielder"].ne("None") & Pl["pos"].isin(IF)]
    Pl.to_parquet("output/fielding_part_2026_players.parquet", index=False)
    print(f"\nby position (sum of runs is ~0 by construction; check):\n" + Pl.groupby("pos").agg(fielders=("fielder", "size"), opps=("opps", "sum"), runs=("runs", "sum"), sd_runs=("runs", "std")).round(2).to_string())
    print("\ntop / bottom infielders by runs (>= 40 opportunities):")
    q = Pl[Pl["opps"] >= 40].sort_values("runs"); print(pd.concat([q.tail(8), q.head(8)])[["fielder", "pos", "team", "games", "opps", "made", "plays", "runs", "gsl_runs", "air_runs"]].round(1).to_string(index=False))
    # --- reliability: alternating games within each fielder-position
    gm = P[["fielder", "pos", "game_id"]].drop_duplicates().sort_values("game_id"); gm["half"] = gm.groupby(["fielder", "pos"]).cumcount() % 2
    H = P.merge(gm, on=["fielder", "pos", "game_id"]).groupby(["fielder", "pos", "half"]).agg(opps=("opp", "sum"), runs=("runs", "sum")).unstack("half")
    H = H.dropna(); H = H[(H[("opps", 0)] >= 15) & (H[("opps", 1)] >= 15)]
    ra, rb = H[("runs", 0)] / H[("opps", 0)], H[("runs", 1)] / H[("opps", 1)]
    r = float(np.corrcoef(ra, rb)[0, 1]); nh = float(((H[("opps", 0)] + H[("opps", 1)]) / 2).mean()); full = 2 * r / (1 + r); n0 = nh * (1 - r) / r if r > 0 else np.inf
    print(f"\nRELIABILITY (alternating games, runs per opportunity, {len(H)} fielder-positions with >= 15 opportunities per half): split-half r {r:.3f}, full-season (Spearman-Brown) {full:.3f}, "
          f"implied n0 {n0:.0f} opportunities (mean opportunities per half {nh:.0f})")
    byp = {}
    for pos, hh in H.groupby(level="pos"):
        if len(hh) >= 10:
            rr = np.corrcoef(hh[("runs", 0)] / hh[("opps", 0)], hh[("runs", 1)] / hh[("opps", 1)])[0, 1]; byp[pos] = (len(hh), round(rr, 3))
    print("  by position (n, split-half r):", byp)
    H = H[H.index.get_level_values("pos").isin(IF)]
    full_time = Pl[Pl["pos"].isin(IF)].groupby("pos")["opps"].quantile(0.9).to_dict()             # a full-time fielder's opportunities at each position (90th percentile)
    rng = np.random.default_rng(5); rel = {}
    for pos, hh in list(H.groupby(level="pos")) + [("all IF", H)]:
        a_, b_ = (hh[("runs", 0)] / hh[("opps", 0)]).to_numpy(), (hh[("runs", 1)] / hh[("opps", 1)]).to_numpy(); nh_ = float(((hh[("opps", 0)] + hh[("opps", 1)]) / 2).mean())
        rs = [np.corrcoef(a_[i], b_[i])[0, 1] for i in (rng.integers(0, len(a_), len(a_)) for _ in range(2000))]
        r_ = float(np.corrcoef(a_, b_)[0, 1]); n0_ = nh_ * (1 - r_) / r_ if r_ > 0 else np.inf
        ft = full_time.get(pos, float(np.mean(list(full_time.values())))); rel_ft = ft / (ft + n0_) if np.isfinite(n0_) else 0.0
        rel[pos] = {"n": len(hh), "split_half_r": r_, "r_90%": [float(np.nanpercentile(rs, 5)), float(np.nanpercentile(rs, 95))], "n0": n0_, "full_time_opps": ft, "reliability_full_time": rel_ft}
    print("\nRELIABILITY at a FULL-TIME season (n0 from the split halves; full-time = 90th percentile of opportunities at the position):")
    print(pd.DataFrame(rel).T.round(3).to_string())
    full = rel["all IF"]["reliability_full_time"]; n0 = rel["all IF"]["n0"]
    rule = full >= 0.40
    print(f"WAR RULE (pre-stated: full-season reliability >= .40): {'MET' if rule else 'NOT met'}")
    json.dump({"split_half_r": r, "full_season_reliability": full, "n0_opportunities": n0, "runs_per_play": val, "use_speed_proxy": bool(use_speed), "rule_met": bool(rule), "by_position": {k: {kk: (vv if not isinstance(vv, float) or np.isfinite(vv) else None) for kk, vv in v.items()} for k, v in rel.items()}},
              open("output/fielding_part_2026_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
