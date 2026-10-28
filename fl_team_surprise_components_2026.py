"""
fl_team_surprise_components_2026.py -- did specific teams' players beat or miss their K% and BB% projections in 2026?  (user 2026-10-28: "also check K% and BB% components"; companion of fl_team_surprise_2026.py)
Hitters: park-neutral 2026 K% and BB% (the panel's neutralised rates) minus the preseason projection (fl_hitter_war_calibration frame: production path run on <= 2025 data; hitters with a prior FL season and >= 50 PA in 2026), weighted by 2026 PA.
Pitchers: 2026 K% = SO / BF and BB% = BB / BF (war_pitchers) minus the cross-fitted preseason projection K_pred / BB_pred of fl_pwar_calibration (>= 40 BF in 2026), weighted by 2026 BF.
Level-free (the weighted league-wide mean miss removed); team = the team he played most for in 2026; teams with >= 4 players; 90% intervals by resampling players within the team; test of whether team differences exceed chance:
SD of the team mean misses vs 2,000 random reassignments of players to teams.  Units: percentage points of K% / BB% (positive = MORE than projected).
Output: output/team_surprise_components_2026_log.txt, output/team_surprise_components_2026.csv
"""
import numpy as np
import pandas as pd

import fl_halfseason_backtest as bt
import fl_hitter_model_dev as dev
import fl_pitcher_features as fp
import fl_position_player_flag as ppf

rng = np.random.default_rng(14)


def team_table(D, miss, w, label, min_players=4):
    D = D.copy(); D["m"] = D[miss] - np.average(D[miss], weights=D[w]); rows = []
    for t, g in D.groupby("team"):
        if len(g) < min_players:
            continue
        bs = [np.average(x["m"], weights=x[w]) for x in (g.iloc[rng.integers(0, len(g), len(g))] for _ in range(1000))]
        rows.append({"team": t, "players": len(g), "work": g[w].sum(), "points vs projection": 100 * np.average(g["m"], weights=g[w]), "90% low": 100 * np.percentile(bs, 5), "90% high": 100 * np.percentile(bs, 95)})
    T = pd.DataFrame(rows).sort_values("points vs projection", ascending=False).reset_index(drop=True); T.index += 1
    obs = D.groupby("team").apply(lambda g: np.average(g["m"], weights=g[w])).std(); sd = []
    for _ in range(2000):
        s = D.assign(team=rng.permutation(D["team"].to_numpy())); sd.append(s.groupby("team").apply(lambda g: np.average(g["m"], weights=g[w])).std())
    print(f"\n{label}: {len(D)} players, {D['team'].nunique()} teams; SD of team mean misses {100 * obs:.2f} pts vs {100 * np.mean(sd):.2f} expected by chance; P(team differences this large by chance) = {np.mean(np.array(sd) >= obs):.2f}")
    print(T.round(2).to_string(), flush=True); T["table"] = label; return T


def main():
    out = []
    h = pd.read_parquet("output/hitter_war_calibration_2026.parquet"); p = dev.load(); a = p[p["year"] == 2026].drop_duplicates("name_clean").set_index("name_clean"); h = h.join(a[["k", "bbr"]].rename(columns={"k": "k26", "bbr": "bb26"}), how="left")
    h = h[(h["pa26"] >= 50)].dropna(subset=["proj_k", "proj_bbr", "k26", "bb26"]).copy()
    hp = pd.read_parquet("2025_2026_pitches.parquet", columns=["year", "game_type", "batter", "batter_team"]); hp = hp[(hp["year"] == 2026) & (hp["game_type"] == "reg_season")]; bt_ = hp.groupby(["batter", "batter_team"]).size().rename("n").reset_index().sort_values("n", ascending=False).drop_duplicates("batter").set_index("batter")["batter_team"]
    h["team"] = h["player"].map(bt_); h = h.dropna(subset=["team"]); h["k_miss"] = h["k26"] - h["proj_k"]; h["bb_miss"] = h["bb26"] - h["proj_bbr"]
    out.append(team_table(h, "k_miss", "pa26", "HITTERS K% (points above projection = more strikeouts)")); out.append(team_table(h, "bb_miss", "pa26", "HITTERS BB% (points above projection = more walks)"))
    import fl_pwar_calibration as cal
    fp.WH_SVA_ESTIMATOR = "swstr_ct"; cal.RETURN_ALL = True; b, war = cal.main(); w26 = war[war["year"] == 2026].groupby("player").agg(so=("so", "sum"), bb=("bb", "sum"), bf=("bf", "sum"))
    b = b.join(w26, on="player", rsuffix="_26") if "bf" not in b.columns else b.merge(w26.rename(columns=lambda c: c + "_26"), left_on="player", right_index=True, how="left")
    b = b[b["bf_26"] >= 40].dropna(subset=["K_pred", "BB_pred"]).copy(); b["k_miss"] = b["so_26"] / b["bf_26"] - b["K_pred"]; b["bb_miss"] = b["bb_26"] / b["bf_26"] - b["BB_pred"]
    raw = ppf.exclude(fp.load_fl_pitches_cached()); d26 = fp.filter_period(raw, bt.START_2026, bt.END_2026)
    pt = d26.groupby(["pitcher", "pitcher_team"]).size().rename("n").reset_index().sort_values("n", ascending=False).drop_duplicates("pitcher").set_index("pitcher")["pitcher_team"]; b["team"] = b["player"].map(pt); b = b.dropna(subset=["team"])
    out.append(team_table(b, "k_miss", "bf_26", "PITCHERS K% (points above projection = more strikeouts = better)")); out.append(team_table(b, "bb_miss", "bf_26", "PITCHERS BB% (points above projection = more walks = worse)"))
    pd.concat(out).to_csv("output/team_surprise_components_2026.csv")


if __name__ == "__main__":
    main()
