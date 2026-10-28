"""
fl_team_surprise_2026.py -- did specific teams' players outperform / underperform their (preseason, <= 2025 data) projections in 2026?  (user 2026-10-28)
Hitters: park-neutral 2026 wOBA minus the preseason projection (fl_hitter_war_calibration frame: production path run on <= 2025 data; hitters with a prior FL season, >= 50 PA in 2026), level-free (the PA-weighted mean miss removed),
runs vs projection = sum of (miss) / 1.0175 x 2026 PA.   Pitchers: 2026 RA9 vs the preseason RA9 projection (fl_pwar_calibration frame; >= 20 IP in 2026), level-free, runs saved vs projection = sum of (projected - actual RA9) x IP / 9
(RA9 includes the pitcher's park, defense and BABIP luck: it is team CONTEXT as much as pitcher skill).  Team = the team he played most for in 2026.  90% intervals by resampling players within the team; test of whether team differences exceed chance: SD of the team means vs 2,000 random reassignments of players to teams.
Output: output/team_surprise_2026_log.txt, output/team_surprise_2026.csv
"""
import numpy as np
import pandas as pd

import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf

SC = 1.0175
rng = np.random.default_rng(4)


def team_table(D, val_col, w_col, to_runs, label, min_players=4):
    D = D.copy(); mu = np.average(D[val_col], weights=D[w_col]); D["m"] = D[val_col] - mu; D["runs"] = D["m"] * D[w_col] * to_runs; rows = []
    for t, g in D.groupby("team"):
        if len(g) < min_players:
            continue
        bs = [g.iloc[rng.integers(0, len(g), len(g))]["runs"].sum() for _ in range(1000)]; rows.append({"team": t, "players": len(g), "work": g[w_col].sum(), "runs vs projection": g["runs"].sum(), "90% low": np.percentile(bs, 5), "90% high": np.percentile(bs, 95)})
    T = pd.DataFrame(rows).sort_values("runs vs projection", ascending=False).reset_index(drop=True); T.index += 1
    obs = D.groupby("team").apply(lambda g: np.average(g["m"], weights=g[w_col])).std(); sd = []
    for _ in range(2000):
        s = D.assign(team=rng.permutation(D["team"].to_numpy())); sd.append(s.groupby("team").apply(lambda g: np.average(g["m"], weights=g[w_col])).std())
    print(f"\n{label}: {len(D)} players, {D['team'].nunique()} teams; SD of team mean misses {obs:.4f} vs {np.mean(sd):.4f} expected by chance; P(team differences this large by chance) = {np.mean(np.array(sd) >= obs):.2f}")
    print(T.round(1).to_string()); return T


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); d26 = fp.filter_period(raw, bt.START_2026, bt.END_2026)
    # hitters
    h = pd.read_parquet("output/hitter_war_calibration_2026.parquet"); h = h[(h["pa26"] >= 50)].dropna(subset=["proj_woba", "woba26"]).copy()
    hp = pd.read_parquet("2025_2026_pitches.parquet", columns=["year", "game_type", "batter", "batter_team"]); hp = hp[(hp["year"] == 2026) & (hp["game_type"] == "reg_season")]; bt_ = hp.groupby(["batter", "batter_team"]).size().rename("n").reset_index().sort_values("n", ascending=False).drop_duplicates("batter").set_index("batter")["batter_team"]; h["team"] = h["player"].map(bt_)
    h = h.dropna(subset=["team"]); h["miss"] = h["woba26"] - h["proj_woba"]; TH = team_table(h.rename(columns={"pa26": "pa"}), "miss", "pa", 1 / SC, "HITTERS (2026 wOBA vs projection; runs over the 2026 PA)")
    # pitchers
    import fl_pwar_calibration as cal
    fp.WH_SVA_ESTIMATOR = "swstr_ct"; cal.RETURN_ALL = True; b, war = cal.main(); b = b[(b["ip"] >= 20)].dropna(subset=["pred", "target"]).copy()
    pt = d26.groupby(["pitcher", "pitcher_team"]).size().rename("n").reset_index().sort_values("n", ascending=False).drop_duplicates("pitcher").set_index("pitcher")["pitcher_team"]; b["team"] = b["player"].map(pt); b = b.dropna(subset=["team"])
    b["miss"] = b["pred"] - b["target"]; TP = team_table(b.rename(columns={"ip": "ipw"}), "miss", "ipw", 1 / 9, "PITCHERS (projected RA9 minus actual RA9, runs saved over the 2026 IP; positive = better than projected)")
    TH["group"] = "hitters"; TP["group"] = "pitchers"; pd.concat([TH, TP]).to_csv("output/team_surprise_2026.csv")


if __name__ == "__main__":
    main()
