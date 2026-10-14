"""
fl_ros_workload.py -- rest-of-season workload (PA for hitters, batters faced for pitchers) = rate per TEAM game x team games remaining.

  rate_ytd   = (PA or BF so far) / (team games since the player first appeared this season)       (call-ups are not charged for games before they arrived)
  rate_prior = last season's PA or BF per team game over the full season (same team-game denominator), if he played
  rate_hat   = (G_eff rate_ytd + k rate_prior') / (G_eff + k)       G_eff = team games since he first appeared;  rate_prior' = rate_prior (or the league mean rate if none)
  ROS workload = rate_hat x (team games remaining)             -- unconditional (it includes the chance he is hurt / released: part-timers and fringe arms carry low rates)
Season length: the observed final team-game count per team (backtest); a live run takes a schedule or a default (SEASON_GAMES_DEFAULT).
Backtest: 2025 and 2026, five chronological cutoffs; players with any PA / BF in the season after the cutoff or before it (everyone active at the cutoff); error in PA / BF over the rest of the season.
Usage:  python fl_ros_workload.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fl_hitter_ros as hr
import fl_pitcher_ros as pr

warnings.filterwarnings("ignore")
CUTS = (0.15, 0.30, 0.45, 0.60, 0.75)
SEASON_GAMES_DEFAULT = 102
TG = pd.read_parquet("output/team_games_2025_2026.parquet")


def team_games(year):
    return TG[TG["year"] == year].groupby("team")["date"].apply(lambda s: np.sort(pd.to_datetime(s).to_numpy()))


def games_between(tg, team, start, end):
    arr = tg.get(team)
    if arr is None:
        return 0
    return int(((arr >= np.datetime64(start)) & (arr < np.datetime64(end))).sum())


def workload_frame(pa, key, team_col, year, asof, tg_cur, tg_prev):
    """Per player: YTD count, games since first appearance, prior-season count and team games, ROS actual count, games remaining."""
    cur = pa[pa["year"] == year]
    ytd = cur[cur["date"] < asof]; ros = cur[cur["date"] >= asof]
    first = ytd.groupby(key)["date"].min(); last_team = cur.sort_values("date").groupby(key)[team_col].last(); n_ytd = ytd.groupby(key).size()
    n_ros = ros.groupby(key).size(); last_date = ytd.groupby(key)["date"].max()
    idx = n_ytd.index.union(n_ros.index)
    f = pd.DataFrame({"n_ytd": n_ytd.reindex(idx).fillna(0), "n_ros": n_ros.reindex(idx).fillna(0)})
    f["team"] = last_team.reindex(idx)
    f["first"] = first.reindex(idx); f["last_date"] = last_date.reindex(idx)
    t0 = cur["date"].min()
    f["g_eff"] = [games_between(tg_cur, tm, fs if pd.notna(fs) else asof, asof) for tm, fs in zip(f["team"], f["first"])]
    f["g_rem"] = [games_between(tg_cur, tm, asof, pd.Timestamp("2100-01-01")) for tm in f["team"]]
    prv = pa[pa["year"] == year - 1]
    if len(prv):
        pc = prv.groupby(key).size(); ptm = prv.sort_values("date").groupby(key)[team_col].last(); f["n_prior"] = pc.reindex(idx).fillna(0)
        f["g_prior"] = [len(tg_prev.get(tm, [])) if pd.notna(tm) else 0 for tm in ptm.reindex(idx)]
    else:
        f["n_prior"], f["g_prior"] = 0.0, 0
    f["days_since"] = (asof - f["last_date"]).dt.days
    return f


def predict_workload(f, k, lg_rate):
    r_ytd = np.where(f["g_eff"] > 0, f["n_ytd"] / f["g_eff"].replace(0, np.nan), lg_rate)
    r_prior = np.where((f["g_prior"] > 0) & (f["n_prior"] > 0), f["n_prior"] / f["g_prior"].replace(0, np.nan), lg_rate)
    r_hat = (f["g_eff"] * r_ytd + k * r_prior) / (f["g_eff"] + k)
    return r_hat * f["g_rem"], r_ytd * f["g_rem"], r_prior * f["g_rem"]


ACTIVE_DAYS = 21


def fit_activity(d, k):
    """Out-of-sample-able constants: scale c on the active group (appeared within ACTIVE_DAYS) and the mean ROS workload of the inactive group, fit on a training season."""
    p, _, _ = predict_workload(d, k, d["lg_rate"]); act = d["days_since"] <= ACTIVE_DAYS
    c = float(d.loc[act, "n_ros"].sum() / max(p[act].sum(), 1e-9)); inactive_mean = float(d.loc[~act, "n_ros"].mean())
    return c, inactive_mean


def predict_final(f, k, c, inactive_mean):
    p, _, _ = predict_workload(f, k, f["lg_rate"]); act = f["days_since"] <= ACTIVE_DAYS
    return np.where(act, c * p, inactive_mean)


def collect(T_h, T_p):
    out = {"hit": [], "pit": []}
    for tag, pa, key, team_col in (("hit", T_h.pa, "name_clean", "batter_team"), ("pit", T_p.pa, "name_clean", "pitcher_team")):
        for year in (2025, 2026):
            tg_cur, tg_prev = team_games(year), team_games(year - 1)
            t0, t1 = pa[pa["year"] == year]["date"].min(), pa[pa["year"] == year]["date"].max()
            for c in CUTS:
                asof = t0 + (t1 - t0) * c
                f = workload_frame(pa, key, team_col, year, asof, tg_cur, tg_prev); f = f[(f["n_ytd"] > 0)]       # everyone who had appeared by the cutoff
                f["year"], f["cut"] = year, c; f["lg_rate"] = (f["n_ytd"].sum() / max(f["g_eff"].sum(), 1))
                out[tag].append(f.reset_index().rename(columns={"index": "name_clean"}))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def main():
    T_h = hr.Tables(); T_p = pr.Tables(); D = collect(T_h, T_p)
    for tag, lab in (("hit", "HITTERS (PA)"), ("pit", "PITCHERS (BF)")):
        d = D[tag]
        print(f"\n{'=' * 90}\n{lab}: {len(d)} player-cutoffs (everyone who had appeared by the cutoff; zeros in the ROS window included)")
        for k in (0, 5, 10, 20, 40, 80):
            pass
        rows = []
        for train, test in ((2025, 2026), (2026, 2025)):
            tr, te = d[d["year"] == train], d[d["year"] == test]
            best = None
            for k in (2, 5, 10, 20, 40, 80, 160):
                p, _, _ = predict_workload(tr, k, tr["lg_rate"]); e = float(np.sqrt(np.mean((tr["n_ros"] - p) ** 2)))
                if best is None or e < best[0]:
                    best = (e, k)
            k = best[1]; p_blend, p_ytd, p_prior = predict_workload(te, k, te["lg_rate"])
            lgp = te["lg_rate"] * te["g_rem"]
            rm = lambda p: float(np.sqrt(np.mean((te["n_ros"] - p) ** 2))); mae = lambda p: float(np.mean(np.abs(te["n_ros"] - p)))
            rows.append({"fit -> score": f"{train} -> {test}", "k fitted": k, "mean actual ROS": te["n_ros"].mean(), "RMSE league-rate": rm(lgp), "RMSE YTD rate": rm(p_ytd), "RMSE prior-season rate": rm(p_prior), "RMSE blend": rm(p_blend),
                         "MAE YTD": mae(p_ytd), "MAE blend": mae(p_blend), "bias blend": float(np.mean(te["n_ros"] - p_blend))})
        print(pd.DataFrame(rows).round(2).to_string(index=False))
        # by cutoff with the forward fit
        tr, te = d[d["year"] == 2025], d[d["year"] == 2026]
        k = rows[0]["k fitted"]; p_blend, p_ytd, _ = predict_workload(te, k, te["lg_rate"])
        print("   2026 RMSE by cutoff (YTD rate / blend):", {c: (round(float(np.sqrt(np.mean((te[te.cut == c]["n_ros"] - p_ytd[te.cut == c]) ** 2))), 1), round(float(np.sqrt(np.mean((te[te.cut == c]["n_ros"] - p_blend[te.cut == c]) ** 2))), 1)) for c in CUTS})
        # activity rule fit on the other season
        rows2 = []
        for train, test in ((2025, 2026), (2026, 2025)):
            tr, te = d[d["year"] == train], d[d["year"] == test]; k = [r for r in rows if r["fit -> score"] == f"{train} -> {test}"][0]["k fitted"]
            c, im = fit_activity(tr, k); pf = predict_final(te, k, c, im); pb_, _, _ = predict_workload(te, k, te["lg_rate"])
            rows2.append({"fit -> score": f"{train} -> {test}", "scale c (active)": c, "inactive mean": im, "RMSE blend": float(np.sqrt(np.mean((te["n_ros"] - pb_) ** 2))), "RMSE with activity rule": float(np.sqrt(np.mean((te["n_ros"] - pf) ** 2))),
                          "MAE blend": float(np.mean(np.abs(te["n_ros"] - pb_))), "MAE with rule": float(np.mean(np.abs(te["n_ros"] - pf))), "bias before": float(np.mean(te["n_ros"] - pb_)), "bias after": float(np.mean(te["n_ros"] - pf))})
        print("   WITH THE ACTIVITY RULE (appeared in the last 21 days; scale and inactive level fit on the other season):"); print(pd.DataFrame(rows2).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
