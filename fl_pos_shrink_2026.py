"""
fl_pos_shrink_2026.py -- position-runs shrinkage (n0) and residual from 2026 ALONE (user 2026-10-14: "check on how we can do shrinkage with just 2026").

Production: pos_rate = (PA x own 2026 position runs / PA + n0 x group rate) / (PA + n0), n0 = 100; WAR simulation residual 6.0 runs / 600 PA -- both assumed (only 2026 has
real positions).  This study estimates them inside 2026.

1. Per-game position runs rebuilt from pbp_events (2026): each event's outs credited to that moment's C (catcher_name), 1B, 2B, 3B, SS, LF, CF, RF -> innings by position;
   DH PA = a hitter's PA in a game where he never appears in his team's alignment.  Runs = the published formula, reverse-engineered on war_players (fl DECISION_LOG
   2026-10-14): FanGraphs constants x innings / 1458 x INN_SCALE + DH -17.2 per 600 DH PA (+ a uniform per-PA shift, irrelevant to shrinkage).  Season totals checked
   against war_players.pos_adj_runs.
2. CHRONOLOGICAL split (league date median): first-half own rate and group (DH if >= 50% of first-half PA at DH, else the position with most first-half innings) predict
   the second-half position runs per PA; hitters with >= MIN_PA PA in each half.  Models: group only, own only, blend for an n0 grid.  Frozen inverse-variance weights
   (fl_eval_weights, fitted once on the group-only residuals); fl_eval_report.compare vs n0 = 100; player bootstrap of the best n0.
   Residual: variance law a + b / PA on the best blend's second-half errors -> sqrt(a) x 600 = role-change spread per 600 PA (compare with the 6.0 used).
3. ALTERNATING-games split (odd / even games per player): pure sampling noise, no time -> the other bound.
CAVEAT: roles change more between seasons (signings, trades, aging, offseason moves) than within one, so the chronological n0 and residual are LOWER bounds for a
season-ahead projection; the alternating split is lower still.
"""
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import text

import fl_hitter_bridge_test as hb
from fl_eval_report import compare, table
from fl_eval_weights import fit_variance_law, inv_var_weights
from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
FG = {"C": 12.5, "SS": 7.5, "2B": 2.5, "3B": 2.5, "CF": 2.5, "LF": -7.5, "RF": -7.5, "1B": -12.5}
DH_PER600 = -17.2
SLOTS = {"catcher_name": "C", "pos_1b": "1B", "pos_2b": "2B", "pos_3b": "3B", "pos_ss": "SS", "pos_lf": "LF", "pos_cf": "CF", "pos_rf": "RF"}
MIN_PA = 50
GRID = [0, 25, 50, 100, 200, 400, 800]


def per_game():
    e = read_sql_df(text("select game_guid, date, batting_side, fielding_side, batter_name, is_pa_end, outs_on_play, " + ", ".join(SLOTS) + " from pbp_events"))
    e["outs_on_play"] = e["outs_on_play"].fillna(0)
    # fielding outs by player x position x game x side
    f = pd.concat([e[["game_guid", "date", "fielding_side", "outs_on_play", c]].rename(columns={c: "name"}).assign(pos=p) for c, p in SLOTS.items()])
    f = f[f["name"].notna()]; f["name_clean"] = f["name"].map(hb.norm)
    fo = f.groupby(["name_clean", "game_guid", "date", "fielding_side", "pos"])["outs_on_play"].sum().reset_index()
    # PA by batter x game x side; DH = never in his side's alignment that game
    pa = e[e["is_pa_end"].astype(bool)].assign(name_clean=lambda q: q["batter_name"].map(hb.norm)).groupby(["name_clean", "game_guid", "date", "batting_side"]).size().rename("pa").reset_index()
    fielded = set(zip(fo["name_clean"], fo["game_guid"], fo["fielding_side"]))
    pa["dh"] = [(n, g, s) not in fielded for n, g, s in zip(pa["name_clean"], pa["game_guid"], pa["batting_side"])]
    inn = fo.pivot_table(index=["name_clean", "game_guid", "date"], columns="pos", values="outs_on_play", aggfunc="sum").fillna(0.0) / 3
    g = pa.groupby(["name_clean", "game_guid", "date"]).agg(pa=("pa", "sum"), dh_pa=("pa", lambda x: x[pa.loc[x.index, "dh"]].sum())).join(inn, how="outer").fillna(0.0).reset_index()
    return g


def runs(g, scale):
    return scale * sum(FG[p] * g[p] for p in FG) / 1458 + DH_PER600 * g["dh_pa"] / 600


def group_of(h):
    inn = h[list(FG)]; return np.where(h["dh_pa"] >= 0.5 * h["pa"], "DH", inn.idxmax(axis=1).where(inn.sum(axis=1) > 0, "none"))


def study(g, split_name, half_of):
    g = g.assign(half=half_of(g))
    H = g.groupby(["name_clean", "half"])[["pa", "dh_pa", "runs"] + list(FG)].sum().unstack("half")
    A = H.xs(0, axis=1, level="half"); B = H.xs(1, axis=1, level="half")
    ok = (A["pa"] >= MIN_PA) & (B["pa"] >= MIN_PA); A, B = A[ok], B[ok]
    A = A.assign(grp=group_of(A)); own = A["runs"] / A["pa"]
    grate = (A.groupby("grp")["runs"].sum() / A.groupby("grp")["pa"].sum()); gr = A["grp"].map(grate)
    y = (B["runs"] / B["pa"]).to_numpy(); n = A["pa"].to_numpy(float); nb = B["pa"].to_numpy(float); ids = A.index.to_numpy()
    a, b = fit_variance_law(y - gr.to_numpy(), nb); w = inv_var_weights(nb, a, b)
    pred = {n0: ((n * own + n0 * gr) / (n + n0)).to_numpy() for n0 in GRID}; pred["group only"] = gr.to_numpy(); pred["own only"] = own.to_numpy()
    rm = {k: float(np.sqrt(np.average((y - v) ** 2, weights=w))) for k, v in pred.items()}
    fine = np.arange(0, 2001, 10); rf = [np.sqrt(np.average((y - (n * own + k * gr) / (n + k)) ** 2, weights=w)) for k in fine]; best = int(fine[int(np.argmin(rf))])
    rng = np.random.default_rng(3); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y)); r_ = [np.sqrt(np.average((y[i] - ((n * own + k * gr) / (n + k)).to_numpy()[i]) ** 2, weights=w[i])) for k in fine]; bs.append(fine[int(np.argmin(r_))])
    print(f"\n== {split_name}: {ok.sum()} hitters with >= {MIN_PA} PA in each half; first-half groups {A['grp'].value_counts().to_dict()}")
    print(f"   group changed between halves for {np.mean(A['grp'].to_numpy() != group_of(B)):.0%} of hitters")
    print("   weighted RMSE of 2nd-half position runs per 600 PA:", {str(k): round(600 * v, 3) for k, v in rm.items()})
    print(f"   best n0 {best} PA (player bootstrap 90%: {np.percentile(bs, 5):.0f} to {np.percentile(bs, 95):.0f})")
    best_p = ((n * own + best * gr) / (n + best)).to_numpy()
    print(table([{"comparison": f"n0 100 -> n0 {best}", **compare(600 * y, 600 * pred[100], 600 * best_p, w, ids, n=2000)},
                 {"comparison": "n0 100 -> group only", **compare(600 * y, 600 * pred[100], 600 * pred["group only"], w, ids, n=2000)}], 3))
    ra, rb = fit_variance_law(y - best_p, nb)
    print(f"   residual around the best blend: role-change spread sqrt(a) = {600 * np.sqrt(ra):.2f} runs / 600 PA (WAR simulation uses 6.0); sampling part at 300 PA "
          f"{600 * np.sqrt(rb / 300):.2f}")
    return best, bs, 600 * np.sqrt(ra)


def main():
    g = per_game()
    tot = g.groupby("name_clean")[["pa", "dh_pa"] + list(FG)].sum()
    w = read_sql_df(text("select player, pa, pos_adj_runs from war_players where game_type = 'reg_season' and year = 2026"))
    w["name_clean"] = w["player"].map(lambda s: hb.norm(" ".join(reversed(str(s).split(", "))) if ", " in str(s) else s)); w = w.groupby("name_clean")[["pa", "pos_adj_runs"]].sum()
    j = tot.join(w, how="inner", rsuffix="_w"); j = j[j["pa_w"] >= 50]
    base = sum(FG[p] * j[p] for p in FG) / 1458
    X = np.column_stack([base, j["dh_pa"], j["pa_w"]]); beta, *_ = np.linalg.lstsq(X, j["pos_adj_runs"].to_numpy(), rcond=None); r = j["pos_adj_runs"] - X @ beta
    print(f"rebuild check vs war_players (>= 50 PA, {len(j)} hitters): innings scale {beta[0]:.3f}, DH {600 * beta[1]:+.1f} per 600 DH PA, shift {600 * beta[2]:+.2f} per 600 PA; "
          f"R2 {1 - r.var() / j['pos_adj_runs'].var():.4f}, max miss {r.abs().max():.2f} runs; PA match {np.corrcoef(j['pa'], j['pa_w'])[0, 1]:.4f}")
    g["runs"] = runs(g, beta[0])
    dates = g["date"].astype(str); mid = np.sort(dates.unique())[len(dates.unique()) // 2]
    study(g, f"CHRONOLOGICAL (before / from {mid})", lambda q: (q["date"].astype(str) >= mid).astype(int))
    study(g, "ALTERNATING GAMES", lambda q: q.sort_values("date").groupby("name_clean").cumcount().reindex(q.index) % 2)


if __name__ == "__main__":
    main()
