"""
fl_bsr_proxy_validation.py -- model-change report for adding the baserunning PROXY (position + height + weight, fl_bsr_proxy.py) to hitter WAR (user 2026-10-14).

(a) Baserunning rate itself, 2026 war_players: 10-fold cross-validation BY PLAYER (repeated 20x with different splits, averaged), out-of-fold predictions from
    league mean (0) / position only / position + height + weight.  Frozen inverse-variance weights w = 1 / (a + b / PA), (a, b) fitted once on the league-mean
    residuals (fl_eval_weights).  Reported with fl_eval_report.compare (player bootstrap).
(b) WAR backtest 2025-and-earlier -> 2026 (fl_hitter_pre2026.build_shifted, the production path), hitters with >= 20 PA in 2026.  Point WAR = projected PA x rate.
    old = bat + position + replacement (current production); new = + proxy baserunning, CROSS-FITTED: each hitter's proxy comes from a fit that excludes his fold
    (10 folds by player), so his own 2026 baserunning never enters his prediction.  POSITION CAVEAT: the backtest's "2025" positions are the 2026 positions copied back (lineup_stats
    has no real 2025 positions; war_players 2025 dh_pa / pos_innings are broken), so both old and new use target-season ROLE (position adjustment too), and the proxy
    uses the 2026 DH flag (fl_bsr_proxy.positions(2026)).  Role leaks; baserunning outcomes do not.
    Target: realised 2026 WAR = batting + position + replacement + realised bsr_runs (the new definition).  Weights: realised PA (the fielding-drop report's
    convention for WAR), hitter bootstrap.
"""
import warnings

import numpy as np
import pandas as pd

import fl_bsr_proxy as bp
from fl_eval_report import compare, table
from fl_eval_weights import fit_variance_law, inv_var_weights

warnings.filterwarnings("ignore")


def oof(d, features, folds=10, reps=20, seed=0):
    rng = np.random.default_rng(seed); out = np.zeros(len(d))
    for _ in range(reps):
        f = rng.permutation(np.arange(len(d)) % folds)
        for k in range(folds):
            tr, te = f != k, f == k
            m = bp.fit(d[tr], features); out[te] += bp.predict(m, d.loc[te, "pos"], d.loc[te, "ht"], d.loc[te, "wt"])
    return out / reps


def part_a():
    d = bp.training_data().reset_index(); y = (d["bsr"] / d["pa"]).to_numpy(); n = d["pa"].to_numpy(float)
    lg = np.zeros(len(d))
    tr_lg = np.zeros(len(d)); rng = np.random.default_rng(0); f = rng.permutation(np.arange(len(d)) % 10)
    for k in range(10):
        tr_lg[f == k] = np.average(y[f != k], weights=n[f != k])               # out-of-fold league mean
    a, b = fit_variance_law(y - tr_lg, n); w = inv_var_weights(n, a, b)
    print(f"(a) 2026 baserunning per PA, {len(d)} hitters; frozen weights from the league-mean residuals: a {a:.3e}, b {b:.3e} (n0_eval {b / a:.0f} PA)")
    p_pos = oof(d, ("pos",)); p_all = oof(d, ("pos", "ht", "wt")); p_hw = oof(d, ("ht", "wt"))
    s = 600.0; ids = d["name_clean"].to_numpy(); rows = []
    for lab, old, new in (("league mean -> position", tr_lg, p_pos), ("league mean -> height + weight", tr_lg, p_hw), ("league mean -> position + height + weight", tr_lg, p_all),
                          ("position -> position + height + weight", p_pos, p_all)):
        rows.append({"comparison (runs per 600 PA)": lab, **compare(s * y, s * old, s * new, w, ids, n=2000)})
    print(table(rows, 4))


def part_b(features=("pos", "ht", "wt")):
    import fl_hitter_baseline_backtest as bt
    import fl_hitter_model_dev as dev
    import fl_hitter_pre2026 as pre
    import fl_hitter_projection as hp
    from sqlalchemy import text
    from fl_supabase import read_sql_df
    dproj, c, fit = pre.build_shifted()
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; lg26 = float(bt.league(q, "woba")[2026])
    act = p[p["year"] == 2026].drop_duplicates("name_clean").set_index("name_clean")
    w26 = read_sql_df(text("select player, pa, bsr_runs from war_players where game_type = 'reg_season' and year = 2026"))
    w26["name_clean"] = w26["player"].map(bp._clean_lastfirst); bsr26 = w26.groupby("name_clean")["bsr_runs"].sum()
    j = dproj.set_index("name_clean").join(act[["pa", "woba_n", "pos_adj_runs"]].rename(columns={"pa": "pa26", "woba_n": "woba26", "pos_adj_runs": "pos26"}), how="inner")
    j = j[(j["pa26"] >= 20) & j["proj_woba"].notna()].copy(); j["bsr26"] = bsr26.reindex(j.index).fillna(0.0)
    j = j.join(bp.bio(), how="left"); j["bsr_pos"] = bp.positions(2026)["pos"].reindex(j.index).fillna(j["primary_pos"])
    # cross-fitted proxy: training rows = 2026 war_players hitters; folds by player name, shared between the training table and the backtest rows
    tr = bp.training_data(); names = sorted(set(tr.index) | set(j.index)); fold = pd.Series(np.random.default_rng(1).permutation(np.arange(len(names)) % 10), index=names)
    prox = pd.Series(np.nan, index=j.index)
    for k in range(10):
        m = bp.fit(tr[fold.reindex(tr.index).to_numpy() != k], features); te = fold.reindex(j.index).to_numpy() == k
        prox[te] = bp.predict(m, j.loc[te, "bsr_pos"], j.loc[te, "ht"], j.loc[te, "wt"])
    base = (j["proj_woba"] - c["lg"]) / hp.SC + j["pos_rate"] + c["repl"]
    old = j["proj_pa"] * base / c["rpw"]; new = j["proj_pa"] * (base + prox) / c["rpw"]
    y = j["pa26"] * ((j["woba26"] - lg26) / hp.SC + j["pos26"].fillna(0) / j["pa26"] + c["repl"]) / c["rpw"] + j["bsr26"] / c["rpw"]
    print(f"\n(b) WAR backtest -> 2026, {len(j)} hitters (>= 20 PA); position (incl. DH) known for {j['bsr_pos'].isin(bp.POS).mean():.0%}, DH {(j['bsr_pos'] == 'DH').sum()}, height for {j['ht'].notna().mean():.0%}; "
          f"proxy SD {600 * prox.std():.2f} runs / 600 PA")
    rows = [{"target": "realised WAR incl. baserunning", **compare(y.to_numpy(), old.to_numpy(), new.to_numpy(), j["pa26"].to_numpy(), j.index.to_numpy(), n=2000)}]
    print(table(rows, 4))
    e0, e1 = old - y, new - y; grp = j["bsr_pos"].where(j["bsr_pos"].isin(bp.POS), "none")
    print("mean error (proj - actual) by position (2026 role, DH = >= 50% of PA), old -> new:\n" + pd.DataFrame({"n": grp.value_counts(), "old": e0.groupby(grp).mean(), "new": e1.groupby(grp).mean()}).round(3).to_string())


if __name__ == "__main__":
    part_a(); part_b(); part_b(("pos",))
